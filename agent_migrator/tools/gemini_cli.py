from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    MessageTurn,
    TextMessage,
    ToolCallMessage,
)
from agent_migrator.tools.base import ToolAdapter


def _gemini_dir() -> Path:
    """Return the base Gemini CLI directory (~/.gemini)."""
    # Follows Gemini CLI's logic: check GEMINI_CLI_HOME or fallback to home
    env_home = os.environ.get("GEMINI_CLI_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".gemini"


def _normalize_path(p: str) -> str:
    """
    Normalizes a path for reliable comparison across platforms,
    matching Gemini CLI's internal normalizePath function.
    """
    resolved = os.path.abspath(p)
    normalized = resolved.replace("\\", "/")
    # Gemini CLI treats Windows and macOS as case-insensitive for hashing
    is_case_insensitive = sys.platform in ("win32", "darwin")
    return normalized.lower() if is_case_insensitive else normalized


def _get_project_hash(project_root: Path) -> str:
    """
    Computes the project hash matching Gemini CLI's internal getProjectHash.
    """
    normalized = _normalize_path(str(project_root))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _find_project_root(path: Path) -> Path:
    """
    Finds the project root by climbing up for .git or .gemini,
    mimicking Gemini CLI's discovery logic.
    """
    current = path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".gemini").exists():
            return parent
    return current


def _get_project_id_from_registry(project_root: Path) -> Optional[str]:
    """
    Reads ~/.gemini/projects.json to find the shortId assigned to this project.
    """
    registry_path = _gemini_dir() / "projects.json"
    if not registry_path.exists():
        return None
    
    try:
        data = json.loads(registry_path.read_text())
        projects = data.get("projects", {})
        normalized_root = _normalize_path(str(project_root))
        return projects.get(normalized_root)
    except Exception:
        return None


def _get_chats_dir(project_path: Path) -> Optional[Path]:
    """
    Locate the Gemini CLI chats directory for the project.
    Uses projects.json registry lookup with fallbacks.
    """
    base_tmp = _gemini_dir() / "tmp"
    if not base_tmp.exists():
        return None

    project_root = _find_project_root(project_path)
    
    # Primary: Check projects.json registry for the shortId/slug
    project_id = _get_project_id_from_registry(project_root)
    if project_id:
        chats_dir = base_tmp / project_id / "chats"
        if chats_dir.exists():
            return chats_dir

    # Secondary: Fallback to SHA-256 hash
    project_hash = _get_project_hash(project_root)
    chats_dir = base_tmp / project_hash / "chats"
    if chats_dir.exists():
        return chats_dir

    # Tertiary: Scan all directories for matching .project_root file
    normalized_target = _normalize_path(str(project_root))
    try:
        for p in base_tmp.iterdir():
            if not p.is_dir():
                continue
            root_file = p / ".project_root"
            if root_file.exists():
                try:
                    stored_root = root_file.read_text().strip()
                    if _normalize_path(stored_root) == normalized_target:
                        return p / "chats"
                except Exception:
                    continue
    except Exception:
        pass

    return None


class GeminiCliAdapter(ToolAdapter):
    name = "Gemini CLI"
    tool_id = "gemini_cli"

    def is_available(self) -> bool:
        return _gemini_dir().exists()

    def list_conversations(self, project_path: Path) -> List[ConversationInfo]:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir or not chats_dir.exists():
            return []

        results: List[ConversationInfo] = []
        # Gemini stores sessions as session-*.jsonl (new) or session-*.json (legacy)
        for json_file in chats_dir.glob("**/session-*.json*"):
            try:
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                
                is_jsonl = json_file.suffix == ".jsonl"
                with open(json_file, "r", encoding="utf-8") as f:
                    if is_jsonl:
                        # First line is always the metadata record
                        first_line = f.readline()
                        if not first_line:
                            continue
                        data = json.loads(first_line)
                    else:
                        data = json.load(f)

                session_id = data.get("sessionId", json_file.stem)
                start_time_str = data.get("startTime")
                updated_time_str = data.get("lastUpdated")

                created_at = (
                    datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    if start_time_str
                    else mtime
                )
                updated_at = (
                    datetime.fromisoformat(updated_time_str.replace("Z", "+00:00"))
                    if updated_time_str
                    else mtime
                )

                # Count messages for the TUI
                message_count = 0
                if not is_jsonl:
                    messages = data.get("messages", [])
                    message_count = sum(1 for m in messages if m.get("type") in ("user", "gemini"))
                else:
                    # Quick estimation for JSONL to keep list_conversations fast
                    message_count = 0 

                results.append(
                    ConversationInfo(
                        id=str(json_file.relative_to(chats_dir)),
                        name=data.get("summary") or f"Session {session_id[:8]}",
                        updated_at=updated_at,
                        created_at=created_at,
                        message_count=message_count,
                        size_bytes=json_file.stat().st_size,
                        source_tool=self.tool_id,
                    )
                )
            except Exception:
                continue

        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir:
            raise FileNotFoundError(f"Could not locate Gemini CLI directory for {project_path}")

        json_file = chats_dir / conv_id
        is_jsonl = json_file.suffix == ".jsonl"
        
        messages: List[MessageTurn] = []
        metadata: Dict[str, Any] = {}

        with open(json_file, "r", encoding="utf-8") as f:
            if is_jsonl:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "$rewindTo" in record:
                        pass 
                    elif "$set" in record:
                        metadata.update(record["$set"])
                    elif "sessionId" in record and "projectHash" in record:
                        metadata.update(record)
                    elif "id" in record and "timestamp" in record:
                        self._parse_message_record(record, messages)
            else:
                data = json.load(f)
                metadata = data
                for msg in data.get("messages", []):
                    self._parse_message_record(msg, messages)

        mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
        info = ConversationInfo(
            id=conv_id,
            name=metadata.get("summary") or f"Session {metadata.get('sessionId', 'unknown')[:8]}",
            updated_at=datetime.fromisoformat(metadata["lastUpdated"].replace("Z", "+00:00"))
            if "lastUpdated" in metadata
            else mtime,
            created_at=datetime.fromisoformat(metadata["startTime"].replace("Z", "+00:00"))
            if "startTime" in metadata
            else mtime,
            message_count=len([m for m in messages if isinstance(m, TextMessage)]),
            size_bytes=json_file.stat().st_size,
            source_tool=self.tool_id,
        )

        return Conversation(
            info=info,
            turns=messages,
            model=metadata.get("model"),
        )

    def _parse_message_record(self, msg: Dict[str, Any], messages: List[MessageTurn]) -> None:
        msg_type = msg.get("type")
        timestamp_str = msg.get("timestamp")
        ts = (
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if timestamp_str
            else None
        )

        content_parts = msg.get("content", [])
        if isinstance(content_parts, str):
            text = content_parts
        elif isinstance(content_parts, list):
            text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
        else:
            text = ""

        if msg_type == "user":
            messages.append(TextMessage(role="user", text=text, timestamp=ts))
        elif msg_type == "gemini":
            thoughts = msg.get("thoughts", [])
            thought_text = ""
            if thoughts:
                thought_text = "\n".join(
                    f"[Thought: {t.get('subject')}] {t.get('description')}"
                    for t in thoughts
                )

            if text or thought_text:
                full_text = text
                if thought_text:
                    full_text = f"{thought_text}\n\n{text}" if text else thought_text
                messages.append(TextMessage(role="assistant", text=full_text, timestamp=ts))

            for tc in msg.get("toolCalls", []):
                results = tc.get("result", [])
                result_str = ""
                if isinstance(results, list):
                    parts = []
                    for part in results:
                        if not isinstance(part, dict): continue
                        if "text" in part:
                            parts.append(part["text"])
                        elif "functionResponse" in part:
                            resp = part["functionResponse"].get("response")
                            parts.append(json.dumps(resp) if isinstance(resp, (dict, list)) else str(resp))
                    result_str = "".join(parts)
                elif isinstance(results, str):
                    result_str = results

                messages.append(
                    ToolCallMessage(
                        name=tc.get("name", "unknown"),
                        input=tc.get("args", {}),
                        result=result_str,
                        timestamp=ts,
                    )
                )

    def write_conversation(self, conv: Conversation, project_path: Path, **kwargs) -> str:
        project_root = _find_project_root(project_path)
        chats_dir = _get_chats_dir(project_path)
        
        if not chats_dir:
            project_id = _get_project_id_from_registry(project_root) or _get_project_hash(project_root)
            chats_dir = _gemini_dir() / "tmp" / project_id / "chats"
            chats_dir.mkdir(parents=True, exist_ok=True)
            
            root_file = chats_dir.parent / ".project_root"
            if not root_file.exists():
                root_file.write_text(_normalize_path(str(project_root)))

        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        filename = f"session-{datetime.now().strftime('%Y-%m-%dT%H-%M')}-{session_id[:8]}.jsonl"
        dest_file = chats_dir / filename
        
        with open(dest_file, "w", encoding="utf-8") as f:
            # 1. Initial metadata record
            metadata = {
                "sessionId": session_id,
                "projectHash": _get_project_hash(project_root),
                "startTime": conv.info.created_at.isoformat().replace("+00:00", "Z"),
                "lastUpdated": now_iso,
                "summary": conv.info.name,
                "model": conv.model or "gemini-2.0-flash",
                "kind": "main",
            }
            f.write(json.dumps(metadata) + "\n")

            # 2. Message records
            for turn in conv.turns:
                ts_iso = (
                    turn.timestamp.isoformat().replace("+00:00", "Z")
                    if turn.timestamp
                    else now_iso
                )
                msg_id = str(uuid.uuid4())

                if isinstance(turn, TextMessage):
                    msg = {
                        "id": msg_id,
                        "timestamp": ts_iso,
                        "type": "user" if turn.role == "user" else "gemini",
                        "content": [{"text": turn.text}],
                    }
                    f.write(json.dumps(msg) + "\n")
                elif isinstance(turn, ToolCallMessage):
                    msg = {
                        "id": msg_id,
                        "timestamp": ts_iso,
                        "type": "gemini",
                        "content": [],
                        "toolCalls": [
                            {
                                "id": f"tc-{uuid.uuid4().hex[:8]}",
                                "name": turn.name,
                                "args": turn.input,
                                "result": [{"text": turn.result}],
                                "status": "success",
                                "timestamp": ts_iso,
                            }
                        ],
                    }
                    f.write(json.dumps(msg) + "\n")

        return filename

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir:
            return
        json_file = chats_dir / conv_id
        if json_file.exists():
            json_file.unlink()
