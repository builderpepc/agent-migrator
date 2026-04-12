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


def _get_chats_dir(project_path: Path) -> Optional[Path]:
    """
    Locate the Gemini CLI chats directory for the project.
    Uses SHA-256 hash lookup with a fallback to scanning .project_root files.
    """
    base_tmp = _gemini_dir() / "tmp"
    if not base_tmp.exists():
        return None

    # Primary: Compute hash
    project_hash = _get_project_hash(project_path)
    chats_dir = base_tmp / project_hash / "chats"
    if chats_dir.exists():
        return chats_dir

    # Fallback: Scan all directories for matching .project_root
    normalized_target = _normalize_path(str(project_path))
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
        # Gemini stores sessions as session-*.json directly in chats/
        # or sometimes in subdirectories.
        for json_file in chats_dir.glob("**/session-*.json"):
            try:
                # Basic metadata extraction without full load
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                with open(json_file, "r", encoding="utf-8") as f:
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

                messages = data.get("messages", [])
                message_count = sum(1 for m in messages if m.get("type") in ("user", "gemini"))

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
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages: List[MessageTurn] = []
        for msg in data.get("messages", []):
            msg_type = msg.get("type")
            timestamp_str = msg.get("timestamp")
            ts = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else None
            )

            # Gemini content is PartListUnion (often array of {text: ...})
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
                # Add thoughts as assistant text if present
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

                # Add tool calls
                for tc in msg.get("toolCalls", []):
                    # Gemini tool results are also PartListUnion
                    results = tc.get("result", [])
                    result_str = ""
                    if isinstance(results, str):
                        result_str = results
                    elif isinstance(results, list):
                        result_str = "".join(
                            part.get("text", "") for part in results if isinstance(part, dict)
                        )

                    messages.append(
                        ToolCallMessage(
                            name=tc.get("name", "unknown"),
                            input=tc.get("args", {}),
                            result=result_str,
                            timestamp=ts,
                        )
                    )

        # Map generic info to ConversationInfo
        mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
        info = ConversationInfo(
            id=conv_id,
            name=data.get("summary") or f"Session {data.get('sessionId', 'unknown')[:8]}",
            updated_at=datetime.fromisoformat(data["lastUpdated"].replace("Z", "+00:00"))
            if "lastUpdated" in data
            else mtime,
            created_at=datetime.fromisoformat(data["startTime"].replace("Z", "+00:00"))
            if "startTime" in data
            else mtime,
            message_count=len([m for m in messages if isinstance(m, TextMessage)]),
            size_bytes=json_file.stat().st_size,
            source_tool=self.tool_id,
        )

        return Conversation(
            info=info,
            turns=messages,
            model=data.get("model"),
        )

    def write_conversation(self, conv: Conversation, project_path: Path, **kwargs) -> str:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir:
            # Create the directory if it doesn't exist (need hash)
            project_hash = _get_project_hash(project_path)
            chats_dir = _gemini_dir() / "tmp" / project_hash / "chats"
            chats_dir.mkdir(parents=True, exist_ok=True)
            # Ensure .project_root is also there for fallback
            root_file = chats_dir.parent / ".project_root"
            if not root_file.exists():
                root_file.write_text(str(project_path.resolve()))

        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        messages = []
        for turn in conv.turns:
            ts_iso = (
                turn.timestamp.isoformat().replace("+00:00", "Z")
                if turn.timestamp
                else now_iso
            )
            msg_id = str(uuid.uuid4())

            if isinstance(turn, TextMessage):
                messages.append(
                    {
                        "id": msg_id,
                        "timestamp": ts_iso,
                        "type": "user" if turn.role == "user" else "gemini",
                        "content": [{"text": turn.text}],
                    }
                )
            elif isinstance(turn, ToolCallMessage):
                # Gemini groups tool calls into the preceding/following gemini message
                # For simplicity in migration, we create a 'gemini' message per tool call
                messages.append(
                    {
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
                )

        record = {
            "sessionId": session_id,
            "projectHash": _get_project_hash(project_path),
            "startTime": conv.info.created_at.isoformat().replace("+00:00", "Z"),
            "lastUpdated": now_iso,
            "messages": messages,
            "summary": conv.info.name,
            "model": conv.model or "gemini-2.0-flash",
        }

        filename = f"session-{datetime.now().strftime('%Y-%m-%dT%H-%M')}-{session_id[:8]}.json"
        dest_file = chats_dir / filename
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return filename

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir:
            return
        json_file = chats_dir / conv_id
        if json_file.exists():
            json_file.unlink()
