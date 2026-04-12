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
    env_home = os.environ.get("GEMINI_CLI_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".gemini"


def _normalize_path(p: str) -> str:
    """
    Normalizes a path for reliable comparison, 
    matching Gemini CLI's ProjectRegistry logic (mostly).
    """
    resolved = os.path.abspath(p)
    # Gemini CLI's ProjectRegistry uses path.resolve().toLowerCase() on Windows
    # but does NOT replace backslashes in the registry keys.
    # We use forward slashes ONLY for internal comparison logic.
    normalized = resolved.replace("\\", "/")
    return normalized.lower()


def _get_project_hash(project_root: Path) -> str:
    """
    Computes the project hash matching Gemini CLI's internal getProjectHash.
    On Windows, this uses BACKSLASHES because Gemini CLI uses process.cwd().
    """
    # Gemini CLI 0.37.1 on Windows uses the raw path from path.resolve()
    # which preserves backslashes and case (though some parts might lowercase).
    # Based on empirical evidence, it uses the backslashed path.
    raw_path = str(project_root.resolve())
    return hashlib.sha256(raw_path.encode()).hexdigest()


def _find_project_root(path: Path) -> Path:
    """Finds the project root by climbing up for .git or .gemini."""
    current = path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".gemini").exists():
            return parent
    return current


def _get_project_id_from_registry(project_root: Path) -> Optional[str]:
    """Reads ~/.gemini/projects.json to find the shortId."""
    registry_path = _gemini_dir() / "projects.json"
    if not registry_path.exists():
        return None
    
    try:
        data = json.loads(registry_path.read_text())
        projects = data.get("projects", {})
        target_norm = _normalize_path(str(project_root))
        
        # Search keys robustly
        for key, val in projects.items():
            if _normalize_path(key) == target_norm:
                return val
    except Exception:
        pass
    return None


def _get_chats_dir(project_path: Path) -> Optional[Path]:
    """Locate the Gemini CLI chats directory."""
    base_tmp = _gemini_dir() / "tmp"
    if not base_tmp.exists():
        return None

    project_root = _find_project_root(project_path)
    
    # 1. Registry lookup (handles slugs and legacy hashes)
    project_id = _get_project_id_from_registry(project_root)
    if project_id:
        chats_dir = base_tmp / project_id / "chats"
        if chats_dir.exists():
            return chats_dir

    # 2. Direct hash lookup (backslashed)
    project_hash = _get_project_hash(project_root)
    chats_dir = base_tmp / project_hash / "chats"
    if chats_dir.exists():
        return chats_dir
    
    # 3. Fallback hash lookup (forward-slashed)
    alt_hash = hashlib.sha256(str(project_root.resolve()).replace("\\", "/").lower().encode()).hexdigest()
    chats_dir = base_tmp / alt_hash / "chats"
    if chats_dir.exists():
        return chats_dir

    # 4. Scan .project_root files
    target_norm = _normalize_path(str(project_root))
    try:
        for p in base_tmp.iterdir():
            if not p.is_dir(): continue
            root_file = p / ".project_root"
            if root_file.exists():
                try:
                    stored_root = root_file.read_text().strip()
                    if _normalize_path(stored_root) == target_norm:
                        return p / "chats"
                except Exception: continue
    except Exception: pass

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
        for json_file in chats_dir.glob("**/session-*.json*"):
            try:
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                is_jsonl = json_file.suffix == ".jsonl"
                
                with open(json_file, "r", encoding="utf-8") as f:
                    if is_jsonl:
                        line = f.readline()
                        if not line: continue
                        data = json.loads(line)
                    else:
                        data = json.load(f)

                session_id = data.get("sessionId", json_file.stem)
                created_at = datetime.fromisoformat(data["startTime"].replace("Z", "+00:00")) if "startTime" in data else mtime
                updated_at = datetime.fromisoformat(data["lastUpdated"].replace("Z", "+00:00")) if "lastUpdated" in data else mtime

                # Count messages
                message_count = 0
                if not is_jsonl:
                    message_count = sum(1 for m in data.get("messages", []) if m.get("type") in ("user", "gemini"))
                
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
            except Exception: continue

        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir: raise FileNotFoundError(f"Missing Gemini directory for {project_path}")

        json_file = chats_dir / conv_id
        is_jsonl = json_file.suffix == ".jsonl"
        
        messages: List[MessageTurn] = []
        metadata: Dict[str, Any] = {}

        with open(json_file, "r", encoding="utf-8") as f:
            if is_jsonl:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    if "$set" in record: metadata.update(record["$set"])
                    elif "sessionId" in record: metadata.update(record)
                    elif "type" in record: self._parse_message_record(record, messages)
            else:
                data = json.load(f)
                metadata = data
                for msg in data.get("messages", []):
                    self._parse_message_record(msg, messages)

        mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
        info = ConversationInfo(
            id=conv_id,
            name=metadata.get("summary") or f"Session {metadata.get('sessionId', 'unknown')[:8]}",
            updated_at=datetime.fromisoformat(metadata["lastUpdated"].replace("Z", "+00:00")) if "lastUpdated" in metadata else mtime,
            created_at=datetime.fromisoformat(metadata["startTime"].replace("Z", "+00:00")) if "startTime" in metadata else mtime,
            message_count=len([m for m in messages if isinstance(m, TextMessage)]),
            size_bytes=json_file.stat().st_size,
            source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=messages, model=metadata.get("model"))

    def _parse_message_record(self, msg: Dict[str, Any], messages: List[MessageTurn]) -> None:
        msg_type = msg.get("type")
        ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")) if "timestamp" in msg else None
        
        content_parts = msg.get("content", [])
        if isinstance(content_parts, str): text = content_parts
        elif isinstance(content_parts, list):
            text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
        else: text = ""

        if msg_type == "user":
            messages.append(TextMessage(role="user", text=text, timestamp=ts))
        elif msg_type == "gemini":
            thoughts = msg.get("thoughts", [])
            thought_text = "\n".join(f"[Thought: {t.get('subject')}] {t.get('description')}" for t in thoughts) if thoughts else ""
            if text or thought_text:
                full_text = f"{thought_text}\n\n{text}" if text and thought_text else (text or thought_text)
                messages.append(TextMessage(role="assistant", text=full_text, timestamp=ts))

            for tc in msg.get("toolCalls", []):
                res_parts = tc.get("result", [])
                res_str = "".join(p.get("text", "") for p in res_parts if isinstance(p, dict)) if isinstance(res_parts, list) else str(res_parts)
                messages.append(ToolCallMessage(name=tc.get("name", "unknown"), input=tc.get("args", {}), result=res_str, timestamp=ts))

    def write_conversation(self, conv: Conversation, project_path: Path, **kwargs) -> str:
        project_root = _find_project_root(project_path)
        chats_dir = _get_chats_dir(project_path)
        
        if not chats_dir:
            project_id = _get_project_id_from_registry(project_root) or _get_project_hash(project_root)
            chats_dir = _gemini_dir() / "tmp" / project_id / "chats"
            chats_dir.mkdir(parents=True, exist_ok=True)
            root_file = chats_dir.parent / ".project_root"
            if not root_file.exists(): root_file.write_text(str(project_root.resolve()))

        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        messages = []
        # Group consecutive tool calls into one gemini message
        current_turn_tools = []
        
        def flush_tools():
            if not current_turn_tools:
                return
            ts_iso = current_turn_tools[0][1]
            msg_id = str(uuid.uuid4())
            messages.append({
                "id": msg_id, "timestamp": ts_iso, "type": "gemini", "content": [],
                "toolCalls": [t[0] for t in current_turn_tools]
            })
            current_turn_tools.clear()

        for turn in conv.turns:
            ts_iso = turn.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if turn.timestamp else now_iso
            msg_id = str(uuid.uuid4())

            if isinstance(turn, TextMessage):
                flush_tools()
                messages.append({
                    "id": msg_id, "timestamp": ts_iso,
                    "type": "user" if turn.role == "user" else "gemini",
                    "content": [{"text": turn.text}]
                })
            elif isinstance(turn, ToolCallMessage):
                # Map specialized tools for Gemini CLI 0.37.1 UI
                name = turn.name
                args = turn.input
                result_display: Any = turn.result
                kind = "other"

                if name == "run_shell_command":
                    name = "run_shell_command"
                    kind = "execute"
                    result_display = [{"text": turn.result}]
                elif name in ("replace", "write_file"):
                    kind = "edit"
                    file_path = args.get("file_path", "unknown")
                    file_name = Path(file_path).name
                    diff_text = turn.result
                    if name == "write_file" and not diff_text.startswith("---"):
                        lines = diff_text.splitlines()
                        diff_text = f"--- /dev/null\n+++ {file_path}\n@@ -0,0 +1,{len(lines)} @@\n" + "\n".join(f"+{l}" for l in lines)
                    
                    name = "edit"
                    result_display = {
                        "fileDiff": diff_text,
                        "fileName": file_name,
                        "filePath": str(project_root / file_path),
                        "originalContent": "",
                        "newContent": ""
                    }
                elif name in ("codebase_investigator", "generalist"):
                    name = "agent"
                    kind = "agent"
                    result_display = {
                        "isSubagentProgress": True,
                        "agentName": "Subagent",
                        "recentActivity": [{
                            "id": str(uuid.uuid4()),
                            "type": "thought",
                            "content": turn.result,
                            "status": "completed"
                        }],
                        "state": "completed",
                        "result": turn.result
                    }
                elif name == "EnterPlanMode":
                    name = "enter_plan_mode"
                    kind = "plan"
                elif name == "ExitPlanMode":
                    name = "exit_plan_mode"
                    kind = "plan"

                tool_call = {
                    "id": f"tc-{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "args": args,
                    "result": result_display,
                    "status": "success",
                    "timestamp": ts_iso,
                    "kind": kind
                }
                current_turn_tools.append((tool_call, ts_iso))
        
        flush_tools()

        record = {
            "sessionId": session_id,
            "projectHash": _get_project_hash(project_root),
            "startTime": conv.info.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "lastUpdated": now_iso,
            "messages": messages,
            "summary": conv.info.name,
            "model": conv.model or "gemini-2.0-flash",
            "kind": "main"
        }

        # Write as .json (monolithic) for compatibility with Gemini CLI 0.37.1
        filename = f"session-{datetime.now().strftime('%Y-%m-%dT%H-%M')}-{session_id[:8]}.json"
        dest_file = chats_dir / filename
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return filename

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir: return
        json_file = chats_dir / conv_id
        if json_file.exists(): json_file.unlink()
