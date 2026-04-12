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
    """Normalizes a path for reliable comparison."""
    resolved = os.path.abspath(p)
    normalized = resolved.replace("\\", "/")
    return normalized.lower()


def _get_project_hash(project_root: Path) -> str:
    """Computes the project hash matching Gemini CLI v0.37.1 on Windows."""
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
    project_id = _get_project_id_from_registry(project_root) or _get_project_hash(project_root)
    chats_dir = base_tmp / project_id / "chats"
    if chats_dir.exists():
        return chats_dir
    target_norm = _normalize_path(str(project_root))
    try:
        for p in base_tmp.iterdir():
            if not p.is_dir(): continue
            root_file = p / ".project_root"
            if root_file.exists():
                try:
                    if _normalize_path(root_file.read_text().strip()) == target_norm:
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
                message_count = sum(1 for m in data.get("messages", []) if m.get("type") in ("user", "gemini")) if not is_jsonl else 0
                results.append(ConversationInfo(
                    id=str(json_file.relative_to(chats_dir)),
                    name=data.get("summary") or f"Session {session_id[:8]}",
                    updated_at=updated_at, created_at=created_at,
                    message_count=message_count, size_bytes=json_file.stat().st_size,
                    source_tool=self.tool_id,
                ))
            except Exception: continue
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir: raise FileNotFoundError(f"Missing Gemini directory")
        json_file = chats_dir / conv_id
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages: List[MessageTurn] = []
        for msg in data.get("messages", []):
            self._parse_message_record(msg, messages)
        info = ConversationInfo(
            id=conv_id, name=data.get("summary", "unknown"),
            updated_at=datetime.fromisoformat(data["lastUpdated"].replace("Z", "+00:00")),
            created_at=datetime.fromisoformat(data["startTime"].replace("Z", "+00:00")),
            message_count=len([m for m in messages if isinstance(m, TextMessage)]),
            size_bytes=json_file.stat().st_size, source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=messages, model=data.get("model"))

    def _parse_message_record(self, msg: Dict[str, Any], messages: List[MessageTurn]) -> None:
        msg_type = msg.get("type")
        ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")) if "timestamp" in msg else None
        content_parts = msg.get("content", [])
        text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict)) if isinstance(content_parts, list) else str(content_parts)
        if msg_type == "user":
            messages.append(TextMessage(role="user", text=text, timestamp=ts))
        elif msg_type == "gemini":
            if text: messages.append(TextMessage(role="assistant", text=text, timestamp=ts))
            for tc in msg.get("toolCalls", []):
                raw_name = tc.get("name", "unknown")
                raw_args = tc.get("args", {})
                
                # Normalize arguments to canonical snake_case intermediary format
                args = {}
                for k, v in raw_args.items():
                    nk = k
                    if k == "filePath": nk = "file_path"
                    elif k == "oldString": nk = "old_string"
                    elif k == "newString": nk = "new_string"
                    elif k == "replaceAll": nk = "replace_all"
                    elif k == "dirPath": nk = "dir_path"
                    args[nk] = v

                # Map back to intermediary tool names
                norm_name = raw_name.lower()
                if norm_name == "read_file": 
                    name = "Read"
                    args["file_path"] = args.get("file_path", args.get("filePath", ""))
                elif norm_name == "replace": 
                    name = "Edit"
                    args["file_path"] = args.get("file_path", args.get("filePath", ""))
                elif norm_name == "write_file": 
                    name = "Write"
                    args["file_path"] = args.get("file_path", args.get("filePath", ""))
                elif norm_name == "run_shell_command": 
                    name = "Bash"
                elif norm_name == "grep_search": 
                    name = "Grep"
                    args["file_path"] = args.get("dir_path", args.get("path", args.get("file_path", ".")))
                elif norm_name == "glob": name = "Glob"
                elif norm_name in ("invoke_agent", "codebase_investigator", "generalist"): name = "Agent"
                elif norm_name == "exitplanmode": name = "ExitPlanMode"
                elif norm_name == "taskcreate": name = "TaskCreate"
                elif norm_name == "taskupdate": name = "TaskUpdate"
                elif norm_name == "tasklist": name = "TaskList"
                elif norm_name == "websearch": name = "WebSearch"
                elif norm_name == "webfetch": name = "WebFetch"
                else: 
                    name = raw_name[0].upper() + raw_name[1:] if raw_name else "Unknown"

                # Extract result
                res_val = tc.get("resultDisplay")
                if not res_val:
                    res_blocks = tc.get("result", [])
                    if isinstance(res_blocks, list):
                        combined_parts = []
                        for block in res_blocks:
                            if not isinstance(block, dict): continue
                            if "functionResponse" in block:
                                fr = block["functionResponse"]
                                resp = fr.get("response", {})
                                if isinstance(resp, dict):
                                    # Handle Bash output/error specifically for tags
                                    if name == "Bash":
                                        stdout = resp.get("output", "")
                                        stderr = resp.get("error", "")
                                        tagged = ""
                                        if stdout: tagged += f"<bash-stdout>{stdout}</bash-stdout>"
                                        if stderr: tagged += f"<bash-stderr>{stderr}</bash-stderr>"
                                        if tagged: combined_parts.append(tagged)
                                    else:
                                        stdout = resp.get("output", "")
                                        stderr = resp.get("error", "")
                                        part = "\n".join(filter(None, [stdout, stderr]))
                                        if part: combined_parts.append(part)
                                else:
                                    combined_parts.append(str(resp))
                            elif "parts" in block:
                                parts = block["parts"]
                                if isinstance(parts, list):
                                    p_text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
                                    if p_text: combined_parts.append(p_text)
                        res_val = "\n".join(combined_parts)
                    if not res_val:
                        res_val = str(res_blocks) if res_blocks else ""
                
                if isinstance(res_val, dict):
                    res_val = res_val.get("newContent") or res_val.get("fileDiff") or res_val.get("output") or res_val.get("summary") or json.dumps(res_val)

                result_text = str(res_val)
                # Clean up Gemini-specific shell markers
                if result_text.startswith("Output: "):
                    result_text = result_text[len("Output: "):].strip()
                if "Process Group PGID:" in result_text:
                    import re
                    result_text = re.sub(r"\n?Process Group PGID: \d+", "", result_text).strip()

                # For Bash tools, wrap in tags in the intermediary format.
                # Native CC sessions use these tags in string-based user content.
                if name == "Bash" and not result_text.startswith("<bash-"):
                    result_text = f"<bash-stdout>{result_text}</bash-stdout>"

                messages.append(ToolCallMessage(name=name, input=args, result=result_text, timestamp=ts))

    def write_conversation(self, conv: Conversation, project_path: Path, **kwargs) -> str:
        project_root = _find_project_root(project_path)
        chats_dir = _get_chats_dir(project_path)
        if not chats_dir:
            project_id = _get_project_id_from_registry(project_root) or _get_project_hash(project_root)
            chats_dir = _gemini_dir() / "tmp" / project_id / "chats"
            chats_dir.mkdir(parents=True, exist_ok=True)
            (chats_dir.parent / ".project_root").write_text(str(project_root.resolve()))

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        messages = []
        current_turn_tools = []
        
        def flush_tools():
            if not current_turn_tools: return
            messages.append({
                "id": str(uuid.uuid4()), "timestamp": current_turn_tools[0][1], "type": "gemini", "content": [],
                "toolCalls": [t[0] for t in current_turn_tools]
            })
            current_turn_tools.clear()

        for turn in conv.turns:
            ts_iso = turn.timestamp.isoformat() if turn.timestamp else now_iso
            if isinstance(turn, TextMessage):
                flush_tools()
                messages.append({
                    "id": str(uuid.uuid4()), "timestamp": ts_iso,
                    "type": "user" if turn.role == "user" else "gemini",
                    "content": [{"text": turn.text}]
                })
            elif isinstance(turn, ToolCallMessage):
                raw_name = turn.name.lower()
                args = turn.input
                name = raw_name
                disp_name = turn.name
                desc = ""
                res_display: Any = turn.result

                if raw_name in ("bash", "run_shell_command"):
                    name, disp_name = "run_shell_command", "Shell"
                    cmd = args.get("command", "")
                    desc = f"{cmd} [current working directory {project_root}]"
                elif raw_name in ("edit", "replace"):
                    name, disp_name = "replace", "Edit"
                    file_path = args.get("file_path", "unknown")
                    desc = file_path
                    
                    old_str = args.get("old_string", "")
                    new_str = args.get("new_string", "")
                    
                    if old_str or new_str:
                        import difflib
                        old_lines = old_str.splitlines(keepends=True)
                        new_lines = new_str.splitlines(keepends=True)
                        if old_lines and not old_lines[-1].endswith("\n"): old_lines[-1] += "\n"
                        if new_lines and not new_lines[-1].endswith("\n"): new_lines[-1] += "\n"
                        
                        diff_list = list(difflib.unified_diff(
                            old_lines, new_lines, 
                            fromfile=f"{file_path}\tOriginal", 
                            tofile=f"{file_path}\tWritten", 
                            n=3
                        ))
                        diff_text = "".join(diff_list)
                    else:
                        diff_text = turn.result

                    res_display = {
                        "fileDiff": diff_text,
                        "fileName": Path(file_path).name,
                        "filePath": str(project_root / file_path),
                        "originalContent": old_str,
                        "newContent": new_str or turn.result,
                        "diffStat": {
                            "model_added_lines": len([l for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]),
                            "model_removed_lines": len([l for l in diff_text.splitlines() if l.startswith("-") and not l.startswith("---")]),
                            "model_added_chars": len(new_str),
                            "model_removed_chars": len(old_str),
                            "user_added_lines": 0, "user_removed_lines": 0,
                            "user_added_chars": 0, "user_removed_chars": 0
                        },
                        "isNewFile": False
                    }
                elif raw_name in ("write", "write_file"):
                    name, disp_name = "write_file", "WriteFile"
                    file_path = args.get("file_path", "unknown")
                    content = args.get("content", "")
                    desc = f"Writing to {file_path}"
                    
                    lines = content.splitlines()
                    diff_text = (
                        f"--- {file_path}\tOriginal\n"
                        f"+++ {file_path}\tWritten\n"
                        f"@@ -1,1 +1,{len(lines)} @@\n"
                    ) + "\n".join(f"+{l}" for l in lines)

                    res_display = {
                        "fileDiff": diff_text,
                        "fileName": Path(file_path).name,
                        "filePath": str(project_root / file_path),
                        "originalContent": "",
                        "newContent": content,
                        "diffStat": {
                            "model_added_lines": len(lines),
                            "model_removed_lines": 0,
                            "model_added_chars": len(content),
                            "model_removed_chars": 0,
                            "user_added_lines": 0, "user_removed_lines": 0,
                            "user_added_chars": 0, "user_removed_chars": 0
                        },
                        "isNewFile": True
                    }
                elif raw_name in ("read", "read_file"):
                    name, disp_name = "read_file", "ReadFile"
                    file_path = args.get("file_path", "unknown")
                    desc = file_path
                elif raw_name in ("grep", "grep_search"):
                    name, disp_name = "grep_search", "GrepSearch"
                    pattern = args.get("pattern", "")
                    path = args.get("dir_path", args.get("file_path", "."))
                    desc = f"'{pattern}' in {path}"
                elif raw_name in ("glob",):
                    name, disp_name = "glob", "Glob"
                    pattern = args.get("pattern", "")
                    desc = pattern
                elif raw_name in ("agent", "invoke_agent", "codebase_investigator", "generalist"):
                    name, disp_name = "invoke_agent", "Agent"
                    desc = args.get("objective", args.get("request", ""))
                    res_display = {"isSubagentProgress": True, "agentName": "Subagent", "recentActivity": [{"id": str(uuid.uuid4()), "type": "thought", "content": turn.result, "status": "completed"}], "state": "completed", "result": turn.result}

                call_id = f"{name}_{int(datetime.now().timestamp() * 1000)}_0"
                current_turn_tools.append(({
                    "id": call_id, "name": name, "displayName": disp_name, "description": desc, "args": args,
                    "result": [{"functionResponse": {"id": call_id, "name": name, "response": {"output": turn.result}}}],
                    "resultDisplay": res_display, "status": "success", "timestamp": ts_iso, "renderOutputAsMarkdown": (name != "run_shell_command")
                }, ts_iso))
        flush_tools()

        record = {
            "sessionId": str(uuid.uuid4()), "projectHash": _get_project_hash(project_root),
            "startTime": conv.info.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "lastUpdated": now_iso, "messages": messages, "summary": conv.info.name, "model": conv.model or "gemini-2.0-flash", "kind": "main"
        }
        filename = f"session-{datetime.now().strftime('%Y-%m-%dT%H-%M')}-{record['sessionId'][:8]}.json"
        with open(chats_dir / filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return filename

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        chats_dir = _get_chats_dir(project_path)
        if chats_dir:
            json_file = chats_dir / conv_id
            if json_file.exists(): json_file.unlink()
