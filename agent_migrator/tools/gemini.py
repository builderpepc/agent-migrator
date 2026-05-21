from __future__ import annotations

import hashlib
import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    StandardToolName,
    TextMessage,
    ToolCallMessage,
    inject_exit_plan_mode,
)
from agent_migrator.tools.base import ToolAdapter

_GEMINI_STORAGE = Path.home() / ".gemini" / "tmp"

# Map Gemini native tool names → standard interchange names.
_GEMINI_TO_STANDARD: dict[str, str] = {
    "write_file":        StandardToolName.WRITE,
    "replace":           StandardToolName.EDIT,
    "run_shell_command": StandardToolName.BASH,
    "read_file":         StandardToolName.READ,
    "list_directory":    StandardToolName.GLOB,
}

# Map standard interchange names → Gemini native tool names.
_STANDARD_TO_GEMINI: dict[str, str] = {v: k for k, v in _GEMINI_TO_STANDARD.items()}

# Gemini-internal tools filtered during read; they have no portable meaning.
_GEMINI_INTERNAL_TOOLS: set[str] = {
    "enter_plan_mode",
    "exit_plan_mode",
    "update_topic",
}


def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _now_str(dt: datetime | None = None) -> str:
    d = dt or datetime.now(timezone.utc)
    return d.isoformat().replace("+00:00", "Z")


def _project_hash(project_path: Path) -> str:
    """Compute the Gemini project hash: SHA256 of the resolved (lowercased on Windows) absolute path."""
    resolved = str(project_path.resolve())
    if platform.system() == "Windows":
        resolved = resolved.lower()
    return hashlib.sha256(resolved.encode()).hexdigest()


def _apply_rewinding(records: list[dict]) -> list[dict]:
    """Apply $rewindTo markers, returning the canonical record sequence.

    A $rewindTo:{id} discards all records accumulated after the record whose
    `id` field matches. Records that appear later with duplicate IDs replace
    the earlier occurrence (they are the replayed canonical version).
    """
    result: list[dict] = []
    id_to_idx: dict[str, int] = {}

    for rec in records:
        if "$rewindTo" in rec:
            target_id = rec["$rewindTo"]
            if target_id in id_to_idx:
                cutoff = id_to_idx[target_id] + 1
                result = result[:cutoff]
                id_to_idx = {r.get("id"): i for i, r in enumerate(result) if r.get("id")}
        else:
            rec_id = rec.get("id")
            if rec_id and rec_id in id_to_idx:
                # Duplicate ID after a rewind — replace the earlier version.
                result[id_to_idx[rec_id]] = rec
            else:
                if rec_id:
                    id_to_idx[rec_id] = len(result)
                result.append(rec)

    return result


def _find_session_file(chats_dir: Path, conv_id: str) -> Path | None:
    """Return the largest JSONL file for the given sessionId (most complete)."""
    best: tuple[int, Path] | None = None
    for f in chats_dir.glob("*.jsonl"):
        try:
            with open(f, encoding="utf-8") as fh:
                meta = json.loads(fh.readline().strip())
            if meta.get("sessionId") == conv_id and meta.get("kind") == "main":
                size = f.stat().st_size
                if best is None or size > best[0]:
                    best = (size, f)
        except Exception:
            continue
    return best[1] if best else None


class GeminiAdapter(ToolAdapter):
    name = "Gemini"
    tool_id = "gemini"

    def _project_dir(self, project_path: Path) -> Path:
        return _GEMINI_STORAGE / project_path.name

    def is_available(self) -> bool:
        return _GEMINI_STORAGE.exists()

    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        chats_dir = self._project_dir(project_path) / "chats"
        if not chats_dir.exists():
            return []

        # Collect one entry per sessionId using the largest (most complete) file.
        by_session: dict[str, tuple[int, Path, dict]] = {}
        for f in chats_dir.glob("*.jsonl"):
            try:
                with open(f, encoding="utf-8") as fh:
                    meta = json.loads(fh.readline().strip())
                if "sessionId" not in meta or meta.get("kind") != "main":
                    continue
                sid = meta["sessionId"]
                size = f.stat().st_size
                if sid not in by_session or size > by_session[sid][0]:
                    by_session[sid] = (size, f, meta)
            except Exception:
                continue

        infos: list[ConversationInfo] = []
        for sid, (size, f, meta) in by_session.items():
            try:
                with open(f, encoding="utf-8") as fh:
                    lines = [l.strip() for l in fh if l.strip()]

                start = _parse_ts(meta.get("startTime", "")) or datetime.now(timezone.utc)
                updated = _parse_ts(meta.get("lastUpdated", "")) or start
                msg_count = sum(
                    1 for l in lines
                    if '"type":"user"' in l or '"type": "user"' in l
                )

                # Use the last update_topic title as the display name.
                name = sid[:8]
                first_user_text = ""
                for line in lines:
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "user":
                            content = obj.get("content", [])
                            if isinstance(content, list) and content:
                                first_user_text = first_user_text or content[0].get("text", "")
                        if obj.get("toolCalls"):
                            for tc in obj["toolCalls"]:
                                if tc.get("name") == "update_topic":
                                    t = tc.get("args", {}).get("title", "")
                                    if t:
                                        name = t
                    except Exception:
                        continue
                if name == sid[:8] and first_user_text:
                    name = first_user_text[:50]

                infos.append(ConversationInfo(
                    id=sid,
                    name=name,
                    updated_at=updated,
                    created_at=start,
                    message_count=msg_count,
                    size_bytes=size,
                    source_tool="gemini",
                ))
            except Exception:
                continue

        infos.sort(key=lambda x: x.updated_at, reverse=True)
        return infos

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        proj_dir = self._project_dir(project_path)
        jsonl_file = _find_session_file(proj_dir / "chats", conv_id)
        if jsonl_file is None:
            raise FileNotFoundError(f"No Gemini session found for id={conv_id}")

        with open(jsonl_file, encoding="utf-8") as fh:
            raw_records = [json.loads(l) for l in fh if l.strip()]

        records = _apply_rewinding(raw_records)

        # Load plan content from the plans directory.
        plan_content: str | None = None
        plans_dir = proj_dir / conv_id / "plans"
        if plans_dir.exists():
            plan_files = sorted(plans_dir.glob("*.md"))
            if plan_files:
                plan_content = plan_files[0].read_text(encoding="utf-8")

        turns: list = []
        _plan_injected = False
        display_name = conv_id[:8]
        first_user_text = ""

        for rec in records:
            rtype = rec.get("type")
            ts = _parse_ts(rec.get("timestamp", ""))

            if rtype == "user":
                content = rec.get("content", [])
                text = (
                    content[0].get("text", "") if isinstance(content, list) and content
                    else content if isinstance(content, str)
                    else ""
                )
                if not text:
                    continue
                if not first_user_text:
                    first_user_text = text
                turns.append(TextMessage(role="user", text=text, timestamp=ts))

            elif rtype == "gemini":
                tool_calls = rec.get("toolCalls", [])
                content = rec.get("content", "")

                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("args", {}) or {}

                    # Capture last update_topic title as display name.
                    if tc_name == "update_topic":
                        t = tc_args.get("title", "")
                        if t:
                            display_name = t
                        continue

                    if tc_name == "enter_plan_mode":
                        if plan_content and not _plan_injected:
                            turns.append(TextMessage(
                                role="assistant",
                                text=f"<proposed_plan>\n{plan_content}\n</proposed_plan>",
                                timestamp=ts,
                            ))
                            _plan_injected = True
                        continue

                    if tc_name in _GEMINI_INTERNAL_TOOLS:
                        continue

                    # Skip write_file calls that write to the session plans dir.
                    file_path_arg = tc_args.get("file_path", "")
                    if tc_name == "write_file" and "plans" in file_path_arg and conv_id in file_path_arg:
                        continue

                    result_str = ""
                    result_list = tc.get("result", [])
                    if result_list:
                        resp = result_list[0].get("functionResponse", {}).get("response", {})
                        result_str = resp.get("output", "")

                    std_name = _GEMINI_TO_STANDARD.get(tc_name, tc_name)

                    if std_name == StandardToolName.WRITE:
                        cc_input: dict = {
                            "file_path": file_path_arg,
                            "content": tc_args.get("content", ""),
                        }
                    elif std_name == StandardToolName.EDIT:
                        cc_input = {
                            "file_path": file_path_arg,
                            "old_string": tc_args.get("old_string", ""),
                            "new_string": tc_args.get("new_string", ""),
                        }
                    elif std_name == StandardToolName.BASH:
                        cc_input = {"command": tc_args.get("command", "")}
                    elif std_name == StandardToolName.READ:
                        cc_input = {"file_path": file_path_arg}
                    elif std_name == StandardToolName.GLOB:
                        dir_path = tc_args.get("dir_path", ".")
                        cc_input = {"pattern": dir_path.rstrip("/\\") + "/**"}
                    else:
                        cc_input = tc_args

                    turns.append(ToolCallMessage(
                        name=std_name,
                        input=cc_input,
                        result=result_str,
                        timestamp=ts,
                    ))

                # Emit assistant text only for non-tool-call turns with content.
                if not tool_calls and isinstance(content, str) and content.strip():
                    turns.append(TextMessage(role="assistant", text=content, timestamp=ts))

        if plan_content:
            turns = inject_exit_plan_mode(turns, plan_content)

        if display_name == conv_id[:8] and first_user_text:
            display_name = first_user_text[:50]

        info = ConversationInfo(
            id=conv_id,
            name=display_name,
            updated_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            message_count=sum(1 for t in turns if isinstance(t, TextMessage)),
            size_bytes=jsonl_file.stat().st_size,
            source_tool="gemini",
        )

        return Conversation(info=info, turns=turns, plan_content=plan_content)

    def write_conversation(
        self,
        conv: Conversation,
        project_path: Path,
        *,
        use_local_backend: bool = False,
    ) -> str:
        proj_dir = self._project_dir(project_path)
        new_session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        ts_iso = _now_str(now)
        ts_file = now.strftime("%Y-%m-%dT%H-%M")
        short_id = new_session_id[:8]
        proj_hash = _project_hash(project_path)

        chats_dir = proj_dir / "chats"
        chats_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = chats_dir / f"session-{ts_file}-{short_id}.jsonl"

        # Write plan file if present.
        plan_content = conv.plan_content
        plan_filename: str | None = None
        if plan_content:
            plans_dir = proj_dir / new_session_id / "plans"
            plans_dir.mkdir(parents=True, exist_ok=True)
            plan_filename = "migrated-plan.md"
            (plans_dir / plan_filename).write_text(plan_content, encoding="utf-8")

        # Ensure ExitPlanMode is in turns (required for inject_exit_plan_mode to work).
        turns = list(conv.turns)
        if plan_content:
            turns = inject_exit_plan_mode(turns, plan_content)

        model = conv.model or "gemini-2.5-pro-preview"

        def _empty_tokens() -> dict:
            return {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "tool": 0, "total": 0}

        def _gemini_record(rec_id: str, ts: str, tool_calls: list, content: str = "") -> dict:
            return {
                "id": rec_id,
                "timestamp": ts,
                "type": "gemini",
                "content": content,
                "thoughts": [],
                "toolCalls": tool_calls,
                "tokens": _empty_tokens(),
                "model": model,
            }

        def _tc_result(name: str, tc_id: str, output: str) -> list:
            return [{"functionResponse": {"id": tc_id, "name": name, "response": {"output": output}}}]

        lines: list[str] = []
        log_entries: list[dict] = []
        msg_idx = 0

        # Session metadata (first line).
        lines.append(json.dumps({
            "sessionId": new_session_id,
            "projectHash": proj_hash,
            "startTime": ts_iso,
            "lastUpdated": ts_iso,
            "kind": "main",
        }))

        for turn in turns:
            rec_id = str(uuid.uuid4())
            turn_ts = _now_str(turn.timestamp)

            if isinstance(turn, TextMessage):
                if turn.role == "user":
                    lines.append(json.dumps({
                        "id": rec_id,
                        "timestamp": turn_ts,
                        "type": "user",
                        "content": [{"text": turn.text}],
                    }))
                    log_entries.append({
                        "sessionId": new_session_id,
                        "messageId": msg_idx,
                        "type": "user",
                        "message": turn.text,
                        "timestamp": turn_ts,
                    })
                    msg_idx += 1
                else:
                    # Assistant text message.
                    lines.append(json.dumps(_gemini_record(rec_id, turn_ts, [], turn.text)))

            elif isinstance(turn, ToolCallMessage):
                if turn.name == StandardToolName.EXIT_PLAN_MODE:
                    plan_text = turn.input.get("plan", plan_content or "")
                    plan_fn = plan_filename or "plan.md"
                    plan_fp = str(proj_dir / new_session_id / "plans" / plan_fn)

                    # enter_plan_mode
                    ep_id = str(uuid.uuid4())
                    ep_tc_id = f"enter_plan_mode_{uuid.uuid4().hex[:16]}_0"
                    lines.append(json.dumps(_gemini_record(ep_id, turn_ts, [{
                        "id": ep_tc_id, "name": "enter_plan_mode",
                        "args": {"reason": "Presenting migrated plan."},
                        "result": _tc_result("enter_plan_mode", ep_tc_id, "Entered plan mode."),
                        "status": "success", "timestamp": turn_ts, "displayName": "EnterPlanMode",
                    }])))

                    # write_file (plan)
                    wf_id = str(uuid.uuid4())
                    wf_tc_id = f"write_file_{uuid.uuid4().hex[:16]}_0"
                    lines.append(json.dumps(_gemini_record(wf_id, turn_ts, [{
                        "id": wf_tc_id, "name": "write_file",
                        "args": {"file_path": plan_fp, "content": plan_text},
                        "result": _tc_result("write_file", wf_tc_id, f"Successfully wrote {plan_fn}."),
                        "status": "success", "timestamp": turn_ts, "displayName": "WriteFile",
                    }])))

                    # exit_plan_mode
                    xp_tc_id = f"exit_plan_mode_{uuid.uuid4().hex[:16]}_0"
                    lines.append(json.dumps(_gemini_record(rec_id, turn_ts, [{
                        "id": xp_tc_id, "name": "exit_plan_mode",
                        "args": {"plan_filename": plan_fn},
                        "result": _tc_result("exit_plan_mode", xp_tc_id, "Exited plan mode."),
                        "status": "success", "timestamp": turn_ts, "displayName": "ExitPlanMode",
                    }])))

                else:
                    gemini_name = _STANDARD_TO_GEMINI.get(turn.name, turn.name)

                    if turn.name == StandardToolName.WRITE:
                        tc_args: dict = {
                            "file_path": turn.input.get("file_path", ""),
                            "content": turn.input.get("content", ""),
                        }
                    elif turn.name == StandardToolName.EDIT:
                        tc_args = {
                            "file_path": turn.input.get("file_path", ""),
                            "old_string": turn.input.get("old_string", ""),
                            "new_string": turn.input.get("new_string", ""),
                            "allow_multiple": False,
                        }
                    elif turn.name == StandardToolName.BASH:
                        tc_args = {
                            "command": turn.input.get("command", ""),
                            "description": "",
                        }
                    elif turn.name == StandardToolName.READ:
                        tc_args = {"file_path": turn.input.get("file_path", "")}
                    elif turn.name == StandardToolName.GLOB:
                        pattern = turn.input.get("pattern", ".")
                        tc_args = {"dir_path": pattern.rstrip("/**")}
                    else:
                        tc_args = turn.input or {}

                    result_str = turn.result if isinstance(turn.result, str) else json.dumps(turn.result)
                    tc_id = f"{gemini_name}_{uuid.uuid4().hex[:16]}_0"

                    lines.append(json.dumps(_gemini_record(rec_id, turn_ts, [{
                        "id": tc_id, "name": gemini_name,
                        "args": tc_args,
                        "result": _tc_result(gemini_name, tc_id, result_str),
                        "status": "success", "timestamp": turn_ts,
                        "displayName": turn.name,
                    }])))

        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Update logs.json with user-only entries.
        logs_path = proj_dir / "logs.json"
        try:
            existing = json.loads(logs_path.read_text(encoding="utf-8")) if logs_path.exists() else []
        except Exception:
            existing = []
        logs_path.write_text(
            json.dumps(existing + log_entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write .project_root if absent.
        root_file = proj_dir / ".project_root"
        if not root_file.exists():
            root_file.write_text(str(project_path), encoding="utf-8")

        return new_session_id

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        proj_dir = self._project_dir(project_path)
        chats_dir = proj_dir / "chats"
        for f in chats_dir.glob("*.jsonl"):
            try:
                with open(f, encoding="utf-8") as fh:
                    meta = json.loads(fh.readline().strip())
                if meta.get("sessionId") == conv_id:
                    f.unlink(missing_ok=True)
            except Exception:
                continue
        session_dir = proj_dir / conv_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
