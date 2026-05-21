from __future__ import annotations

import json
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    StandardToolName,
    TextMessage,
    ToolCallMessage,
)
from agent_migrator.tools.base import ToolAdapter

_PROPOSED_PLAN_DISPLAY_RE = re.compile(
    r"\s*<proposed_plan>(.*?)</proposed_plan>\s*", re.DOTALL | re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

def _codex_dir() -> Path:
    return Path.home() / ".codex"

def _sessions_dir() -> Path:
    return _codex_dir() / "sessions"

def _archived_sessions_dir() -> Path:
    return _codex_dir() / "archived_sessions"


# ---------------------------------------------------------------------------
# Read-path filters
# ---------------------------------------------------------------------------

# Outer record types to skip entirely
_SKIP_OUTER_TYPES = {"event_msg", "turn_context"}

# response_item payload types to skip
_SKIP_PAYLOAD_TYPES = {"reasoning", "compacted"}

# Message roles that carry system/framework injections, not user content
_SKIP_ROLES = {"developer", "system"}

# Regex to detect system-injected user message content
_CODEX_SYSTEM_CONTENT_RE = re.compile(
    r"^<(?:environment_context|permissions\s+instructions|collaboration_mode)[\s>]",
    re.IGNORECASE,
)

# Proposed-plan block
_PROPOSED_PLAN_RE = re.compile(r"<proposed_plan>(.*?)</proposed_plan>", re.DOTALL)

# Codex Bash Mode: user-initiated shell commands stored in user messages
_USER_SHELL_CMD_RE = re.compile(
    r"<user_shell_command>\s*<command>(.*?)</command>\s*<result>(.*?)</result>\s*</user_shell_command>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Tool name maps
# ---------------------------------------------------------------------------

_CODEX_TO_STANDARD: dict[str, str] = {
    "shell_command": StandardToolName.BASH,
    "shell":         StandardToolName.BASH,
    "file_read":     StandardToolName.READ,
    "file_write":    StandardToolName.WRITE,
    "file_edit":     StandardToolName.EDIT,
}

# Standard → codex native tool name (determines function_call vs custom_tool_call)
_APPLY_PATCH_TOOLS = {StandardToolName.WRITE, StandardToolName.EDIT}
_SHELL_COMMAND_TOOLS = {
    StandardToolName.BASH,
    StandardToolName.READ,
    StandardToolName.GREP,
    StandardToolName.GLOB,
}


# ---------------------------------------------------------------------------
# UUID v7 (Python 3.10 compatible — no external deps)
# ---------------------------------------------------------------------------

def _uuid7() -> str:
    """Generate a UUID v7 (time-ordered) compatible with Codex's ThreadId format."""
    ms = int(time.time() * 1000)
    rand_a = random.getrandbits(12)
    rand_b = random.getrandbits(62)
    high = (ms << 16) | (0b0111 << 12) | rand_a
    low = (0b10 << 62) | rand_b
    return str(uuid.UUID(int=(high << 64) | low))


# ---------------------------------------------------------------------------
# apply_patch format helpers
# ---------------------------------------------------------------------------

def _apply_patch_write(file_path: str, content: str) -> str:
    """Build an apply_patch payload that creates/replaces a file."""
    lines = "\n".join(f"+{line}" for line in content.splitlines())
    return f"*** Begin Patch\n*** Add File: {file_path}\n{lines}\n*** End Patch"


def _apply_patch_edit(file_path: str, old_string: str, new_string: str) -> str:
    """Build an apply_patch payload that replaces old_string with new_string."""
    old_lines = "\n".join(f"-{l}" for l in old_string.splitlines())
    new_lines = "\n".join(f"+{l}" for l in new_string.splitlines())
    return (
        f"*** Begin Patch\n*** Update File: {file_path}\n@@\n"
        f"{old_lines}\n{new_lines}\n*** End Patch"
    )


def _parse_apply_patch_changes(patch_input: str) -> dict:
    """
    Parse an apply_patch payload into the changes dict expected by patch_apply_end.

    Handles single-file patches (Add File / Update File / Delete File).
    Returns: {file_path: {"type": "add"|"update"|"delete", ...}}
    """
    changes: dict = {}
    current_file: str | None = None
    current_type: str | None = None
    current_lines: list[str] = []

    def _flush():
        if not current_file or not current_type:
            return
        if current_type == "add":
            content = "\n".join(
                ln[1:] if ln.startswith("+") else ln for ln in current_lines
            )
            changes[current_file] = {"type": "add", "content": content}
        elif current_type == "update":
            old_lines = [ln[1:] for ln in current_lines if ln.startswith("-")]
            new_lines = [ln[1:] for ln in current_lines if ln.startswith("+")]
            diff = f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n"
            diff += "\n".join(f"-{l}" for l in old_lines)
            if old_lines and new_lines:
                diff += "\n"
            diff += "\n".join(f"+{l}" for l in new_lines)
            changes[current_file] = {
                "type": "update",
                "unified_diff": diff,
                "move_path": None,
            }
        elif current_type == "delete":
            content = "\n".join(
                ln[1:] if ln.startswith("-") else ln for ln in current_lines
            )
            changes[current_file] = {"type": "delete", "content": content}

    for raw_line in patch_input.splitlines():
        if raw_line in ("*** Begin Patch", "*** End Patch", "@@"):
            if raw_line == "*** End Patch":
                _flush()
                current_file = None
                current_type = None
                current_lines = []
            continue
        if raw_line.startswith("*** Add File: "):
            _flush()
            current_file = raw_line[len("*** Add File: "):]
            current_type = "add"
            current_lines = []
        elif raw_line.startswith("*** Update File: "):
            _flush()
            current_file = raw_line[len("*** Update File: "):]
            current_type = "update"
            current_lines = []
        elif raw_line.startswith("*** Delete File: "):
            _flush()
            current_file = raw_line[len("*** Delete File: "):]
            current_type = "delete"
            current_lines = []
        elif current_file is not None:
            current_lines.append(raw_line)

    _flush()
    return changes


# ---------------------------------------------------------------------------
# File scanning helpers
# ---------------------------------------------------------------------------

def _all_rollout_files() -> list[Path]:
    """Return all rollout JSONL files (sessions + archived), newest-mtime first."""
    files: list[Path] = []
    for base in (_sessions_dir(), _archived_sessions_dir()):
        if base.exists():
            files.extend(base.rglob("rollout-*.jsonl"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _read_session_meta(path: Path) -> dict | None:
    """Read the session_meta payload from the first line of a rollout file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            line = f.readline().strip()
            if not line:
                return None
            rec = json.loads(line)
            if rec.get("type") == "session_meta":
                return rec.get("payload", {})
    except Exception:
        pass
    return None


def _find_rollout_file(session_id: str) -> Path | None:
    """Locate the rollout file whose name contains the given session UUID."""
    for path in _all_rollout_files():
        if session_id in path.name:
            return path
    return None


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _last_timestamp(path: Path) -> datetime | None:
    """Return the timestamp of the last parseable record in a rollout file."""
    last_ts = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ts = _parse_timestamp(rec.get("timestamp"))
                    if ts:
                        last_ts = ts
                except Exception:
                    continue
    except Exception:
        pass
    return last_ts


def _count_message_lines(path: Path) -> int:
    """Fast count of user/assistant message records without full parsing."""
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if (
                    '"type":"response_item"' in stripped or '"type": "response_item"' in stripped
                ) and (
                    '"role":"user"' in stripped
                    or '"role": "user"' in stripped
                    or '"role":"assistant"' in stripped
                    or '"role": "assistant"' in stripped
                ):
                    count += 1
    except Exception:
        pass
    return count


def _extract_display_name(path: Path, session_id: str) -> str:
    """
    Extract the conversation display name from a rollout file.
    Priority: slug in filename → first meaningful user message → session_id.
    """
    # 1. Slug embedded in filename: rollout-...-{uuid}-{slug}.jsonl
    stem = path.stem  # e.g. rollout-2026-04-12T19-31-59-019d84ae-...-my_slug
    # UUID is 36 chars; slug follows the last occurrence of UUID-ish pattern
    slug_match = re.search(
        r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-(.+)$",
        stem,
        re.IGNORECASE,
    )
    if slug_match:
        return slug_match.group(1).replace("_", " ")

    # 2. First meaningful user message
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("type") != "response_item":
                        continue
                    payload = rec.get("payload", {})
                    if payload.get("type") != "message" or payload.get("role") != "user":
                        continue
                    content = payload.get("content", [])
                    for block in content:
                        text = block.get("text", "").strip()
                        if text and not _CODEX_SYSTEM_CONTENT_RE.match(text):
                            return text[:80].replace("\n", " ")
                except Exception:
                    continue
    except Exception:
        pass

    return session_id


# ---------------------------------------------------------------------------
# CodexAdapter
# ---------------------------------------------------------------------------

class CodexAdapter(ToolAdapter):
    name = "Codex"
    tool_id = "codex"

    def is_available(self) -> bool:
        return _codex_dir().exists()

    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        resolved = project_path.resolve()
        results: list[ConversationInfo] = []

        for rollout_file in _all_rollout_files():
            meta = _read_session_meta(rollout_file)
            if not meta:
                continue

            # Compare cwd to project_path (case-insensitive on Windows)
            cwd = meta.get("cwd", "")
            try:
                if Path(cwd).resolve() != resolved:
                    continue
            except Exception:
                continue

            session_id = meta.get("id", rollout_file.stem)
            name = _extract_display_name(rollout_file, session_id)

            size = rollout_file.stat().st_size
            updated_at = _last_timestamp(rollout_file) or datetime.fromtimestamp(
                rollout_file.stat().st_mtime, tz=timezone.utc
            )
            created_at = (
                _parse_timestamp(meta.get("timestamp"))
                or datetime.fromtimestamp(rollout_file.stat().st_ctime, tz=timezone.utc)
            )
            message_count = _count_message_lines(rollout_file)

            results.append(ConversationInfo(
                id=session_id,
                name=name,
                updated_at=updated_at,
                created_at=created_at,
                message_count=message_count,
                size_bytes=size,
                source_tool=self.tool_id,
            ))

        results.sort(key=lambda c: c.updated_at, reverse=True)
        return results

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        rollout_file = _find_rollout_file(conv_id)
        if rollout_file is None:
            raise FileNotFoundError(f"Codex session not found: {conv_id}")

        turns: list = []
        pending_tool_calls: dict[str, ToolCallMessage] = {}
        model: str | None = None

        meta = _read_session_meta(rollout_file)

        with open(rollout_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                outer_type = rec.get("type")
                ts = _parse_timestamp(rec.get("timestamp"))

                # Skip non-conversation record types
                if outer_type in _SKIP_OUTER_TYPES or outer_type == "session_meta":
                    continue

                # Extract model from turn_context (already skipped above, but capture first)
                if outer_type == "turn_context":
                    if model is None:
                        model = rec.get("payload", {}).get("model") or None
                    continue

                if outer_type != "response_item":
                    continue

                payload = rec.get("payload", {})
                ptype = payload.get("type")

                # Skip reasoning, compacted, etc.
                if ptype in _SKIP_PAYLOAD_TYPES:
                    continue

                role = payload.get("role")

                # ---- Message -----------------------------------------------
                if ptype == "message":
                    if role in _SKIP_ROLES:
                        continue

                    content_blocks = payload.get("content", [])
                    if not isinstance(content_blocks, list):
                        content_blocks = []

                    if role == "user":
                        # Collect input_text blocks; skip if all are system-injected.
                        # Each block may be a Bash Mode <user_shell_command> — if so,
                        # convert it to a ToolCallMessage(Bash) instead of user text.
                        texts: list[str] = []
                        for block in content_blocks:
                            text = block.get("text", "").strip() if isinstance(block, dict) else ""
                            if not text:
                                continue
                            if _CODEX_SYSTEM_CONTENT_RE.match(text):
                                continue
                            # Bash Mode: the entire text is a <user_shell_command> block.
                            m = _USER_SHELL_CMD_RE.fullmatch(text)
                            if m:
                                cmd = m.group(1).strip()
                                result = m.group(2).strip()
                                turns.append(ToolCallMessage(
                                    name=StandardToolName.BASH,
                                    input={"command": cmd},
                                    result=result,
                                    timestamp=ts,
                                ))
                            else:
                                texts.append(text)
                        if texts:
                            turns.append(TextMessage(
                                role="user",
                                text="\n".join(texts),
                                timestamp=ts,
                            ))

                    elif role == "assistant":
                        texts = []
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get("type") in (
                                "output_text", "input_text"
                            ):
                                text = block.get("text", "").strip()
                                if text:
                                    texts.append(text)
                        combined = "\n".join(texts)
                        if combined:
                            turns.append(TextMessage(
                                role="assistant",
                                text=combined,
                                timestamp=ts,
                            ))

                # ---- function_call -----------------------------------------
                elif ptype == "function_call":
                    native_name = payload.get("name", "unknown")
                    std_name = _CODEX_TO_STANDARD.get(native_name, native_name)
                    raw_args = payload.get("arguments", "{}")
                    try:
                        input_dict = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        input_dict = {"raw": raw_args}
                    call_id = payload.get("call_id", "")
                    tc = ToolCallMessage(
                        name=std_name,
                        input=input_dict,
                        result="",
                        timestamp=ts,
                    )
                    turns.append(tc)
                    if call_id:
                        pending_tool_calls[call_id] = tc

                # ---- function_call_output -----------------------------------
                elif ptype == "function_call_output":
                    call_id = payload.get("call_id", "")
                    output = payload.get("output", "")
                    if not isinstance(output, str):
                        output = json.dumps(output)
                    if call_id in pending_tool_calls:
                        pending_tool_calls[call_id].result = output

                # ---- custom_tool_call (apply_patch etc.) -------------------
                elif ptype == "custom_tool_call":
                    native_name = payload.get("name", "unknown")
                    std_name = _CODEX_TO_STANDARD.get(native_name, native_name)
                    raw_input = payload.get("input", "")
                    # Store raw patch string in input dict for round-trip fidelity
                    input_dict = {"patch": raw_input} if isinstance(raw_input, str) else raw_input
                    call_id = payload.get("call_id", "")
                    tc = ToolCallMessage(
                        name=std_name,
                        input=input_dict,
                        result="",
                        timestamp=ts,
                    )
                    turns.append(tc)
                    if call_id:
                        pending_tool_calls[call_id] = tc

                # ---- custom_tool_call_output --------------------------------
                elif ptype == "custom_tool_call_output":
                    call_id = payload.get("call_id", "")
                    output = payload.get("output", "")
                    if not isinstance(output, str):
                        output = json.dumps(output)
                    if call_id in pending_tool_calls:
                        pending_tool_calls[call_id].result = output

        # Re-scan turn_context for model (it was skipped in the main loop above)
        if model is None:
            try:
                with open(rollout_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                        except Exception:
                            continue
                        if rec.get("type") == "turn_context":
                            m = rec.get("payload", {}).get("model")
                            if m:
                                model = m
                                break
            except Exception:
                pass

        # Extract plan_content from the last assistant message containing <proposed_plan>
        plan_content: str | None = None
        for turn in reversed(turns):
            if isinstance(turn, TextMessage) and turn.role == "assistant":
                m = _PROPOSED_PLAN_RE.search(turn.text)
                if m:
                    plan_content = m.group(1).strip() or None
                    break

        name = _extract_display_name(rollout_file, conv_id)
        updated_at = _last_timestamp(rollout_file) or datetime.fromtimestamp(
            rollout_file.stat().st_mtime, tz=timezone.utc
        )
        created_at = (
            _parse_timestamp(meta.get("timestamp") if meta else None)
            or datetime.fromtimestamp(rollout_file.stat().st_ctime, tz=timezone.utc)
        )
        info = ConversationInfo(
            id=conv_id,
            name=name,
            updated_at=updated_at,
            created_at=created_at,
            message_count=sum(1 for t in turns if isinstance(t, TextMessage)),
            size_bytes=rollout_file.stat().st_size,
            source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=turns, plan_content=plan_content, model=model)

    def write_conversation(
        self,
        conv: Conversation,
        project_path: Path,
        *,
        use_local_backend: bool = False,
    ) -> str:
        # use_local_backend is silently ignored — Codex has no remote upload path.
        now = datetime.now(timezone.utc)
        session_id = _uuid7()

        ts_str = now.strftime("%Y-%m-%dT%H-%M-%S")
        # Codex's filename parser (parse_timestamp_uuid_from_filename) scans from
        # the RIGHT for a segment that parses as a UUID.  Any suffix after the UUID
        # (e.g. a human-readable slug) breaks this scan — the file would be silently
        # skipped and never appear in /resume.  Native Codex never appends a slug.
        filename = f"rollout-{ts_str}-{session_id}.jsonl"

        target_dir = (
            _sessions_dir()
            / str(now.year)
            / f"{now.month:02d}"
            / f"{now.day:02d}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        final_path = target_dir / filename
        tmp_path = target_dir / f"{filename}.tmp"

        # Codex's session picker filters by model_provider and only shows sessions
        # that match the configured provider (always "openai" on a standard
        # install).  Migrated sessions must use "openai" to appear in /resume.
        model_provider = "openai"

        def _make_event(payload: dict) -> str:
            """Wrap a payload as an event_msg record."""
            return json.dumps(
                {"timestamp": _now_iso(), "type": "event_msg", "payload": payload},
                ensure_ascii=False,
            )

        def _make_response_item(ts: str, payload: dict) -> str:
            return json.dumps(
                {"timestamp": ts, "type": "response_item", "payload": payload},
                ensure_ascii=False,
            )

        def _new_call_id() -> str:
            return f"call_{uuid.uuid4().hex[:24]}"

        # ── Group turns into (user_turn, [agent_turns]) exchanges ────────────
        # Each exchange begins with a user TextMessage (or None for leading agent
        # turns) and contains all subsequent non-user turns until the next user
        # TextMessage.
        exchanges: list[tuple] = []
        cur_user: TextMessage | None = None
        cur_agent: list = []
        for turn in conv.turns:
            if isinstance(turn, TextMessage) and turn.role == "user":
                exchanges.append((cur_user, cur_agent))
                cur_user = turn
                cur_agent = []
            else:
                cur_agent.append(turn)
        exchanges.append((cur_user, cur_agent))
        # Drop any leading (None, []) placeholder
        exchanges = [(u, a) for u, a in exchanges if u is not None or a]

        # Whether the plan is already embedded in an assistant turn
        plan_already_written = conv.plan_content is not None and any(
            isinstance(t, TextMessage)
            and t.role == "assistant"
            and "<proposed_plan>" in t.text
            for t in conv.turns
        )

        cwd_str = str(project_path.resolve())

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # ── Session meta (first record) ──────────────────────────────
                f.write(json.dumps({
                    "timestamp": now.isoformat(),
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "timestamp": now.isoformat(),
                        "cwd": cwd_str,
                        "originator": "agent-migrator",
                        "cli_version": "0.120.0",
                        "source": "cli",
                        "model_provider": model_provider,
                    },
                }, ensure_ascii=False) + "\n")

                # ── Exchanges ────────────────────────────────────────────────
                for user_turn, agent_turns in exchanges:
                    turn_id = _uuid7()
                    turn_ts = (
                        user_turn.timestamp.isoformat()
                        if user_turn and user_turn.timestamp
                        else _now_iso()
                    )

                    # task_started — required for /resume list display
                    started_at = (
                        int(user_turn.timestamp.timestamp())
                        if user_turn and user_turn.timestamp
                        else None
                    )
                    f.write(_make_event({
                        "type": "task_started",
                        "turn_id": turn_id,
                        "started_at": started_at,
                        "model_context_window": None,
                        "collaboration_mode_kind": "default",
                    }) + "\n")

                    # User message
                    if user_turn:
                        f.write(_make_event({
                            "type": "user_message",
                            "message": user_turn.text,
                            "images": [],
                            "local_images": [],
                            "text_elements": [],
                        }) + "\n")
                        f.write(_make_response_item(turn_ts, {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_turn.text}],
                        }) + "\n")

                    # Agent turns
                    for turn in agent_turns:
                        t_ts = (
                            turn.timestamp.isoformat()
                            if turn.timestamp
                            else _now_iso()
                        )

                        if isinstance(turn, TextMessage):
                            # response_item/message/assistant — keep full text (round-trip)
                            f.write(_make_response_item(t_ts, {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": turn.text}],
                            }) + "\n")
                            # event_msg/agent_message — TUI display; strip <proposed_plan>
                            # XML so raw tags don't appear as literal text.
                            display_text = _PROPOSED_PLAN_DISPLAY_RE.sub(
                                lambda m: "\n\n" + m.group(1).strip() + "\n\n", turn.text
                            ).strip()
                            if display_text:
                                f.write(_make_event({
                                    "type": "agent_message",
                                    "message": display_text,
                                    "phase": "commentary",
                                    "memory_citation": None,
                                }) + "\n")

                        elif isinstance(turn, ToolCallMessage):
                            call_id = _new_call_id()
                            name = turn.name

                            if name in _APPLY_PATCH_TOOLS or name == "apply_patch":
                                # ── apply_patch (Write / Edit) ───────────────
                                if name == StandardToolName.WRITE:
                                    patch_input = _apply_patch_write(
                                        turn.input.get("file_path", ""),
                                        turn.input.get("content", ""),
                                    )
                                elif name == StandardToolName.EDIT:
                                    patch_input = _apply_patch_edit(
                                        turn.input.get("file_path", ""),
                                        turn.input.get("old_string", ""),
                                        turn.input.get("new_string", ""),
                                    )
                                else:
                                    patch_input = turn.input.get(
                                        "patch", json.dumps(turn.input)
                                    )

                                f.write(_make_response_item(t_ts, {
                                    "type": "custom_tool_call",
                                    "name": "apply_patch",
                                    "input": patch_input,
                                    "call_id": call_id,
                                    "status": "completed",
                                }) + "\n")

                                # event_msg/patch_apply_begin + patch_apply_end pair.
                                # NOTE: patch_apply_end{success=true} is a no-op in TUI
                                # history replay — the "Edited" block is only added by
                                # on_patch_apply_begin during live sessions.  We emit a
                                # synthetic agent_message per changed file so that file
                                # operations are visible in the /resume history view.
                                changes = _parse_apply_patch_changes(patch_input)
                                for changed_path, change_info in changes.items():
                                    change_type = change_info.get("type", "update")
                                    verb = {"add": "Created", "delete": "Deleted"}.get(
                                        change_type, "Updated"
                                    )
                                    f.write(_make_event({
                                        "type": "agent_message",
                                        "message": f"{verb} `{changed_path}`",
                                        "phase": "commentary",
                                        "memory_citation": None,
                                    }) + "\n")
                                f.write(_make_event({
                                    "type": "patch_apply_begin",
                                    "call_id": call_id,
                                    "turn_id": turn_id,
                                    "auto_approved": True,
                                    "changes": changes,
                                }) + "\n")
                                f.write(_make_event({
                                    "type": "patch_apply_end",
                                    "call_id": call_id,
                                    "turn_id": turn_id,
                                    "stdout": turn.result or "Applied.",
                                    "stderr": "",
                                    "success": True,
                                    "changes": changes,
                                    "status": "completed",
                                }) + "\n")

                                f.write(_make_response_item(t_ts, {
                                    "type": "custom_tool_call_output",
                                    "call_id": call_id,
                                    "output": turn.result,
                                }) + "\n")

                            elif name in _SHELL_COMMAND_TOOLS:
                                # ── shell_command (Bash / Read / Grep / Glob) ─
                                if name == StandardToolName.BASH:
                                    cmd_str = turn.input.get("command", "")
                                elif name == StandardToolName.READ:
                                    cmd_str = f"cat {turn.input.get('file_path', '')}"
                                elif name == StandardToolName.GREP:
                                    pat = turn.input.get("pattern", "")
                                    path_arg = turn.input.get("path", ".")
                                    cmd_str = f"grep -r {json.dumps(pat)} {path_arg}"
                                else:  # GLOB
                                    cmd_str = (
                                        f"find . -name {json.dumps(turn.input.get('pattern', '*'))}"
                                    )

                                f.write(_make_response_item(t_ts, {
                                    "type": "function_call",
                                    "name": "shell_command",
                                    "arguments": json.dumps({"command": cmd_str}),
                                    "call_id": call_id,
                                }) + "\n")

                                # event_msg/exec_command_end — TUI displays command output
                                f.write(_make_event({
                                    "type": "exec_command_end",
                                    "call_id": call_id,
                                    "process_id": None,
                                    "turn_id": turn_id,
                                    "command": ["bash", "-c", cmd_str],
                                    "cwd": cwd_str,
                                    "parsed_cmd": [{"type": "unknown", "cmd": cmd_str}],
                                    "source": "agent",
                                    "interaction_input": None,
                                    "stdout": turn.result or "",
                                    "stderr": "",
                                    "aggregated_output": turn.result or "",
                                    "exit_code": 0,
                                    "duration": {"secs": 0, "nanos": 0},
                                    "formatted_output": turn.result or "",
                                    "status": "completed",
                                }) + "\n")

                                f.write(_make_response_item(t_ts, {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": turn.result,
                                }) + "\n")

                            else:
                                # ── Passthrough (unknown tool) ────────────────
                                f.write(_make_response_item(t_ts, {
                                    "type": "function_call",
                                    "name": name,
                                    "arguments": json.dumps(turn.input or {}),
                                    "call_id": call_id,
                                }) + "\n")
                                f.write(_make_response_item(t_ts, {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": turn.result,
                                }) + "\n")

                    # task_complete — closes out the exchange
                    last_agent_text: str | None = None
                    for t in reversed(agent_turns):
                        if isinstance(t, TextMessage) and t.role == "assistant":
                            last_agent_text = (t.text[:200] if t.text else None)
                            break
                    f.write(_make_event({
                        "type": "task_complete",
                        "turn_id": turn_id,
                        "last_agent_message": last_agent_text,
                        "completed_at": None,
                        "duration_ms": None,
                    }) + "\n")

                # ── Append plan if not already embedded ──────────────────────
                if conv.plan_content and not plan_already_written:
                    # response_item keeps the XML wrapper so read_conversation() can
                    # extract plan_content on round-trip; agent_message shows the
                    # clean markdown without raw XML tags.
                    plan_xml = (
                        f"<proposed_plan>\n{conv.plan_content}\n</proposed_plan>"
                    )
                    f.write(_make_response_item(_now_iso(), {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": plan_xml}],
                    }) + "\n")
                    f.write(_make_event({
                        "type": "agent_message",
                        "message": conv.plan_content.strip(),
                        "phase": "planning",
                        "memory_citation": None,
                    }) + "\n")

            # Atomic rename
            tmp_path.replace(final_path)

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return session_id

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        rollout_file = _find_rollout_file(conv_id)
        if rollout_file and rollout_file.exists():
            rollout_file.unlink()
