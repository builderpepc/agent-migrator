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
        slug_base = re.sub(r"[^a-z0-9]+", "_", conv.info.name.lower()).strip("_")[:40]
        filename = f"rollout-{ts_str}-{session_id}"
        if slug_base:
            filename += f"-{slug_base}"
        filename += ".jsonl"

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

        def _make_record(rtype: str, payload: dict) -> str:
            rec = {
                "timestamp": _now_iso(),
                "type": rtype,
                "payload": payload,
            }
            return json.dumps(rec, ensure_ascii=False)

        def _new_call_id() -> str:
            return f"call_{uuid.uuid4().hex[:24]}"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # Session meta (first record)
                f.write(_make_record("session_meta", {
                    "id": session_id,
                    "timestamp": now.isoformat(),
                    "cwd": str(project_path.resolve()),
                    "originator": "agent-migrator",
                    "cli_version": "0.120.0",
                    "source": "cli",
                    "model_provider": model_provider,
                }) + "\n")

                # Track whether plan has already been written (present in an assistant turn)
                plan_already_written = conv.plan_content is not None and any(
                    isinstance(t, TextMessage)
                    and t.role == "assistant"
                    and "<proposed_plan>" in t.text
                    for t in conv.turns
                )

                for turn in conv.turns:
                    turn_ts = turn.timestamp.isoformat() if turn.timestamp else _now_iso()

                    if isinstance(turn, TextMessage):
                        content_type = "input_text" if turn.role == "user" else "output_text"
                        payload = {
                            "type": "message",
                            "role": turn.role,
                            "content": [{"type": content_type, "text": turn.text}],
                        }
                        rec = {
                            "timestamp": turn_ts,
                            "type": "response_item",
                            "payload": payload,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    elif isinstance(turn, ToolCallMessage):
                        call_id = _new_call_id()
                        name = turn.name

                        if name in _APPLY_PATCH_TOOLS or (
                            name not in _SHELL_COMMAND_TOOLS
                            and name == "apply_patch"
                        ):
                            # custom_tool_call with apply_patch format
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
                                # Passthrough: raw patch already stored
                                patch_input = turn.input.get("patch", json.dumps(turn.input))

                            call_rec = {
                                "timestamp": turn_ts,
                                "type": "response_item",
                                "payload": {
                                    "type": "custom_tool_call",
                                    "name": "apply_patch",
                                    "input": patch_input,
                                    "call_id": call_id,
                                    "status": "completed",
                                },
                            }
                            result_rec = {
                                "timestamp": turn_ts,
                                "type": "response_item",
                                "payload": {
                                    "type": "custom_tool_call_output",
                                    "call_id": call_id,
                                    "output": turn.result,
                                },
                            }
                        else:
                            # function_call with shell_command
                            if name == StandardToolName.BASH:
                                arguments = {"command": turn.input.get("command", "")}
                            elif name == StandardToolName.READ:
                                fp = turn.input.get("file_path", "")
                                arguments = {"command": f"cat {fp}"}
                            elif name == StandardToolName.GREP:
                                pattern = turn.input.get("pattern", "")
                                path_arg = turn.input.get("path", ".")
                                arguments = {"command": f"grep -r {json.dumps(pattern)} {path_arg}"}
                            elif name == StandardToolName.GLOB:
                                glob_pattern = turn.input.get("pattern", "*")
                                arguments = {"command": f"find . -name {json.dumps(glob_pattern)}"}
                            else:
                                # Passthrough: preserve the native/unknown tool name as-is.
                                # Don't remap to shell_command — that would lose the name on read.
                                call_rec = {
                                    "timestamp": turn_ts,
                                    "type": "response_item",
                                    "payload": {
                                        "type": "function_call",
                                        "name": name,
                                        "arguments": json.dumps(turn.input or {}),
                                        "call_id": call_id,
                                    },
                                }
                                result_rec = {
                                    "timestamp": turn_ts,
                                    "type": "response_item",
                                    "payload": {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": turn.result,
                                    },
                                }
                                f.write(json.dumps(call_rec, ensure_ascii=False) + "\n")
                                f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")
                                continue  # skip the common write below

                            call_rec = {
                                "timestamp": turn_ts,
                                "type": "response_item",
                                "payload": {
                                    "type": "function_call",
                                    "name": "shell_command",
                                    "arguments": json.dumps(arguments),
                                    "call_id": call_id,
                                },
                            }
                            result_rec = {
                                "timestamp": turn_ts,
                                "type": "response_item",
                                "payload": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": turn.result,
                                },
                            }

                        f.write(json.dumps(call_rec, ensure_ascii=False) + "\n")
                        f.write(json.dumps(result_rec, ensure_ascii=False) + "\n")

                # Append plan as a final assistant message if not already present
                if conv.plan_content and not plan_already_written:
                    plan_rec = {
                        "timestamp": _now_iso(),
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": f"<proposed_plan>\n{conv.plan_content}\n</proposed_plan>",
                            }],
                        },
                    }
                    f.write(json.dumps(plan_rec, ensure_ascii=False) + "\n")

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
