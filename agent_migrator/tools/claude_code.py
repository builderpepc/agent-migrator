from __future__ import annotations

import difflib
import html as _html
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    TextMessage,
    ToolCallMessage,
)
from agent_migrator.tools.base import ToolAdapter

# Record types to skip when reading conversations
_SKIP_TYPES = {"file-history-snapshot", "progress", "system"}


def _claude_dir() -> Path:
    return Path.home() / ".claude"


def _projects_dir() -> Path:
    return _claude_dir() / "projects"


def encode_project_path(project_path: Path) -> str:
    """
    Encode an absolute project path to Claude Code's folder-name format.

    Replace every ':', '/' and '\\' with '-'.
    Unix:    /Users/me/my-app    -> -Users-me-my-app
    Windows: C:/Users/me/my-app  -> C--Users-me-my-app
    """
    # Normalise to forward slashes first for consistent encoding
    normalised = str(project_path).replace("\\", "/")
    return re.sub(r"[:/]", "-", normalised)


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_TAIL_READ_BYTES = 32768  # 32 KB — enough to find appended title fields
# Pattern for display-name extraction only — broad heuristic, not used for migration filtering
_SKIP_FIRST_PROMPT_RE = re.compile(r'^(?:\s*<[a-z][\w-]*[\s>]|\[Request interrupted by user)')

# Specific Claude Code internal protocol tags that should never be migrated as user turns.
# These are machine-generated markers, not user-authored content.
_CC_INTERNAL_CONTENT_RE = re.compile(
    r'^<(?:'
    r'command-(?:name|message|args)|'          # slash commands: /exit, /clear, etc.
    r'local-command-(?:caveat|stdout|stderr)|' # local command wrappers
    r'anysphere-remote-filesystem-content|'    # remote file content
    r'function_calls|result'                   # tool call wrappers
    r')[\s>]',
    re.IGNORECASE,
)

_BASH_TAG_RE = re.compile(r'<(bash-input|bash-stdout|bash-stderr)>([\s\S]*?)</\1>', re.IGNORECASE)


def _extract_json_string_field(text: str, key: str) -> str | None:
    """Find the LAST occurrence of "key":"value" in text (handles both spaced/non-spaced)."""
    last_value: str | None = None
    for pattern in (f'"{key}":"', f'"{key}": "'):
        search_from = 0
        while True:
            idx = text.find(pattern, search_from)
            if idx < 0:
                break
            value_start = idx + len(pattern)
            i = value_start
            chars: list[str] = []
            while i < len(text):
                ch = text[i]
                if ch == "\\":
                    if i + 1 < len(text):
                        nc = text[i + 1]
                        if nc == '"':
                            chars.append('"')
                        elif nc == "n":
                            chars.append("\n")
                        elif nc == "t":
                            chars.append("\t")
                        elif nc == "\\":
                            chars.append("\\")
                        else:
                            chars.append(nc)
                    i += 2
                    continue
                if ch == '"':
                    last_value = "".join(chars)
                    break
                chars.append(ch)
                i += 1
            search_from = i + 1
    return last_value


def _display_name_from_file(path: Path) -> str | None:
    """
    Extract the conversation display name exactly as Claude Code's UI does:
      customTitle > aiTitle > lastPrompt > summary > firstPrompt > slug
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            # Read head (for firstPrompt + slug fallback)
            head_bytes = fh.read(min(size, _TAIL_READ_BYTES))
            head = head_bytes.decode("utf-8", errors="replace")

            # Read tail (for customTitle / aiTitle / lastPrompt / summary)
            if size <= _TAIL_READ_BYTES:
                tail = head
            else:
                fh.seek(max(0, size - _TAIL_READ_BYTES))
                tail = fh.read(_TAIL_READ_BYTES).decode("utf-8", errors="replace")

        # Priority 1 & 2: customTitle / aiTitle (appended records — check tail first)
        for field in ("customTitle", "aiTitle"):
            value = _extract_json_string_field(tail, field) or _extract_json_string_field(head, field)
            if value:
                return value

        # Priority 3 & 4: lastPrompt / summary (tail only)
        for field in ("lastPrompt", "summary"):
            value = _extract_json_string_field(tail, field)
            if value:
                return value

        # Priority 5: first meaningful user prompt (head scan)
        first_prompt = _extract_first_prompt(head)
        if first_prompt:
            return first_prompt

        # Priority 6: slug (random word combo used for plan files)
        slug = _extract_json_string_field(head, "slug")
        if slug:
            return slug

    except Exception:
        pass
    return None


def _extract_first_prompt(head: str) -> str | None:
    """
    Find the first meaningful user message text, skipping tool_result,
    isMeta, isCompactSummary, and auto-generated XML-prefixed messages.
    Mirrors Claude Code's extractFirstPromptFromHead() logic.
    """
    command_fallback: str | None = None
    for line in head.splitlines():
        if '"type":"user"' not in line and '"type": "user"' not in line:
            continue
        if '"tool_result"' in line:
            continue
        if '"isMeta":true' in line or '"isMeta": true' in line:
            continue
        if '"isCompactSummary":true' in line or '"isCompactSummary": true' in line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("type") != "user":
                continue
            message = rec.get("message", {})
            content = message.get("content", "")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "")
                        if isinstance(t, str):
                            texts.append(t)
            for raw in texts:
                result = raw.replace("\n", " ").strip()
                if not result:
                    continue
                # Slash commands: remember as fallback but keep looking
                cmd_match = re.search(r'<command-name>(.*?)</command-name>', result)
                if cmd_match:
                    if not command_fallback:
                        command_fallback = cmd_match.group(1)
                    continue
                if _SKIP_FIRST_PROMPT_RE.match(result):
                    continue
                if len(result) > 200:
                    result = result[:200].rstrip() + "\u2026"
                return result
        except Exception:
            continue
    return command_fallback


def _last_timestamp(path: Path) -> datetime | None:
    """Return the timestamp of the last parseable record in a JSONL file."""
    last_ts = None
    try:
        with open(path, encoding="utf-8") as f:
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
    """Fast count of user/assistant records without parsing full JSON."""
    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if '"type":"user"' in stripped or '"type":"assistant"' in stripped:
                    count += 1
    except Exception:
        pass
    return count


def _structured_patch(old_string: str, new_string: str) -> list[dict]:
    """
    Compute a structuredPatch list in the format CC uses for Edit toolUseResult.
    Each hunk: {oldStart, oldLines, newStart, newLines, lines: ["-old", "+new", " ctx"]}
    """
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks = []
    for group in matcher.get_grouped_opcodes(3):
        old_start = group[0][1] + 1  # 1-based
        new_start = group[0][3] + 1
        old_count = 0
        new_count = 0
        lines: list[str] = []
        for op, i1, i2, j1, j2 in group:
            if op == "equal":
                for line in old_lines[i1:i2]:
                    lines.append(f" {line}")
                old_count += i2 - i1
                new_count += i2 - i1
            elif op in ("delete", "replace"):
                for line in old_lines[i1:i2]:
                    lines.append(f"-{line}")
                old_count += i2 - i1
                if op == "replace":
                    for line in new_lines[j1:j2]:
                        lines.append(f"+{line}")
                    new_count += j2 - j1
            elif op == "insert":
                for line in new_lines[j1:j2]:
                    lines.append(f"+{line}")
                new_count += j2 - j1
        hunks.append({
            "oldStart": old_start,
            "oldLines": old_count,
            "newStart": new_start,
            "newLines": new_count,
            "lines": lines,
        })
    return hunks


class ClaudeCodeAdapter(ToolAdapter):
    name = "Claude Code"
    tool_id = "claude_code"

    def is_available(self) -> bool:
        return _claude_dir().exists()

    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        encoded = encode_project_path(project_path.resolve())
        session_dir = _projects_dir() / encoded
        if not session_dir.exists():
            return []

        results: list[ConversationInfo] = []
        for jsonl_file in session_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            name = _display_name_from_file(jsonl_file) or session_id

            size = jsonl_file.stat().st_size
            updated_at = _last_timestamp(jsonl_file) or datetime.fromtimestamp(
                jsonl_file.stat().st_mtime, tz=timezone.utc
            )
            created_at = datetime.fromtimestamp(
                jsonl_file.stat().st_ctime, tz=timezone.utc
            )
            message_count = _count_message_lines(jsonl_file)

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
        encoded = encode_project_path(project_path.resolve())
        jsonl_file = _projects_dir() / encoded / f"{conv_id}.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Claude Code session not found: {jsonl_file}")

        name = _display_name_from_file(jsonl_file) or conv_id
        turns: list = []

        # Map tool_use_id -> ToolCallMessage (pending result)
        pending_tool_calls: dict[str, ToolCallMessage] = {}
        # Bash mode: accumulate command from <bash-input> to combine with its output
        pending_bash_cmd: str | None = None

        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                rtype = rec.get("type")
                if rtype in _SKIP_TYPES:
                    continue

                # Skip meta/internal records (hook output, compact summaries, etc.)
                # Also skip any user turn with a defined `origin` — these come from
                # slash commands, hooks, cron, or other synthetic sources, not keyboard input.
                if rec.get("isMeta") or rec.get("isCompactSummary"):
                    continue
                if rec.get("type") == "user" and rec.get("origin") is not None:
                    continue

                message = rec.get("message", {})
                role = message.get("role")
                content = message.get("content", "")

                if rtype == "assistant" and role == "assistant":
                    if isinstance(content, list):
                        for block in content:
                            btype = block.get("type")
                            if btype == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    turns.append(TextMessage(
                                        role="assistant",
                                        text=text,
                                        timestamp=_parse_timestamp(rec.get("timestamp")),
                                    ))
                            elif btype == "tool_use":
                                tc = ToolCallMessage(
                                    name=block.get("name", "unknown"),
                                    input=block.get("input", {}),
                                    result="",
                                    timestamp=_parse_timestamp(rec.get("timestamp")),
                                )
                                turns.append(tc)
                                pending_tool_calls[block.get("id", "")] = tc

                elif rtype == "user" and role == "user":
                    if isinstance(content, list):
                        # Check for tool_result blocks
                        is_tool_result = any(
                            b.get("type") == "tool_result" for b in content
                        )
                        if is_tool_result:
                            for block in content:
                                if block.get("type") == "tool_result":
                                    tuid = block.get("tool_use_id", "")
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        result_text = "\n".join(
                                            b.get("text", "") for b in result_content
                                            if b.get("type") == "text"
                                        )
                                    else:
                                        result_text = str(result_content)
                                    if tuid in pending_tool_calls:
                                        pending_tool_calls[tuid].result = result_text
                        else:
                            # Mixed content with text blocks — skip internal CC protocol messages
                            for block in content:
                                if block.get("type") == "text":
                                    text = block.get("text", "").strip()
                                    if text and not _CC_INTERNAL_CONTENT_RE.match(text):
                                        turns.append(TextMessage(
                                            role="user",
                                            text=text,
                                            timestamp=_parse_timestamp(rec.get("timestamp")),
                                        ))
                    elif isinstance(content, str):
                        text = content.strip()
                        ts = _parse_timestamp(rec.get("timestamp"))
                        if text.startswith("<bash-input>"):
                            m = _BASH_TAG_RE.match(text)
                            raw = m.group(2) if m else text
                            pending_bash_cmd = _html.unescape(raw).replace("\r\n", "\n").strip()
                        elif text.startswith("<bash-stdout") or text.startswith("<bash-stderr"):
                            # Collect stdout and stderr from this record
                            stdout = ""
                            stderr = ""
                            for m in _BASH_TAG_RE.finditer(text):
                                cleaned = _html.unescape(m.group(2)).replace("\r\n", "\n").strip()
                                if m.group(1).lower() == "bash-stdout":
                                    stdout = cleaned
                                elif m.group(1).lower() == "bash-stderr":
                                    stderr = cleaned
                            output = "\n".join(filter(None, [stdout, stderr]))
                            # Combine with pending command into one message
                            if pending_bash_cmd is not None:
                                combined = f"$ {pending_bash_cmd}"
                                if output:
                                    combined += f"\n{output}"
                                turns.append(TextMessage(role="user", text=combined, timestamp=ts))
                                pending_bash_cmd = None
                            elif output:
                                turns.append(TextMessage(role="user", text=output, timestamp=ts))
                        else:
                            # Flush any orphaned pending command before moving on
                            if pending_bash_cmd is not None:
                                turns.append(TextMessage(role="user", text=f"$ {pending_bash_cmd}", timestamp=ts))
                                pending_bash_cmd = None
                            if text and not _CC_INTERNAL_CONTENT_RE.match(text):
                                turns.append(TextMessage(
                                    role="user",
                                    text=text,
                                    timestamp=ts,
                                ))

        updated_at = _last_timestamp(jsonl_file) or datetime.fromtimestamp(
            jsonl_file.stat().st_mtime, tz=timezone.utc
        )
        created_at = datetime.fromtimestamp(jsonl_file.stat().st_ctime, tz=timezone.utc)

        info = ConversationInfo(
            id=conv_id,
            name=name,
            updated_at=updated_at,
            created_at=created_at,
            message_count=sum(1 for t in turns if isinstance(t, TextMessage)),
            size_bytes=jsonl_file.stat().st_size,
            source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=turns)

    def write_conversation(self, conv: Conversation, project_path: Path) -> str:
        encoded = encode_project_path(project_path.resolve())
        session_dir = _projects_dir() / encoded
        session_dir.mkdir(parents=True, exist_ok=True)

        session_id = str(uuid.uuid4())
        final_path = session_dir / f"{session_id}.jsonl"
        tmp_path = session_dir / f"{session_id}.jsonl.tmp"

        # Pick a slug: use source slug or derive from name
        slug = re.sub(r"[^a-z0-9]+", "-", conv.info.name.lower()).strip("-") or session_id[:8]

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                prev_uuid: str | None = None

                def _make_base(rec_type: str, rec_uuid: str, ts: str) -> dict:
                    return {
                        "type": rec_type,
                        "uuid": rec_uuid,
                        "parentUuid": prev_uuid,
                        "sessionId": session_id,
                        "timestamp": ts,
                        "isSidechain": False,
                        "userType": "external",
                        "cwd": str(project_path),
                        "slug": slug,
                    }

                def _msg_id() -> str:
                    # Each assistant record needs a unique message.id.
                    # normalizeMessagesForAPI merges any two assistant records whose
                    # message.id are equal — including both undefined — walking back
                    # over tool_result records to find a "matching" prior assistant.
                    # This causes all assistant turns to collapse into one with
                    # mismatched tool_use/tool_result counts → API 400.
                    return f"msg_{uuid.uuid4().hex[:20]}"

                def _tool_use_result(tc: "ToolCallMessage") -> dict | None:
                    """Build a toolUseResult metadata block for file-writing tools."""
                    if tc.name == "Write":
                        return {
                            "type": "create",
                            "filePath": tc.input.get("file_path", ""),
                            "content": tc.input.get("content", ""),
                            "structuredPatch": [],
                            "originalFile": None,
                        }
                    if tc.name == "Edit":
                        old_str = tc.input.get("old_string", "")
                        new_str = tc.input.get("new_string", "")
                        return {
                            "filePath": tc.input.get("file_path", ""),
                            "oldString": old_str,
                            "newString": new_str,
                            "originalFile": old_str,
                            "structuredPatch": _structured_patch(old_str, new_str),
                            "userModified": False,
                            "replaceAll": tc.input.get("replace_all", False),
                        }
                    return None

                def _write_tool_batch(
                    tool_batch: list[tuple[str, "ToolCallMessage"]],
                    ts: str,
                ) -> None:
                    nonlocal prev_uuid
                    # One assistant record with all tool_use blocks
                    asst_uuid = str(uuid.uuid4())
                    asst_record = _make_base("assistant", asst_uuid, ts)
                    asst_record["message"] = {
                        "id": _msg_id(),
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": tc.name,
                                "input": tc.input,
                            }
                            for tool_use_id, tc in tool_batch
                        ],
                    }
                    f.write(json.dumps(asst_record) + "\n")
                    prev_uuid = asst_uuid

                    # One user record per tool_result (matches native CC format)
                    for tool_use_id, tc in tool_batch:
                        result_uuid = str(uuid.uuid4())
                        result_record = _make_base("user", result_uuid, ts)
                        result_record["message"] = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": tc.result,
                                }
                            ],
                        }
                        result_record["sourceToolAssistantUUID"] = asst_uuid
                        tur = _tool_use_result(tc)
                        if tur is not None:
                            result_record["toolUseResult"] = tur
                        f.write(json.dumps(result_record) + "\n")
                        prev_uuid = result_uuid

                turns = conv.turns
                i = 0
                while i < len(turns):
                    turn = turns[i]
                    ts = turn.timestamp.isoformat() if turn.timestamp else _now_iso()

                    if isinstance(turn, TextMessage) and turn.role == "user":
                        rec_uuid = str(uuid.uuid4())
                        record = _make_base("user", rec_uuid, ts)
                        record["message"] = {"role": "user", "content": turn.text}
                        f.write(json.dumps(record) + "\n")
                        prev_uuid = rec_uuid
                        i += 1

                    elif isinstance(turn, TextMessage) and turn.role == "assistant":
                        # Collect any tool calls that immediately follow this text block —
                        # they belong in the same assistant message per the Claude API contract.
                        content_blocks: list[dict] = [{"type": "text", "text": turn.text}]
                        tool_batch: list[tuple[str, ToolCallMessage]] = []
                        i += 1
                        while i < len(turns) and isinstance(turns[i], ToolCallMessage):
                            tc = turns[i]
                            tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": tc.name,
                                "input": tc.input,
                            })
                            tool_batch.append((tool_use_id, tc))
                            i += 1

                        asst_uuid = str(uuid.uuid4())
                        asst_record = _make_base("assistant", asst_uuid, ts)
                        asst_record["message"] = {
                            "id": _msg_id(),
                            "role": "assistant",
                            "content": content_blocks,
                        }
                        f.write(json.dumps(asst_record) + "\n")
                        prev_uuid = asst_uuid

                        if tool_batch:
                            for tuid, tc in tool_batch:
                                result_uuid = str(uuid.uuid4())
                                result_record = _make_base("user", result_uuid, ts)
                                result_record["message"] = {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tuid,
                                            "content": tc.result,
                                        }
                                    ],
                                }
                                result_record["sourceToolAssistantUUID"] = asst_uuid
                                tur = _tool_use_result(tc)
                                if tur is not None:
                                    result_record["toolUseResult"] = tur
                                f.write(json.dumps(result_record) + "\n")
                                prev_uuid = result_uuid

                    else:
                        # ToolCallMessage not preceded by assistant text — collect the
                        # full consecutive batch and emit as one assistant+user pair.
                        tool_batch = []
                        while i < len(turns) and isinstance(turns[i], ToolCallMessage):
                            tc = turns[i]
                            tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
                            tool_batch.append((tool_use_id, tc))
                            i += 1
                        _write_tool_batch(tool_batch, ts)

            # Atomic rename
            tmp_path.replace(final_path)

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return session_id

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        encoded = encode_project_path(project_path.resolve())
        jsonl_file = _projects_dir() / encoded / f"{conv_id}.jsonl"
        if jsonl_file.exists():
            jsonl_file.unlink()
