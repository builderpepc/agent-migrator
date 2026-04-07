from __future__ import annotations

import base64
import json
import os
import platform
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    TextMessage,
    ToolCallMessage,
)
from agent_migrator.tools.base import ToolAdapter

# Numeric bubble type constants (Cursor internal schema v2)
_TYPE_USER = 1
_TYPE_ASSISTANT = 2
_CAPABILITY_TOOL = 15
_CAPABILITY_SKIP = {22, 30}  # internal/streaming placeholders

# Reverse map: Cursor tool names → Claude Code tool names.
_CURSOR_TO_CC_TOOL_MAP: dict[str, str] = {
    "read_file":               "Read",
    "read_file_v2":            "Read",
    "edit_file":               "Edit",
    "edit_file_v2":            "Edit",
    "search_replace":          "Edit",   # rawArgs has file_path/old_string/new_string
    "write":                   "Write",  # rawArgs has file_path/contents
    "run_terminal_cmd":        "Bash",
    "run_terminal_command_v2": "Bash",
    "ripgrep_raw_search":      "Grep",
    "grep":                    "Grep",
    "glob_file_search":        "Glob",
    "file_search":             "Glob",
    "list_dir":                "Bash",
    "list_dir_v2":             "Bash",
    "web_search":              "WebSearch",
    "web_fetch":               "WebFetch",
    "codebase_search":         "Grep",
    "semantic_search_full":    "Grep",
}


def _adapt_cursor_tool(tfd: dict, bubble: dict) -> tuple[str, dict, str]:
    """
    Map a Cursor toolFormerData record to a CC-compatible (name, input, result).
    rawArgs carries the model's original arguments; params has the processed form.
    Falls back to params for tools where rawArgs is empty (e.g. edit_file_v2).
    """
    name = tfd.get("name", "unknown")
    raw_args_str = tfd.get("rawArgs", "")
    params_str = tfd.get("params", "")
    result_str = tfd.get("result", "")

    try:
        raw = json.loads(raw_args_str) if raw_args_str else {}
    except Exception:
        raw = {}
    try:
        params = json.loads(params_str) if params_str else {}
    except Exception:
        params = {}
    try:
        result_obj = json.loads(result_str) if result_str else {}
    except Exception:
        result_obj = {}

    cc_name = _CURSOR_TO_CC_TOOL_MAP.get(name, name)

    if cc_name == "Read":
        # rawArgs: {"path": "..."} — maps to CC file_path
        file_path = raw.get("path") or params.get("targetFile", "")
        cc_input = {"file_path": file_path}
        cc_result = result_obj.get("contents", result_str)

    elif cc_name == "Edit":
        if name == "search_replace":
            # rawArgs has full edit context — perfect round-trip
            cc_input = {
                "file_path":  raw.get("file_path", ""),
                "old_string": raw.get("old_string", ""),
                "new_string": raw.get("new_string", ""),
            }
            cc_result = "Applied edit."
        else:
            # edit_file_v2: rawArgs is always empty. Cursor stores the new file
            # content in params.streamingContent (falling back to codeBlocks).
            # Since we only have the final content (no old_string), map to Write
            # so the full file content is visible in CC history.
            file_path = raw.get("file_path") or params.get("relativeWorkspacePath", "")
            code_blocks = bubble.get("codeBlocks", [])
            new_content = (
                params.get("streamingContent", "")
                or (code_blocks[0].get("content", "") if code_blocks else "")
            )
            cc_name = "Write"
            cc_input = {"file_path": file_path, "content": new_content}
            cc_result = "File written successfully."

    elif cc_name == "Write":
        # rawArgs: {"file_path": "...", "contents": "..."}
        cc_input = {
            "file_path": raw.get("file_path") or params.get("relativeWorkspacePath", ""),
            "content":   raw.get("contents", raw.get("content", "")),
        }
        cc_result = "File written successfully."

    elif cc_name == "Bash":
        if name in ("list_dir", "list_dir_v2"):
            dir_path = raw.get("relative_workspace_path") or raw.get("path", ".")
            cc_input = {"command": f"ls {dir_path}"}
            files = result_obj.get("files") or result_obj.get("directoryTreeRoot", "")
            cc_result = json.dumps(files) if not isinstance(files, str) else files
        else:
            cc_input = {"command": raw.get("command", "")}
            cc_result = result_obj.get("output", result_str)

    elif cc_name == "Grep":
        if name in ("codebase_search", "semantic_search_full"):
            cc_input = {"pattern": raw.get("query", raw.get("search_query", ""))}
        else:
            cc_input = {
                "pattern": raw.get("pattern", ""),
                "path":    raw.get("path", ""),
            }
        cc_result = json.dumps(result_obj) if result_obj else result_str

    elif cc_name == "Glob":
        cc_input = {
            "pattern": raw.get("glob_pattern") or raw.get("globPattern", ""),
        }
        cc_result = json.dumps(result_obj) if result_obj else result_str

    elif cc_name == "WebFetch":
        cc_input = {"url": raw.get("url", "")}
        cc_result = result_obj.get("markdown", result_str)

    elif cc_name == "WebSearch":
        cc_input = {"query": raw.get("search_term") or raw.get("searchTerm", "")}
        refs = result_obj.get("references", result_str)
        cc_result = json.dumps(refs) if not isinstance(refs, str) else refs

    else:
        cc_input = raw or params
        cc_result = result_str

    return cc_name, cc_input, cc_result


# Map Claude Code tool names to Cursor equivalents (name, numeric tool ID).
# Cursor's UI only renders tool bubbles for known numeric IDs; tool=0 is invisible.
_CC_TOOL_MAP: dict[str, tuple[str, int]] = {
    "Read":         ("read_file_v2",       40),
    "Edit":         ("search_replace",     38),
    "Write":        ("write",              38),
    "MultiEdit":    ("search_replace",    38),
    "Bash":         ("run_terminal_cmd",   15),
    "Glob":         ("glob_file_search",   42),
    "Grep":         ("ripgrep_raw_search", 41),
    "WebFetch":     ("web_fetch",          57),
    "WebSearch":    ("web_search",         18),
    "NotebookRead": ("read_file_v2",       40),
    "NotebookEdit": ("edit_file_v2",       38),
    "TodoWrite":    ("todo_write",         35),
    "TodoRead":     ("read_file_v2",       40),
    # CC plan-mode tool — converted to create_plan (tool 43) so Cursor's plan
    # service recognises the conversation as plan-bearing and shows the plan UI.
    "ExitPlanMode":   ("create_plan", 43),
    "ExitPlanModeV2": ("create_plan", 43),
    "exit-plan-mode-v2": ("create_plan", 43),
}


# ---------------------------------------------------------------------------
# Minimal protobuf encoder (no external deps)
# ---------------------------------------------------------------------------

def _pb_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    buf = bytearray()
    while value > 0x7F:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value)
    return bytes(buf)


def _pb_field(field_num: int, wire_type: int, payload: bytes) -> bytes:
    """Encode a protobuf tag + payload."""
    tag = (field_num << 3) | wire_type
    return _pb_varint(tag) + payload


def _pb_string(field_num: int, value: str) -> bytes:
    """Encode a protobuf string field (wire type 2)."""
    encoded = value.encode("utf-8")
    return _pb_field(field_num, 2, _pb_varint(len(encoded)) + encoded)


def _pb_bytes(field_num: int, value: bytes) -> bytes:
    """Encode a protobuf bytes field (wire type 2)."""
    return _pb_field(field_num, 2, _pb_varint(len(value)) + value)


def _pb_message(field_num: int, payload: bytes) -> bytes:
    """Encode a protobuf embedded message field (wire type 2)."""
    return _pb_field(field_num, 2, _pb_varint(len(payload)) + payload)


def _new_blob_id() -> bytes:
    """Generate a random 32-byte blob ID (matches Cursor's random UUID-based IDs)."""
    return uuid.uuid4().bytes + uuid.uuid4().bytes  # 32 bytes


def _store_blob(con: "sqlite3.Connection", blob_id: bytes, data: bytes) -> None:
    """Write a blob to agentKv:blob:{hex_id} in cursorDiskKV."""
    con.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (f"agentKv:blob:{blob_id.hex()}", data),
    )


def _encode_tool_call_step(turn: "ToolCallMessage") -> bytes:
    """
    Encode a ToolCallMessage as a ConversationStep (proto agent.v1.ConversationStep)
    using field 2 (tool_call → TruncatedToolCall, field 34 in ToolCall).

    TruncatedToolCall (field 34 in ToolCall):
      field 1: original_step_blob_id (bytes) — any 32-byte id, unused for display
      field 3: result (TruncatedToolCallResult)
        field 1: success (TruncatedToolCallSuccess) — empty message

    We use TruncatedToolCall because:
    - It's designed for summarised/imported history
    - Does not require tool-specific argument encoding
    - Cursor still shows it correctly in conversation history
    """
    # TruncatedToolCallSuccess — empty message, no fields
    trunc_success = b""
    # TruncatedToolCallResult — field 1: success (embedded message)
    trunc_result = _pb_message(1, trunc_success)
    # TruncatedToolCall — field 1: original_step_blob_id, field 3: result
    placeholder_id = _new_blob_id()
    trunc_call = _pb_bytes(1, placeholder_id) + _pb_message(3, trunc_result)
    # ToolCall — field 34: truncated_tool_call
    tool_call_msg = _pb_message(34, trunc_call)
    # ConversationStep — field 2: tool_call (oneof)
    return _pb_message(2, tool_call_msg)


def _store_json_blob(con: "sqlite3.Connection", data: dict) -> bytes:
    """
    Serialize *data* as a compact JSON blob, store it in agentKv:blob:{hex},
    and return the 32-byte blob ID.

    These blobs are the ConversationStateStructure.rootPromptMessagesJson entries
    (field 1) — the server reads them to reconstruct conversation history.
    """
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob_id = _new_blob_id()
    _store_blob(con, blob_id, raw)
    return blob_id


def _write_agent_kv(conv: "Conversation", con: "sqlite3.Connection") -> str:
    """
    Write native binary protobuf blobs to agentKv:blob:{id} and return the
    ConversationStateStructure (bk) encoded as '~' + base64 for composerData.

    Cursor's agent stores two parallel representations of conversation history:

    A) rootPromptMessagesJson (field 1, repeated bytes) — blob IDs pointing to
       JSON message blobs in the format {"role":"user/assistant/tool","content":...}.
       The SERVER reads these when it needs to reconstruct the prior conversation.
       WITHOUT field 1, the agent sees no history and treats every new message
       as the first message in a fresh conversation.

    B) turns[] (field 8, repeated bytes) — blob IDs pointing to
       ConversationTurnStructure protobuf blobs.  Cursor's LOCAL QWe function
       reads these to build the UI conversation display (bubbles, headers).

    We write both so that:
    - The server has full context when the user sends a new message.
    - The Cursor UI shows prior bubbles correctly.
    """
    # -----------------------------------------------------------------------
    # Group turns: user message + following assistant/tool turns = one group.
    # -----------------------------------------------------------------------
    groups: list[tuple[str, list["MessageTurn"]]] = []
    current_user: str | None = None
    current_steps: list["MessageTurn"] = []

    for turn in conv.turns:
        if isinstance(turn, TextMessage) and turn.role == "user":
            if current_user is not None:
                groups.append((current_user, current_steps))
            current_user = turn.text
            current_steps = []
        else:
            if current_user is None:
                current_user = ""
            current_steps.append(turn)

    if current_user is not None:
        groups.append((current_user, current_steps))

    # -----------------------------------------------------------------------
    # Build field 1 (rootPromptMessagesJson) and field 8 (turns) simultaneously.
    #
    # Anthropic Messages API constraint: no two consecutive same-role messages.
    # Strategy: accumulate assistant content blocks (text + tool-calls) in a
    # pending buffer; flush to ONE assistant blob when a tool-call is seen
    # (so the tool-result can follow immediately), then reset the buffer.
    # Any remaining text at end-of-group becomes its own assistant blob.
    # -----------------------------------------------------------------------
    root_msg_blob_ids: list[bytes] = []
    turn_blob_ids: list[bytes] = []

    for user_text, steps in groups:
        # ---- A: rootPromptMessagesJson user message ----
        user_json_id = _store_json_blob(con, {
            "role": "user",
            "content": f"<user_query>\n{user_text}\n</user_query>",
        })
        root_msg_blob_ids.append(user_json_id)

        # ---- B: protobuf UserMessage blob (for field 8 turns) ----
        msg_id = str(uuid.uuid4())
        user_pb_blob = _pb_string(1, user_text) + _pb_string(2, msg_id)
        user_pb_id = _new_blob_id()
        _store_blob(con, user_pb_id, user_pb_blob)

        step_blob_ids: list[bytes] = []

        # Pending assistant content blocks (text and/or tool-calls) that will be
        # flushed as ONE assistant blob once a tool-call is encountered or at
        # end-of-steps.
        pending_asst: list[dict] = []

        def _flush_pending_asst() -> None:
            nonlocal pending_asst
            if not pending_asst:
                return
            # Native Cursor assistant blobs always have a top-level "id": "1".
            # This is required for the Cursor server to correctly parse the blob.
            asst_json_id = _store_json_blob(con, {
                "id": "1",
                "role": "assistant",
                "content": pending_asst,
            })
            root_msg_blob_ids.append(asst_json_id)
            pending_asst = []

        for step in steps:
            if isinstance(step, TextMessage):
                if step.role != "assistant":
                    continue
                # Accumulate; do NOT flush yet — wait for a tool-call or end.
                pending_asst.append({"type": "text", "text": step.text})

                # ---- B: protobuf AssistantMessage step ----
                step_blob = _pb_message(1, _pb_string(1, step.text))
                step_id = _new_blob_id()
                _store_blob(con, step_id, step_blob)
                step_blob_ids.append(step_id)

            else:
                # ToolCallMessage — add its tool-call block to pending, then
                # flush the whole batch (text prefix + this tool-call) as ONE
                # assistant blob, followed immediately by its tool-result blob.
                tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                cursor_name = _CC_TOOL_MAP.get(step.name, (step.name,))[0]

                pending_asst.append({
                    "type": "tool-call",
                    "toolCallId": tool_call_id,
                    "toolName": cursor_name,
                    "args": step.input or {},
                    "providerOptions": {
                        "cursor": {
                            "rawToolCallArgs": json.dumps(
                                step.input or {}, ensure_ascii=False
                            ),
                        },
                    },
                })
                _flush_pending_asst()

                # ---- A: rootPromptMessagesJson tool-result message ----
                # Native tool blobs have top-level "id" = the tool call ID,
                # and "providerOptions" at the top level.
                result_text = (
                    step.result
                    if isinstance(step.result, str)
                    else json.dumps(step.result)
                )
                tool_json_id = _store_json_blob(con, {
                    "role": "tool",
                    "id": tool_call_id,
                    "content": [{
                        "type": "tool-result",
                        "toolName": cursor_name,
                        "toolCallId": tool_call_id,
                        "result": result_text,
                    }],
                    "providerOptions": {
                        "cursor": {
                            "highLevelToolCallResult": {
                                "output": {"success": True},
                            },
                        },
                    },
                })
                root_msg_blob_ids.append(tool_json_id)

                # ---- B: protobuf TruncatedToolCall step ----
                step_blob = _encode_tool_call_step(step)
                step_id = _new_blob_id()
                _store_blob(con, step_id, step_blob)
                step_blob_ids.append(step_id)

        # Flush any remaining assistant text that came after the last tool-call
        # (or if there were only text steps with no tool-calls at all).
        _flush_pending_asst()

        # ---- B: protobuf ConversationTurnStructure ----
        turn_struct = _pb_bytes(1, user_pb_id)
        for sid in step_blob_ids:
            turn_struct += _pb_bytes(2, sid)
        turn_struct += _pb_string(3, str(uuid.uuid4()))

        conv_turn_struct = _pb_message(1, turn_struct)
        turn_id = _new_blob_id()
        _store_blob(con, turn_id, conv_turn_struct)
        turn_blob_ids.append(turn_id)

    # -----------------------------------------------------------------------
    # Build ConversationStateStructure (bk):
    #   field 1 (repeated bytes) = rootPromptMessagesJson blob IDs
    #   field 8 (repeated bytes) = ConversationTurnStructure blob IDs
    # -----------------------------------------------------------------------
    state_buf = bytearray()
    for mid in root_msg_blob_ids:
        state_buf += _pb_bytes(1, mid)
    for tid in turn_blob_ids:
        state_buf += _pb_bytes(8, tid)

    return "~" + base64.b64encode(bytes(state_buf)).decode()


# Map Claude Code API model IDs to Cursor's internal model name strings.
# Cursor uses its own naming scheme (observed from native conversation data).
_DEFAULT_CURSOR_MODEL = "claude-4.6-sonnet-medium-thinking"
_CC_TO_CURSOR_MODEL: dict[str, str] = {
    "claude-sonnet-4-6":              "claude-4.6-sonnet-medium-thinking",
    "claude-opus-4-6":                "claude-4.6-opus-medium-thinking",
    "claude-sonnet-4-5":              "claude-4.5-sonnet-thinking",
    "claude-sonnet-4-5-20250929":     "claude-4.5-sonnet-thinking",
    "claude-opus-4-5":                "claude-4.5-opus-high-thinking",
    "claude-opus-4-5-20251101":       "claude-4.5-opus-high-thinking",
    "claude-haiku-4-5":               "claude-4.5-haiku-thinking",
    "claude-haiku-4-5-20251001":      "claude-4.5-haiku-thinking",
    "claude-opus-4-20250514":         "claude-4.0-opus-thinking",
    "claude-sonnet-4-20250514":       "claude-4.0-sonnet-thinking",
    "claude-3-7-sonnet-20250219":     "claude-3.7-sonnet-thinking",
    "claude-3-5-sonnet-20241022":     "claude-3.5-sonnet",
    "claude-3-5-haiku-20241022":      "claude-3.5-haiku",
}


def _cc_model_to_cursor(api_model: str | None) -> str:
    """Return the Cursor model name for a CC API model ID, or the default."""
    if not api_model:
        return _DEFAULT_CURSOR_MODEL
    # Exact match first.
    if api_model in _CC_TO_CURSOR_MODEL:
        return _CC_TO_CURSOR_MODEL[api_model]
    # Prefix match for versioned IDs (e.g. "claude-sonnet-4-6-20260101").
    for cc_id, cursor_name in _CC_TO_CURSOR_MODEL.items():
        if api_model.startswith(cc_id):
            return cursor_name
    return _DEFAULT_CURSOR_MODEL


def _file_uri(file_path: str) -> dict:
    """Build a VS Code URI object for a file path (used in codeBlocks)."""
    # Normalise to forward slashes
    fwd = file_path.replace("\\", "/")
    # VS Code path: leading slash on Windows (e.g. /c:/Users/...)
    vsc_path = fwd if fwd.startswith("/") else f"/{fwd}"
    # URL-encoded form: colon → %3A
    encoded = vsc_path.replace(":", "%3A")
    # Windows fs path uses backslashes; strip the leading / we added
    fs_path = fwd.lstrip("/").replace("/", "\\") if "\\" in file_path or ":" in file_path else None
    return {
        "scheme": "file",
        "authority": "",
        "path": vsc_path,
        "query": "",
        "fragment": "",
        "_formatted": f"file://{encoded}",
        "_fsPath": fs_path or fwd,
    }


def _adapt_cc_tool(
    turn: "ToolCallMessage", project_path: Path, plan_uri: str = ""
) -> tuple[str, int, str, str, list, dict]:
    """
    Map a Claude Code ToolCallMessage to Cursor's toolFormerData fields.
    Returns (cursor_tool_name, cursor_tool_num, params_json, result_json,
             code_blocks, extra_bubble_fields).

    code_blocks populates the bubble's top-level codeBlocks field.
    extra_bubble_fields contains additional top-level bubble fields (e.g. todos
    for todo_write bubbles).
    """
    name = turn.name
    inp = turn.input or {}
    raw_result = turn.result or ""

    # MCP tools (e.g. mcp_perplexity_perplexity_ask) → Cursor MCP type
    if name.startswith("mcp_"):
        return name, 19, json.dumps(inp), raw_result, [], {}

    cursor_name, tool_num = _CC_TOOL_MAP.get(name, (name, 0))
    code_blocks: list = []

    # Build Cursor-format params, result, and codeBlocks based on tool type.
    if name == "Read":
        file_path = inp.get("file_path", "")
        params = {"targetFile": file_path}
        result = json.dumps({"contents": raw_result})
        # codeBlocks lets Cursor render the file content inline
        if file_path:
            cb_id = str(uuid.uuid4())
            code_blocks = [{
                "uri": _file_uri(file_path),
                "codeblockId": cb_id,
                "codeBlockIdx": 0,
                "content": raw_result,
            }]
    elif name == "Edit":
        file_path = inp.get("file_path", "")
        params = {"relativeWorkspacePath": file_path}
        old_str = inp.get("old_string", "")
        new_str = inp.get("new_string", "")
        diff_lines = [f"- {l}" for l in old_str.splitlines()] + \
                     [f"+ {l}" for l in new_str.splitlines()]
        result = json.dumps({
            "diff": {"chunks": [{"diffString": "\r\n".join(diff_lines)}]},
            "shouldAutoFixLints": False,
            "resultForModel": "Applied edit.",
        })
        if file_path:
            cb_id = str(uuid.uuid4())
            code_blocks = [{
                "uri": _file_uri(file_path),
                "codeblockId": cb_id,
                "codeBlockIdx": 0,
                "content": new_str,
            }]
    elif name == "MultiEdit":
        file_path = inp.get("file_path", "")
        params = {"relativeWorkspacePath": file_path}
        chunks = []
        all_new = []
        for edit in inp.get("edits", []):
            old_str = edit.get("old_string", "")
            new_str = edit.get("new_string", "")
            diff_lines = [f"- {l}" for l in old_str.splitlines()] + \
                         [f"+ {l}" for l in new_str.splitlines()]
            chunks.append({"diffString": "\r\n".join(diff_lines)})
            all_new.append(new_str)
        result = json.dumps({
            "diff": {"chunks": chunks},
            "shouldAutoFixLints": False,
            "resultForModel": "Applied edits.",
        })
        if file_path:
            cb_id = str(uuid.uuid4())
            code_blocks = [{
                "uri": _file_uri(file_path),
                "codeblockId": cb_id,
                "codeBlockIdx": 0,
                "content": "\n".join(all_new),
            }]
    elif name == "Write":
        file_path = inp.get("file_path", "")
        content = inp.get("content", "")
        params = {"relativeWorkspacePath": file_path}
        diff_lines = ["- "] + [f"+ {l}" for l in content.splitlines()]
        result = json.dumps({
            "diff": {"chunks": [{"diffString": "\r\n".join(diff_lines)}]},
            "shouldAutoFixLints": False,
            "resultForModel": "Created file.",
        })
        if file_path:
            cb_id = str(uuid.uuid4())
            code_blocks = [{
                "uri": _file_uri(file_path),
                "codeblockId": cb_id,
                "codeBlockIdx": 0,
                "content": content,
            }]
    elif name == "Bash":
        cmd = inp.get("command", "")
        params = {"command": cmd, "requireUserApproval": False}
        result = json.dumps({"output": raw_result})
    elif name == "Glob":
        params = {
            "globPattern": inp.get("pattern", ""),
            "targetDirectory": str(project_path),
        }
        result = json.dumps({"files": raw_result})
    elif name == "Grep":
        params = {
            "pattern": inp.get("pattern", ""),
            "path": inp.get("path", str(project_path)),
        }
        result = json.dumps({"output": raw_result})
    elif name == "WebFetch":
        url = inp.get("url", "")
        params = {"url": url}
        result = json.dumps({"url": url, "markdown": raw_result})
    elif name == "WebSearch":
        params = {"searchTerm": inp.get("query", "")}
        result = json.dumps({"references": raw_result})
    elif name in ("NotebookRead", "TodoRead"):
        file_path = inp.get("notebook_path", inp.get("file_path", ""))
        params = {"targetFile": file_path}
        result = json.dumps({"contents": raw_result})
        if file_path:
            cb_id = str(uuid.uuid4())
            code_blocks = [{
                "uri": _file_uri(file_path),
                "codeblockId": cb_id,
                "codeBlockIdx": 0,
                "content": raw_result,
            }]
    elif name in ("NotebookEdit", "TodoWrite"):
        file_path = inp.get("notebook_path", inp.get("file_path", ""))
        params = {"relativeWorkspacePath": file_path}
        result = json.dumps({"success": True})
    elif name in ("ExitPlanMode", "ExitPlanModeV2", "exit-plan-mode-v2"):
        # Convert CC plan-mode exit to a Cursor create_plan bubble (tool=43).
        # Cursor's ComposerPlanService.getPlanBubbleData() scans FCHO for a
        # bubble with toolFormerData.tool === 43 to activate the plan UI.
        plan_text = inp.get("plan", "")
        # Extract a title from the first H1, falling back to a generic name.
        h1 = re.search(r"^#\s+(.+)$", plan_text, re.MULTILINE)
        plan_name = h1.group(1).strip() if h1 else "Migrated Plan"
        # Extract overview from the first non-heading paragraph.
        overview_match = re.search(r"^(?!#)(.+)$", plan_text, re.MULTILINE)
        plan_overview = overview_match.group(1).strip() if overview_match else ""
        todos = _extract_todos_from_markdown(plan_text) if plan_text else []
        params = {
            "plan": plan_text,
            "name": plan_name,
            "overview": plan_overview,
            "todos": todos,
        }
        additional_data: dict = {}
        if plan_uri:
            additional_data["planUri"] = plan_uri
        result = json.dumps({"success": True})
        # Also surface todos as a top-level bubble field so the UI can render
        # the todo list (same as native todo_write bubbles).
        extra_fields = {
            "todos": [json.dumps(t) for t in todos],
            "_additional_data_override": additional_data,
        }
        return cursor_name, tool_num, json.dumps(params), result, code_blocks, extra_fields
    else:
        params = inp
        result = raw_result

    return cursor_name, tool_num, json.dumps(params), result, code_blocks, {}


def _global_db_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ["APPDATA"])
        return base / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    else:
        return Path.home() / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb"


def _workspace_storage_dir() -> Path:
    return _global_db_path().parent.parent / "workspaceStorage"


def _cursor_plans_dir() -> Path:
    """~/.cursor/plans/ — where Cursor stores .plan.md files."""
    return Path.home() / ".cursor" / "plans"


_CURSOR_PLAN_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _strip_cursor_plan_frontmatter(content: str) -> str:
    """Return just the markdown body of a Cursor .plan.md (strip YAML frontmatter)."""
    m = _CURSOR_PLAN_FRONTMATTER_RE.match(content)
    return content[m.end():].strip() if m else content.strip()


def _build_cursor_plan_id(name: str) -> str:
    """
    Generate a Cursor plan ID: lowercase-underscored name + underscore + 8-hex hash.
    e.g. "My Plan" -> "my_plan_a1b2c3d4"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    suffix = secrets.token_hex(4)
    return f"{slug}_{suffix}"


def _extract_todos_from_markdown(md: str) -> list[dict]:
    """
    Extract structured todos from a CC plan (pure markdown).
    Priority: checkboxes → H2 headings → top-level numbered items → single catch-all.
    Returns [{id, content, status}].
    """
    # 1. Checkbox items: - [x] done  /  - [ ] pending
    checkbox_re = re.compile(r"^[-*]\s+\[([ xX])\]\s+(.+)$", re.MULTILINE)
    todos = []
    for m in checkbox_re.finditer(md):
        done = m.group(1).strip().lower() == "x"
        content = m.group(2).strip()
        slug_id = re.sub(r"[^a-z0-9]+", "-", content.lower())[:40].strip("-")
        todos.append({"id": slug_id, "content": content, "status": "completed" if done else "pending"})
    if todos:
        return todos

    # 2. Level-2 headings (major plan phases, skip the H1 title)
    heading_re = re.compile(r"^##\s+(?:\d+[.)]\s+)?(.+)$", re.MULTILINE)
    for i, m in enumerate(heading_re.finditer(md), 1):
        content = m.group(1).strip()
        slug_id = re.sub(r"[^a-z0-9]+", "-", content.lower())[:40].strip("-") or str(i)
        todos.append({"id": f"{i}-{slug_id}", "content": content, "status": "pending"})
    if todos:
        return todos

    # 3. Top-level numbered list items (only lines not preceded by indentation)
    numbered_re = re.compile(r"^(\d+)[.)]\s+\*{0,2}(.+?)\*{0,2}$", re.MULTILINE)
    for m in numbered_re.finditer(md):
        i = int(m.group(1))
        content = m.group(2).strip().rstrip(":")
        slug_id = re.sub(r"[^a-z0-9]+", "-", content.lower())[:40].strip("-") or str(i)
        todos.append({"id": f"{i}-{slug_id}", "content": content, "status": "pending"})
    if todos:
        return todos

    # 4. Single catch-all
    return [{"id": "implement-plan", "content": "Implement plan", "status": "pending"}]


def _build_cursor_plan_file(name: str, plan_content: str) -> str:
    """
    Convert a CC plan (pure markdown) to a Cursor .plan.md file.
    Generates YAML frontmatter from the plan content and appends the body.
    """
    # Overview: first non-blank, non-heading paragraph after the H1 title
    body_lines = plan_content.splitlines()
    overview = ""
    past_h1 = False
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            if overview:  # stop at blank line after finding content
                break
            continue
        if stripped.startswith("#"):
            past_h1 = True
            continue
        if past_h1:
            overview = stripped
            # Keep collecting until blank line (handled above)

    todos = _extract_todos_from_markdown(plan_content)

    # Build YAML frontmatter (simple manual serialisation to avoid a dep on PyYAML)
    def yaml_str(s: str) -> str:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    lines = ["---", f"name: {name}", f"overview: {overview}", "todos:"]
    for t in todos:
        lines.append(f"  - id: {t['id']}")
        lines.append(f"    content: {yaml_str(t['content'])}")
        lines.append(f"    status: {t['status']}")
    lines.append("---")
    lines.append("")
    lines.append(plan_content)
    return "\n".join(lines)


def _encode_cursor_projects_path(project_path: Path) -> str:
    """
    Encode a project path for use in ~/.cursor/projects/{encoded}/.
    C:\\Users\\troyh\\...\\project → c-Users-troyh-...-project
    (lowercase drive letter, remove colon, all separators → dash)
    """
    s = str(project_path)
    # Windows: lowercase the drive letter and remove the colon (C: → c)
    if len(s) >= 2 and s[1] == ":":
        s = s[0].lower() + s[2:]
    # Replace path separators with dashes
    s = s.replace("\\", "-").replace("/", "-")
    # Collapse consecutive dashes and trim
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _write_agent_transcript(conv: "Conversation", composer_id: str, project_path: Path) -> None:
    """
    Write ~/.cursor/projects/{encoded}/agent-transcripts/{composer_id}/{composer_id}.jsonl

    The Cursor agent reads this file to reconstruct conversation history context when
    the user resumes a conversation. Without it the agent cannot see any prior work.

    Format (one JSON object per line):
    - user turn:      {"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\n...\\n</user_query>"}]}}
    - assistant turn: {"role":"assistant","message":{"content":[text_block?, ...tool_use_blocks]}}

    Tool *result* messages are NOT written — native transcripts never include them.
    Consecutive assistant turns (text + tool calls) between two user messages are
    merged into one assistant entry.
    """
    encoded = _encode_cursor_projects_path(project_path)
    transcript_dir = (
        Path.home() / ".cursor" / "projects" / encoded
        / "agent-transcripts" / composer_id
    )
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = transcript_dir / f"{composer_id}.jsonl"

    lines: list[str] = []
    pending_asst: list[dict] = []

    def _flush_asst() -> None:
        nonlocal pending_asst
        if pending_asst:
            lines.append(json.dumps(
                {"role": "assistant", "message": {"content": pending_asst}},
                ensure_ascii=False, separators=(",", ":"),
            ))
            pending_asst = []

    for turn in conv.turns:
        if isinstance(turn, TextMessage):
            if turn.role == "user":
                _flush_asst()
                lines.append(json.dumps(
                    {"role": "user", "message": {"content": [
                        {"type": "text", "text": f"<user_query>\n{turn.text}\n</user_query>"},
                    ]}},
                    ensure_ascii=False, separators=(",", ":"),
                ))
            else:
                pending_asst.append({"type": "text", "text": turn.text})
        else:
            # ToolCallMessage — represent as tool_use; no tool_result in transcript.
            # Native Cursor transcripts use only type/name/input (no "id" field).
            cursor_name = _CC_TOOL_MAP.get(turn.name, (turn.name,))[0]
            pending_asst.append({
                "type": "tool_use",
                "name": cursor_name,
                "input": turn.input,
            })

    _flush_asst()

    transcript_file.write_text("\n".join(lines), encoding="utf-8")


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _folder_uri_to_path(uri: str) -> Path:
    """Convert a VS Code folder URI (file:///c%3A/...) to a Path."""
    parsed = urlparse(uri)
    decoded = unquote(parsed.path)
    # On Windows the path starts with /c:/... — strip the leading slash
    if platform.system() == "Windows" and decoded.startswith("/"):
        decoded = decoded[1:]
    return Path(decoded)


def _find_workspace_dir(project_path: Path) -> Path | None:
    """
    Walk workspaceStorage looking for the directory whose workspace.json
    folder URI resolves to *project_path*.

    Comparison is case-insensitive on Windows (os.path.normcase handles this).
    """
    ws_dir = _workspace_storage_dir()
    if not ws_dir.exists():
        return None
    target = os.path.normcase(str(project_path.resolve()))
    for entry in ws_dir.iterdir():
        ws_json = entry / "workspace.json"
        if not ws_json.exists():
            continue
        try:
            data = json.loads(ws_json.read_text(encoding="utf-8"))
            folder_uri = data.get("folder", "")
            if not folder_uri:
                continue
            # Only handle local file:// URIs; skip vscode-remote:// etc.
            if not folder_uri.startswith("file://"):
                continue
            candidate = os.path.normcase(str(_folder_uri_to_path(folder_uri).resolve()))
            if candidate == target:
                return entry
        except Exception:
            continue
    return None


def _get_composer_ids(workspace_db: Path) -> list[str]:
    """Read the list of composerIds for a workspace.

    Priority order:
    1. workspace DB composer.composerData.allComposers (legacy / temporary)
    2. Global composer.composerHeaders filtered by this workspace's hash ID
       (this is what Cursor uses as its authoritative sidebar source)
    3. Workspace DB pane values — each value is a dict keyed by
       'workbench.panel.aichat.view.<composerId>', giving only IDs for panes
       actually opened in this workspace (avoids cross-workspace contamination).
    """
    if not workspace_db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{workspace_db}?mode=ro", uri=True)
        row = con.execute(
            "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
        ).fetchone()

        data = json.loads(row[0]) if (row and row[0]) else {}
        all_composers = [c["composerId"] for c in data.get("allComposers", []) if c.get("composerId")]

        if all_composers:
            con.close()
            return all_composers

        # Primary fallback: global composer.composerHeaders filtered by workspace hash.
        workspace_hash = workspace_db.parent.name
        global_db = _global_db_path()
        if global_db.exists():
            gcon = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True)
            grow = gcon.execute(
                "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'"
            ).fetchone()
            gcon.close()
            if grow and grow[0]:
                try:
                    gh = json.loads(grow[0])
                    ids = [
                        c["composerId"]
                        for c in gh.get("allComposers", [])
                        if c.get("composerId")
                        and c.get("workspaceIdentifier", {}).get("id") == workspace_hash
                    ]
                    if ids:
                        con.close()
                        return ids
                except Exception:
                    pass

        # Last resort: parse pane values from this workspace's DB.
        pane_rows = con.execute(
            "SELECT value FROM ItemTable WHERE key LIKE 'workbench.panel.composerChatViewPane.%'"
            " AND key NOT LIKE '%.hidden'"
        ).fetchall()
        con.close()

        prefix = "workbench.panel.aichat.view."
        seen: set[str] = set()
        composer_ids: list[str] = []
        for (val,) in pane_rows:
            if not val:
                continue
            try:
                d = json.loads(val)
            except Exception:
                continue
            for vk in d.keys():
                if vk.startswith(prefix):
                    cid = vk[len(prefix):]
                    if cid and cid not in seen:
                        seen.add(cid)
                        composer_ids.append(cid)
        return composer_ids
    except Exception:
        return []


def _bubble_ids_from_composer(data: dict) -> list[dict]:
    """
    Return an ordered list of {bubbleId, ...} dicts from a composerData record,
    supporting both current (fullConversationHeadersOnly) and older (conversation) schemas.
    """
    # Newer schema (Cursor _v 13+): fullConversationHeadersOnly
    headers = data.get("fullConversationHeadersOnly")
    if headers:
        return headers
    # Older schema: conversation array
    return data.get("conversation") or []


def _first_user_text(global_db: sqlite3.Connection, composer_id: str, bubble_items: list) -> str:
    """Return the text of the first user bubble, for use as a fallback name."""
    for item in bubble_items:
        bid = item.get("bubbleId")
        if not bid:
            continue
        row = global_db.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"bubbleId:{composer_id}:{bid}",),
        ).fetchone()
        if not row or not row[0]:
            continue
        bubble = json.loads(row[0])
        if bubble.get("type") == _TYPE_USER and bubble.get("text", "").strip():
            return bubble["text"].strip()[:60]
    return composer_id[:8]


class CursorAdapter(ToolAdapter):
    name = "Cursor"
    tool_id = "cursor"

    def is_available(self) -> bool:
        return _global_db_path().exists()

    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        ws_dir = _find_workspace_dir(project_path)
        if ws_dir is None:
            return []

        composer_ids = _get_composer_ids(ws_dir / "state.vscdb")
        if not composer_ids:
            return []

        global_db = _global_db_path()
        con = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True)
        results: list[ConversationInfo] = []

        for cid in composer_ids:
            row = con.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"composerData:{cid}",),
            ).fetchone()
            if not row or not row[0]:
                continue
            try:
                data = json.loads(row[0])
            except Exception:
                continue

            bubble_items = _bubble_ids_from_composer(data)
            name = data.get("name", "").strip()
            if not name:
                name = _first_user_text(con, cid, bubble_items)

            updated_ms = data.get("lastUpdatedAt") or data.get("createdAt") or _now_ms()
            created_ms = data.get("createdAt") or updated_ms

            # Total bubble count (user + assistant + tool calls) — a more honest
            # measure of conversation size than user-only turns, since agentic
            # conversations can have hundreds of tool call bubbles per user message.
            text_turn_count = len(bubble_items)

            # Approximate size: sum of raw value bytes for all bubbles in this conversation
            size = 0
            for item in bubble_items:
                bid = item.get("bubbleId")
                if bid:
                    brow = con.execute(
                        "SELECT length(value) FROM cursorDiskKV WHERE key = ?",
                        (f"bubbleId:{cid}:{bid}",),
                    ).fetchone()
                    if brow and brow[0]:
                        size += brow[0]

            results.append(ConversationInfo(
                id=cid,
                name=name,
                updated_at=_ms_to_dt(updated_ms),
                created_at=_ms_to_dt(created_ms),
                message_count=text_turn_count,
                size_bytes=size,
                source_tool=self.tool_id,
            ))

        con.close()
        results.sort(key=lambda c: c.updated_at, reverse=True)
        return results

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        global_db = _global_db_path()
        con = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True)

        row = con.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"composerData:{conv_id}",),
        ).fetchone()
        if not row or not row[0]:
            con.close()
            raise ValueError(f"Cursor conversation not found: {conv_id}")

        data = json.loads(row[0])
        bubble_items = _bubble_ids_from_composer(data)
        name = data.get("name", "").strip() or conv_id[:8]
        updated_ms = data.get("lastUpdatedAt") or data.get("createdAt") or _now_ms()
        created_ms = data.get("createdAt") or updated_ms

        turns = []
        for item in bubble_items:
            bid = item.get("bubbleId")
            if not bid:
                continue
            brow = con.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"bubbleId:{conv_id}:{bid}",),
            ).fetchone()
            if not brow or not brow[0]:
                continue
            try:
                bubble = json.loads(brow[0])
            except Exception:
                continue

            btype = bubble.get("type")
            capability = bubble.get("capabilityType")
            text = bubble.get("text", "").strip()

            if btype == _TYPE_USER:
                if text:
                    turns.append(TextMessage(role="user", text=text))
            elif btype == _TYPE_ASSISTANT:
                if capability in _CAPABILITY_SKIP:
                    continue
                elif capability == _CAPABILITY_TOOL:
                    tfd = bubble.get("toolFormerData") or {}
                    tool_name, input_dict, result = _adapt_cursor_tool(tfd, bubble)
                    turns.append(ToolCallMessage(
                        name=tool_name,
                        input=input_dict,
                        result=result,
                    ))
                else:
                    # Regular assistant text bubble
                    if text:
                        turns.append(TextMessage(role="assistant", text=text))

        con.close()

        # Read any plan associated with this conversation via planRegistry.
        plan_content: str | None = None
        global_db = _global_db_path()
        if global_db.exists():
            try:
                gcon = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True)
                grow = gcon.execute(
                    "SELECT value FROM ItemTable WHERE key = 'composer.planRegistry'"
                ).fetchone()
                gcon.close()
                if grow and grow[0]:
                    registry = json.loads(grow[0])
                    # Collect all plans associated with this conversation, pick most recent.
                    matching: list[tuple[int, Path]] = []
                    for plan_meta in registry.values():
                        if conv_id in plan_meta.get("referencedBy", []) or \
                                conv_id == plan_meta.get("createdBy"):
                            uri = plan_meta.get("uri", {})
                            plan_path = Path(uri.get("fsPath", ""))
                            if plan_path.exists():
                                ts = plan_meta.get("lastUpdatedAt", 0)
                                matching.append((ts, plan_path))
                    if matching:
                        matching.sort(key=lambda x: x[0], reverse=True)
                        _, plan_path = matching[0]
                        plan_content = _strip_cursor_plan_frontmatter(
                            plan_path.read_text(encoding="utf-8")
                        ) or None
            except Exception:
                pass

        info = ConversationInfo(
            id=conv_id,
            name=name,
            updated_at=_ms_to_dt(updated_ms),
            created_at=_ms_to_dt(created_ms),
            message_count=sum(1 for t in turns if isinstance(t, TextMessage)),
            size_bytes=0,
            source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=turns, plan_content=plan_content)

    def write_conversation(self, conv: Conversation, project_path: Path) -> str:
        ws_dir = _find_workspace_dir(project_path)
        if ws_dir is None:
            raise RuntimeError(
                f"No Cursor workspace found for {project_path}. "
                "Open this directory in Cursor at least once first."
            )

        global_db = _global_db_path()
        composer_id = str(uuid.uuid4())
        now_ms = _now_ms()
        conversation_items = []

        # Capability statuses required on every bubble (all empty lists = not yet run)
        _CAPABILITY_STATUSES = {
            "mutate-request": [], "start-submit-chat": [], "before-submit-chat": [],
            "chat-stream-finished": [], "before-apply": [], "after-apply": [],
            "accept-all-edits": [], "composer-done": [], "process-stream": [],
            "add-pending-action": [],
        }

        # Context structure for user bubbles (matches native _v:3 bubble context)
        _BUBBLE_CONTEXT = {
            "composers": [], "quotes": [], "selectedCommits": [],
            "selectedPullRequests": [], "selectedImages": [],
            "folderSelections": [], "fileSelections": [], "terminalFiles": [],
            "selections": [], "terminalSelections": [], "selectedDocs": [],
            "externalLinks": [], "cursorRules": [], "cursorCommands": [],
            "uiElementSelections": [], "consoleLogs": [], "ideEditorsState": [],
            "mentions": {
                "composers": {}, "selectedCommits": {}, "selectedPullRequests": {},
                "gitDiff": [], "gitDiffFromBranchToMain": [], "selectedImages": {},
                "folderSelections": {}, "fileSelections": {}, "terminalFiles": {},
                "selections": {}, "terminalSelections": {}, "selectedDocs": {},
                "externalLinks": {}, "diffHistory": [], "cursorRules": {},
                "cursorCommands": {}, "uiElementSelections": [], "consoleLogs": [],
                "ideEditorsState": [], "gitPRDiffSelections": {},
                "subagentSelections": {}, "browserSelections": {},
            },
        }

        def _make_bubble_base(bubble_id: str, btype: int, ts_iso: str) -> dict:
            """Common _v:3 fields shared by all bubble types."""
            return {
                "_v": 3,
                "type": btype,
                "approximateLintErrors": [],
                "lints": [],
                "codebaseContextChunks": [],
                "commits": [],
                "pullRequests": [],
                "attachedCodeChunks": [],
                "assistantSuggestedDiffs": [],
                "gitDiffs": [],
                "interpreterResults": [],
                "images": [],
                "attachedFolders": [],
                "attachedFoldersNew": [],
                "bubbleId": bubble_id,
                "userResponsesToSuggestedCodeBlocks": [],
                "suggestedCodeBlocks": [],
                "diffsForCompressingFiles": [],
                "relevantFiles": [],
                "toolResults": [],
                "notepads": [],
                "capabilities": [],
                "capabilityStatuses": _CAPABILITY_STATUSES,
                "multiFileLinterErrors": [],
                "diffHistories": [],
                "recentLocationsHistory": [],
                "recentlyViewedFiles": [],
                "isAgentic": False,
                "fileDiffTrajectories": [],
                "existedSubsequentTerminalCommand": False,
                "existedPreviousTerminalCommand": False,
                "docsReferences": [],
                "webReferences": [],
                "aiWebSearchResults": [],
                "requestId": "",
                "attachedFoldersListDirResults": [],
                "humanChanges": [],
                "summarizedComposers": [],
                "cursorRules": [],
                "contextPieces": [],
                "editTrailContexts": [],
                "allThinkingBlocks": [],
                "diffsSinceLastApply": [],
                "deletedFiles": [],
                "supportedTools": [],
                "tokenCount": {"inputTokens": 0, "outputTokens": 0},
                "attachedFileCodeChunksMetadataOnly": [],
                "consoleLogs": [],
                "uiElementPicked": [],
                "isRefunded": False,
                "knowledgeItems": [],
                "documentationSelections": [],
                "externalLinks": [],
                "useWeb": False,
                "projectLayouts": [],
                "unifiedMode": 2,
                "capabilityContexts": [],
                "todos": [],
                "createdAt": ts_iso,
                "mcpDescriptors": [],
                "workspaceUris": [],
                "text": "",
                # Present on all native bubbles; empty string = no serialized state,
                # read content from the bubble's own fields.
                "conversationState": "",
            }

        # Capabilities list matching native _v:10 composers (conversations with content)
        _COMPOSER_CAPABILITIES = [
            {"type": 30, "data": {}},
            {"type": 34, "data": {}},
            {"type": 15, "data": {"bubbleDataMap": "{}"}},
            {"type": 22, "data": {}},
            {"type": 18, "data": {}},
            {"type": 19, "data": {}},
            {"type": 33, "data": {}},
            {"type": 32, "data": {}},
            {"type": 23, "data": {}},
            {"type": 16, "data": {}},
            {"type": 24, "data": {}},
            {"type": 25, "data": {}},
            {"type": 21, "data": {}},
            {"type": 31, "data": {}},
            {"type": 29, "data": {}},
        ]

        # Pre-compute plan file path and URI before the bubble loop so that
        # _adapt_cc_tool can embed planUri in the create_plan bubble's additionalData.
        plan_id: str | None = None
        plan_entry: dict | None = None
        plan_uri: str = ""
        if conv.plan_content:
            _plan_name = conv.info.name or "Migrated Plan"
            _h1 = re.search(r"^#\s+(.+)$", conv.plan_content, re.MULTILINE)
            if _h1:
                _plan_name = _h1.group(1).strip()
            plan_id = _build_cursor_plan_id(_plan_name)
            _plans_dir = _cursor_plans_dir()
            _plans_dir.mkdir(parents=True, exist_ok=True)
            _plan_file = _plans_dir / f"{plan_id}.plan.md"
            _plan_file.write_text(
                _build_cursor_plan_file(_plan_name, conv.plan_content),
                encoding="utf-8",
            )
            _plan_uri_path = str(_plan_file).replace("\\", "/")
            plan_uri = f"file:///{_plan_uri_path.lstrip('/')}"
            plan_entry = {
                "id": plan_id,
                "name": _plan_name,
                "uri": {
                    "$mid": 1,
                    "fsPath": str(_plan_file),
                    "_sep": 1,
                    "external": plan_uri,
                    "path": f"/{_plan_uri_path.lstrip('/')}",
                    "scheme": "file",
                },
                "createdBy": composer_id,
                "editedBy": [composer_id],
                "referencedBy": [composer_id],
                "builtBy": {},
                "lastUpdatedAt": now_ms,
                "createdAt": now_ms,
            }

        con = sqlite3.connect(str(global_db), timeout=30)
        try:
            with con:  # transaction — rolls back on exception; con closed in finally
                last_user_request_id = ""
                last_bubble_id = ""
                last_checkpoint_id = ""
                first_todo_bubble_id = ""
                plan_todos: list = []
                is_agentic = any(isinstance(t, ToolCallMessage) for t in conv.turns)
                # Collect every bubble dict so we can populate conversationMap.
                # Cursor's getLoadedConversation() looks up each bubbleId from
                # fullConversationHeadersOnly in conversationMap — if a bubbleId is
                # missing it stops and treats the conversation as empty.  Native
                # conversations rely on a loadBubblesByIds() call to fill this map,
                # but that pipeline is skipped when hasLoaded=True.  We pre-populate
                # the map ourselves so Cursor sees the full history immediately.
                conversation_map: dict = {}

                for turn in conv.turns:
                    bubble_id = str(uuid.uuid4())
                    last_bubble_id = bubble_id
                    ts_iso = (
                        turn.timestamp.isoformat()
                        if turn.timestamp
                        else datetime.now(timezone.utc).isoformat()
                    )

                    if isinstance(turn, TextMessage):
                        btype = _TYPE_USER if turn.role == "user" else _TYPE_ASSISTANT
                        conversation_items.append({"bubbleId": bubble_id, "type": btype})
                        bubble = _make_bubble_base(bubble_id, btype, ts_iso)
                        bubble["text"] = turn.text
                        # Native Cursor: isAgentic=True only on user bubbles;
                        # all assistant bubbles (text and tool) use False.
                        bubble["isAgentic"] = is_agentic if turn.role == "user" else False

                        if turn.role == "user":
                            request_id = str(uuid.uuid4())
                            checkpoint_id = str(uuid.uuid4())
                            last_user_request_id = request_id
                            last_checkpoint_id = checkpoint_id
                            bubble["requestId"] = request_id
                            bubble["checkpointId"] = checkpoint_id
                            bubble["richText"] = ""
                            bubble["context"] = _BUBBLE_CONTEXT
                            bubble["modelInfo"] = {"modelName": ""}
                            bubble["isPlanExecution"] = False
                            bubble["isNudge"] = False
                            bubble["skipRendering"] = False
                            bubble["editToolSupportsSearchAndReplace"] = True
                        else:
                            # Assistant text bubbles: requestId stays "" (empty) to
                            # match native Cursor behaviour. usageUuid links this
                            # response back to the triggering user request.
                            bubble["codeBlocks"] = []
                            bubble["attachedHumanChanges"] = False
                            bubble["usageUuid"] = last_user_request_id
                            bubble["modelInfo"] = {"modelName": ""}
                            bubble["timingInfo"] = {}
                    else:
                        # ToolCallMessage → capabilityType 15 bubble.
                        # Tool bubbles have neither requestId nor usageUuid set
                        # (both stay as "" from _make_bubble_base) — matching
                        # native Cursor's pattern.
                        conversation_items.append({"bubbleId": bubble_id, "type": _TYPE_ASSISTANT})
                        tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        cursor_name, tool_num, params_str, result_str, code_blocks, extra_fields = _adapt_cc_tool(
                            turn, project_path, plan_uri=plan_uri
                        )
                        bubble = _make_bubble_base(bubble_id, _TYPE_ASSISTANT, ts_iso)
                        bubble["capabilityType"] = _CAPABILITY_TOOL
                        bubble["codeBlocks"] = code_blocks
                        bubble["attachedHumanChanges"] = False
                        bubble["isAgentic"] = False
                        # additionalData: start with codeblockId (for file tools),
                        # then allow the tool adapter to override/extend it (e.g.
                        # create_plan needs planUri here).
                        additional_data: dict = {}
                        if code_blocks:
                            additional_data["codeblockId"] = code_blocks[0]["codeblockId"]
                        if "_additional_data_override" in extra_fields:
                            additional_data.update(extra_fields.pop("_additional_data_override"))
                        bubble["toolFormerData"] = {
                            "toolCallId": tool_call_id,
                            "toolIndex": 0,
                            "modelCallId": tool_call_id,
                            "status": "completed",
                            "name": cursor_name,
                            "rawArgs": json.dumps(turn.input),
                            "tool": tool_num,
                            "params": params_str,
                            "result": result_str,
                            "additionalData": additional_data,
                        }
                        # Merge any extra top-level fields (e.g. `todos`).
                        bubble.update(extra_fields)
                        # Track the create_plan bubble so composerData.firstTodoWriteBubble
                        # points to it (Cursor uses this field to locate the plan anchor).
                        if cursor_name == "create_plan" and not first_todo_bubble_id:
                            first_todo_bubble_id = bubble_id
                            plan_todos = [json.loads(t) for t in extra_fields.get("todos", [])]

                    con.execute(
                        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                        (f"bubbleId:{composer_id}:{bubble_id}", json.dumps(bubble)),
                    )
                    conversation_map[bubble_id] = bubble

                    # Write auxiliary records and the cap:30 anchor bubble for
                    # every user bubble.
                    if isinstance(turn, TextMessage) and turn.role == "user":
                        msg_ctx = {
                            "terminalFiles": [],
                            "cursorRules": [],
                            "attachedFoldersListDirResults": [],
                            "summarizedComposers": [],
                        }
                        con.execute(
                            "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                            (f"messageRequestContext:{composer_id}:{request_id}", json.dumps(msg_ctx)),
                        )
                        chk_state = {
                            "files": [],
                            "nonExistentFiles": [],
                            "newlyCreatedFolders": [],
                            "activeInlineDiffs": [],
                            "inlineDiffNewlyCreatedResources": {"files": [], "folders": []},
                        }
                        con.execute(
                            "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                            (f"checkpointId:{composer_id}:{checkpoint_id}", json.dumps(chk_state)),
                        )
                        # Insert a capabilityType:30 "thinking anchor" bubble
                        # immediately after each user bubble. Native Cursor
                        # conversations always have one of these — Cursor uses
                        # the matching requestId to locate the start of the
                        # assistant's response turn when building API context.
                        cap30_id = str(uuid.uuid4())
                        cap30 = _make_bubble_base(cap30_id, _TYPE_ASSISTANT, ts_iso)
                        cap30["capabilityType"] = 30
                        cap30["requestId"] = request_id   # same as user bubble — the anchor
                        cap30["usageUuid"] = request_id
                        cap30["codeBlocks"] = []
                        cap30["attachedHumanChanges"] = False
                        cap30["modelInfo"] = {"modelName": _cc_model_to_cursor(conv.model)}
                        cap30["timingInfo"] = {}
                        cap30["serverBubbleId"] = str(uuid.uuid4())
                        conversation_items.append({"bubbleId": cap30_id, "type": _TYPE_ASSISTANT})
                        con.execute(
                            "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                            (f"bubbleId:{composer_id}:{cap30_id}", json.dumps(cap30)),
                        )
                        conversation_map[cap30_id] = cap30

                # If the conversation has a plan but no ExitPlanMode tool call
                # produced a create_plan bubble, synthesize one now.  This happens
                # when CC stored the plan in a separate plan file rather than
                # embedding it in an ExitPlanMode turn.
                if conv.plan_content and not first_todo_bubble_id:
                    _ts_now = datetime.now(timezone.utc).isoformat()
                    cp_bubble_id = str(uuid.uuid4())
                    _h1 = re.search(r"^#\s+(.+)$", conv.plan_content, re.MULTILINE)
                    _plan_name = _h1.group(1).strip() if _h1 else (conv.info.name or "Migrated Plan")
                    _ov = re.search(r"^(?!#)(.+)$", conv.plan_content, re.MULTILINE)
                    _plan_overview = _ov.group(1).strip() if _ov else ""
                    _todos = _extract_todos_from_markdown(conv.plan_content)
                    _cp_params = json.dumps({
                        "plan": conv.plan_content,
                        "name": _plan_name,
                        "overview": _plan_overview,
                        "todos": _todos,
                    })
                    _cp_add = {"planUri": plan_uri} if plan_uri else {}
                    cp_bubble = _make_bubble_base(cp_bubble_id, _TYPE_ASSISTANT, _ts_now)
                    cp_bubble["capabilityType"] = _CAPABILITY_TOOL
                    cp_bubble["codeBlocks"] = []
                    cp_bubble["attachedHumanChanges"] = False
                    cp_bubble["isAgentic"] = False
                    cp_bubble["todos"] = [json.dumps(t) for t in _todos]
                    cp_bubble["toolFormerData"] = {
                        "toolCallId": str(uuid.uuid4()),
                        "toolIndex": 0,
                        "modelCallId": str(uuid.uuid4()),
                        "status": "completed",
                        "name": "create_plan",
                        "rawArgs": _cp_params,
                        "tool": 43,
                        "params": _cp_params,
                        "result": json.dumps({"success": True}),
                        "additionalData": _cp_add,
                    }
                    conversation_items.append({"bubbleId": cp_bubble_id, "type": _TYPE_ASSISTANT})
                    con.execute(
                        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                        (f"bubbleId:{composer_id}:{cp_bubble_id}", json.dumps(cp_bubble)),
                    )
                    conversation_map[cp_bubble_id] = cp_bubble
                    first_todo_bubble_id = cp_bubble_id
                    plan_todos = _todos

                # Build the agent context store entries and conversationState.
                # Cursor's agent reads history from agentKv:blob: entries keyed by
                # SHA-256 hash; conversationState is a protobuf listing those hashes.
                # Individual bubbleId: keys are only used for UI rendering.
                conversation_state = _write_agent_kv(conv, con)

                # Generate a random encryption key for speculative summarization.
                # Cursor uses this to encrypt summarization data for this conversation.
                # A fresh random key allows new messages to be sent normally.
                speculative_key = base64.b64encode(secrets.token_bytes(32)).decode()

                _COMPOSER_CONTEXT = {
                    "composers": [], "selectedCommits": [], "selectedPullRequests": [],
                    "selectedImages": [], "folderSelections": [], "fileSelections": [],
                    "selections": [], "terminalSelections": [], "selectedDocs": [],
                    "externalLinks": [], "cursorRules": [], "cursorCommands": [],
                    "gitPRDiffSelections": [], "subagentSelections": [],
                    "browserSelections": [],
                    "mentions": {
                        "composers": {}, "selectedCommits": {}, "selectedPullRequests": {},
                        "gitDiff": [], "gitDiffFromBranchToMain": [], "selectedImages": {},
                        "folderSelections": {}, "fileSelections": {}, "terminalFiles": {},
                        "selections": {}, "terminalSelections": {}, "selectedDocs": {},
                        "externalLinks": {}, "diffHistory": [], "cursorRules": {},
                        "cursorCommands": {}, "uiElementSelections": [], "consoleLogs": [],
                        "ideEditorsState": [], "gitPRDiffSelections": {},
                        "subagentSelections": {}, "browserSelections": {},
                    },
                }
                composer_data = {
                    "_v": 14,
                    "composerId": composer_id,
                    "name": conv.info.name,
                    "subtitle": "",
                    "richText": "",
                    "hasLoaded": True,
                    "text": "",
                    "fullConversationHeadersOnly": conversation_items,
                    "conversationMap": conversation_map,
                    "status": "completed",
                    "context": _COMPOSER_CONTEXT,
                    "gitGraphFileSuggestions": [],
                    "generatingBubbleIds": [],
                    "isReadingLongFile": False,
                    "codeBlockData": {},
                    "originalFileStates": {},
                    "newlyCreatedFiles": [],
                    "newlyCreatedFolders": [],
                    "lastUpdatedAt": now_ms,
                    "createdAt": now_ms,
                    "hasChangedContext": False,
                    "activeTabsShouldBeReactive": True,
                    "capabilities": _COMPOSER_CAPABILITIES,
                    "isFileListExpanded": False,
                    "browserChipManuallyDisabled": False,
                    "browserChipManuallyEnabled": False,
                    "unifiedMode": "agent",
                    "forceMode": "edit",
                    "usageData": {},
                    "contextUsagePercent": 0,
                    "contextTokensUsed": 0,
                    "contextTokenLimit": 0,
                    "allAttachedFileCodeChunksUris": [],
                    "modelConfig": {"modelName": _cc_model_to_cursor(conv.model), "maxMode": False},
                    "subComposerIds": [],
                    "capabilityContexts": [],
                    "isQueueExpanded": True,
                    "hasUnreadMessages": False,
                    "gitHubPromptDismissed": False,
                    "totalLinesAdded": 0,
                    "totalLinesRemoved": 0,
                    "addedFiles": 0,
                    "removedFiles": 0,
                    "filesChangedCount": 0,
                    "isArchived": False,
                    "isDraft": False,
                    "isCreatingWorktree": False,
                    "isApplyingWorktree": False,
                    "isUndoingWorktree": False,
                    "applied": False,
                    "pendingCreateWorktree": False,
                    "isBestOfNSubcomposer": False,
                    "isBestOfNParent": False,
                    "bestOfNJudgeWinner": False,
                    "isSpec": False,
                    "isSpecSubagentDone": False,
                    "stopHookLoopCount": 0,
                    "speculativeSummarizationEncryptionKey": speculative_key,
                    "isNAL": True,
                    "agentBackend": "cursor-agent",
                    # conversationState encodes the full message history as a
                    # protobuf of SHA-256 hashes pointing to agentKv:blob: entries.
                    "conversationState": conversation_state,
                    "planModeSuggestionUsed": False,
                    "latestChatGenerationUUID": last_user_request_id,
                    "isAgentic": is_agentic,
                    "debugModeSuggestionUsed": False,
                    # Plan support: populate todos and point to the first todo_write bubble.
                    "todos": plan_todos,
                    "firstTodoWriteBubble": first_todo_bubble_id,
                }
                con.execute(
                    "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"composerData:{composer_id}", json.dumps(composer_data)),
                )

        finally:
            con.close()

        # Register in the workspace's composer list.
        ws_db = ws_dir / "state.vscdb"
        ws_con = sqlite3.connect(str(ws_db), timeout=30)
        try:
            with ws_con:
                row = ws_con.execute(
                    "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
                ).fetchone()
                if row and row[0]:
                    existing = json.loads(row[0])
                else:
                    existing = {}
                # Cursor workspaces use one of two formats for tracking composers:
                #   Old: allComposers[]  — ordered list of composer metadata objects
                #   New: selectedComposerIds[] + lastFocusedComposerIds[] — ID-only lists
                # Detect which format this workspace uses and update accordingly.
                new_entry = {
                    "type": "head",
                    "composerId": composer_id,
                    "name": conv.info.name,
                    "lastUpdatedAt": now_ms,
                    "createdAt": now_ms,
                    "unifiedMode": "agent",
                    "forceMode": "edit",
                    "hasUnreadMessages": False,
                    "isArchived": False,
                    "isDraft": False,
                    "isWorktree": False,
                    "isSpec": False,
                    "isProject": False,
                    "isBestOfNSubcomposer": False,
                    "numSubComposers": 0,
                    "referencedPlans": [],
                    "totalLinesAdded": 0,
                    "totalLinesRemoved": 0,
                    "filesChangedCount": 0,
                    "worktreeStartedReadOnly": False,
                }
                # Always update allComposers — Cursor uses this list for the
                # sidebar regardless of whether hasMigratedComposerData is set.
                existing.setdefault("allComposers", [])
                existing["allComposers"] = [new_entry] + existing["allComposers"]
                # Also update the ID-only lists used by migrated workspaces.
                existing.setdefault("selectedComposerIds", [])
                existing.setdefault("lastFocusedComposerIds", [])
                if composer_id not in existing["selectedComposerIds"]:
                    existing["selectedComposerIds"] = [composer_id] + existing["selectedComposerIds"]
                if composer_id not in existing["lastFocusedComposerIds"]:
                    existing["lastFocusedComposerIds"] = [composer_id] + existing["lastFocusedComposerIds"]
                ws_con.execute(
                    "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                    ("composer.composerData", json.dumps(existing)),
                )
        finally:
            ws_con.close()

        # Update global ItemTable in a single connection:
        #   - composer.composerHeaders  (always)
        #   - composer.planRegistry     (if plan)
        # Using referencedPlans in composerHeaders entry (if plan).
        ws_json = ws_dir / "workspace.json"
        try:
            ws_json_data = json.loads(ws_json.read_text(encoding="utf-8"))
            folder_uri = ws_json_data.get("folder", "")
        except Exception:
            folder_uri = ""
        workspace_identifier = {
            "id": ws_dir.name,
            "uri": {
                "$mid": 1,
                "fsPath": str(project_path).replace("/", "\\"),
                "_sep": 1,
                "external": folder_uri,
                "path": "/" + str(project_path).replace("\\", "/").lstrip("/"),
                "scheme": "file",
            },
        }
        global_header_entry = {**new_entry, "workspaceIdentifier": workspace_identifier}
        gh_con = sqlite3.connect(str(global_db), timeout=30)
        try:
            with gh_con:
                # composerHeaders
                gh_row = gh_con.execute(
                    "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'"
                ).fetchone()
                gh_data = json.loads(gh_row[0]) if (gh_row and gh_row[0]) else {"allComposers": []}
                gh_data.setdefault("allComposers", [])
                gh_data["allComposers"] = [
                    c for c in gh_data["allComposers"] if c.get("composerId") != composer_id
                ]
                gh_data["allComposers"] = [global_header_entry] + gh_data["allComposers"]

                # Attach referencedPlans to the new header entry if we have a plan.
                if plan_id:
                    for c in gh_data["allComposers"]:
                        if c.get("composerId") == composer_id:
                            c["referencedPlans"] = [plan_id]
                            break

                gh_con.execute(
                    "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                    ("composer.composerHeaders", json.dumps(gh_data)),
                )

                # planRegistry
                if plan_entry:
                    pr_row = gh_con.execute(
                        "SELECT value FROM ItemTable WHERE key = 'composer.planRegistry'"
                    ).fetchone()
                    registry = json.loads(pr_row[0]) if (pr_row and pr_row[0]) else {}
                    registry[plan_id] = plan_entry
                    gh_con.execute(
                        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                        ("composer.planRegistry", json.dumps(registry)),
                    )
        finally:
            gh_con.close()

        # Remove the agent-transcript directory for this conversation (if any).
        try:
            import shutil
            encoded = _encode_cursor_projects_path(project_path)
            transcript_dir = (
                Path.home() / ".cursor" / "projects" / encoded
                / "agent-transcripts" / conv_id
            )
            if transcript_dir.exists():
                shutil.rmtree(transcript_dir, ignore_errors=True)
        except Exception:
            pass

        # Write the agent-transcripts JSONL so Cursor's agent can see the full
        # conversation history when the user resumes this migrated conversation.
        _write_agent_transcript(conv, composer_id, project_path)

        return composer_id

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        global_db = _global_db_path()
        con = sqlite3.connect(str(global_db))
        try:
            with con:
                con.execute(
                    "DELETE FROM cursorDiskKV WHERE key = ?",
                    (f"composerData:{conv_id}",),
                )
                con.execute(
                    "DELETE FROM cursorDiskKV WHERE key LIKE ?",
                    (f"bubbleId:{conv_id}:%",),
                )
        finally:
            con.close()

        # Remove from workspace composer list
        ws_dir = _find_workspace_dir(project_path)
        if ws_dir is None:
            return
        ws_db = ws_dir / "state.vscdb"
        if not ws_db.exists():
            return
        ws_con = sqlite3.connect(str(ws_db), timeout=30)
        try:
            with ws_con:
                row = ws_con.execute(
                    "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
                ).fetchone()
                if not row or not row[0]:
                    return
                existing = json.loads(row[0])
                existing["allComposers"] = [
                    c for c in existing.get("allComposers", [])
                    if c.get("composerId") != conv_id
                ]
                existing["selectedComposerIds"] = [
                    x for x in existing.get("selectedComposerIds", []) if x != conv_id
                ]
                existing["lastFocusedComposerIds"] = [
                    x for x in existing.get("lastFocusedComposerIds", []) if x != conv_id
                ]
                ws_con.execute(
                    "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                    ("composer.composerData", json.dumps(existing)),
                )
        finally:
            ws_con.close()

        # Remove from global composer.composerHeaders so Cursor's sidebar
        # reflects the deletion immediately.
        gh_con = sqlite3.connect(str(global_db))
        try:
            with gh_con:
                gh_row = gh_con.execute(
                    "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'"
                ).fetchone()
                if gh_row and gh_row[0]:
                    gh_data = json.loads(gh_row[0])
                    gh_data["allComposers"] = [
                        c for c in gh_data.get("allComposers", [])
                        if c.get("composerId") != conv_id
                    ]
                    gh_con.execute(
                        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                        ("composer.composerHeaders", json.dumps(gh_data)),
                    )
        finally:
            gh_con.close()
