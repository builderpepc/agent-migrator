from __future__ import annotations

import base64
import json
import os
import platform
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
}


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
    turn: "ToolCallMessage", project_path: Path
) -> tuple[str, int, str, str, list]:
    """
    Map a Claude Code ToolCallMessage to Cursor's toolFormerData fields.
    Returns (cursor_tool_name, cursor_tool_num, params_json, result_json, code_blocks).

    code_blocks populates the bubble's top-level codeBlocks field, which is what
    Cursor's renderer reads to display file content and diffs inline.
    """
    name = turn.name
    inp = turn.input or {}
    raw_result = turn.result or ""

    # MCP tools (e.g. mcp_perplexity_perplexity_ask) → Cursor MCP type
    if name.startswith("mcp_"):
        return name, 19, json.dumps(inp), raw_result, []

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
    else:
        params = inp
        result = raw_result

    return cursor_name, tool_num, json.dumps(params), result, code_blocks


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

        info = ConversationInfo(
            id=conv_id,
            name=name,
            updated_at=_ms_to_dt(updated_ms),
            created_at=_ms_to_dt(created_ms),
            message_count=sum(1 for t in turns if isinstance(t, TextMessage)),
            size_bytes=0,
            source_tool=self.tool_id,
        )
        return Conversation(info=info, turns=turns)

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

        con = sqlite3.connect(str(global_db))
        try:
            with con:  # transaction — rolls back automatically on exception
                last_user_request_id = ""
                last_bubble_id = ""
                last_checkpoint_id = ""

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

                        if turn.role == "user":
                            request_id = str(uuid.uuid4())
                            checkpoint_id = str(uuid.uuid4())
                            last_user_request_id = request_id
                            last_checkpoint_id = checkpoint_id
                            bubble["requestId"] = request_id
                            bubble["checkpointId"] = checkpoint_id
                            bubble["richText"] = ""
                            bubble["cursorCommands"] = []
                            bubble["cursorCommandsExplicitlySet"] = False
                            bubble["pastChats"] = []
                            bubble["pastChatsExplicitlySet"] = False
                            bubble["context"] = _BUBBLE_CONTEXT
                            bubble["modelInfo"] = {"modelName": ""}
                            bubble["isPlanExecution"] = False
                            bubble["isNudge"] = False
                            bubble["skipRendering"] = False
                            bubble["editToolSupportsSearchAndReplace"] = True
                            bubble["workspaceProjectDir"] = ""
                            bubble["toolFormerData"] = {"additionalData": {}}
                        else:
                            # Assistant text bubbles
                            bubble["codeBlocks"] = []
                            bubble["attachedHumanChanges"] = False
                            bubble["usageUuid"] = last_user_request_id
                            bubble["symbolLinks"] = []
                            bubble["fileLinks"] = []
                    else:
                        # ToolCallMessage → capabilityType 15 bubble
                        conversation_items.append({"bubbleId": bubble_id, "type": _TYPE_ASSISTANT})
                        tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        cursor_name, tool_num, params_str, result_str, code_blocks = _adapt_cc_tool(
                            turn, project_path
                        )
                        bubble = _make_bubble_base(bubble_id, _TYPE_ASSISTANT, ts_iso)
                        bubble["capabilityType"] = _CAPABILITY_TOOL
                        bubble["codeBlocks"] = code_blocks
                        bubble["attachedHumanChanges"] = False
                        bubble["usageUuid"] = last_user_request_id
                        bubble["symbolLinks"] = []
                        bubble["fileLinks"] = []
                        # additionalData carries the codeblockId so the renderer can
                        # cross-reference codeBlocks entries by ID.
                        additional_data: dict = {}
                        if code_blocks:
                            additional_data["codeblockId"] = code_blocks[0]["codeblockId"]
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

                    con.execute(
                        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                        (f"bubbleId:{composer_id}:{bubble_id}", json.dumps(bubble)),
                    )

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
                is_agentic = any(isinstance(t, ToolCallMessage) for t in conv.turns)
                composer_data = {
                    "_v": 10,
                    "composerId": composer_id,
                    "name": conv.info.name,
                    "subtitle": "",
                    "richText": "",
                    "hasLoaded": True,
                    "text": "",
                    "fullConversationHeadersOnly": conversation_items,
                    "conversationMap": {},
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
                    "latestCheckpointId": last_checkpoint_id,
                    "currentBubbleId": last_bubble_id,
                    "editingBubbleId": "",
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
                    "modelConfig": {"modelName": "", "maxMode": False},
                    "subComposerIds": [],
                    "capabilityContexts": [],
                    "todos": [],
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
                    "isSpec": False,
                    "isSpecSubagentDone": False,
                    "stopHookLoopCount": 0,
                    "createdOnBranch": "",
                    "speculativeSummarizationEncryptionKey": speculative_key,
                    "isNAL": True,
                    "agentBackend": "cursor-agent",
                    "planModeSuggestionUsed": False,
                    "latestChatGenerationUUID": last_user_request_id,
                    "isAgentic": is_agentic,
                    "debugModeSuggestionUsed": False,
                }
                con.execute(
                    "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"composerData:{composer_id}", json.dumps(composer_data)),
                )

            # Register in the workspace's composer list (separate connection/transaction)
            ws_db = ws_dir / "state.vscdb"
            ws_con = sqlite3.connect(str(ws_db))
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

            # Update global composer.composerHeaders so Cursor's sidebar
            # immediately reflects the new conversation.
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
            gh_con = sqlite3.connect(str(global_db))
            try:
                with gh_con:
                    gh_row = gh_con.execute(
                        "SELECT value FROM ItemTable WHERE key = 'composer.composerHeaders'"
                    ).fetchone()
                    gh_data = json.loads(gh_row[0]) if (gh_row and gh_row[0]) else {"allComposers": []}
                    gh_data.setdefault("allComposers", [])
                    # Remove stale entry for this ID if present, then prepend.
                    gh_data["allComposers"] = [
                        c for c in gh_data["allComposers"] if c.get("composerId") != composer_id
                    ]
                    gh_data["allComposers"] = [global_header_entry] + gh_data["allComposers"]
                    gh_con.execute(
                        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                        ("composer.composerHeaders", json.dumps(gh_data)),
                    )
            finally:
                gh_con.close()

        finally:
            con.close()

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
        ws_con = sqlite3.connect(str(ws_db))
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
