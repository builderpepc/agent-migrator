# Adding a New Tool to agent-migrator

This guide explains the architecture of agent-migrator and documents the contracts you must satisfy when adding support for a new AI coding tool.

---

## Architecture Overview

```
cli.py
  └── MigrationEngine (migrator.py)
        ├── SourceAdapter.read_conversation()   → Conversation (interchange format)
        └── DestAdapter.write_conversation()    ← Conversation (interchange format)
```

The migration pipeline is:

1. **Read**: the source adapter loads its native storage and returns a `Conversation` using the interchange format.
2. **Write**: the destination adapter receives that `Conversation` and persists it in its own native storage.

No adapter knows anything about another adapter's internals. All coupling goes through the interchange types in `models.py`.

---

## The Interchange Format (`models.py`)

### `Conversation`

```python
@dataclass
class Conversation:
    info: ConversationInfo      # lightweight metadata
    turns: list[MessageTurn]    # ordered list of messages and tool calls
    plan_content: str | None    # raw markdown plan (tool-agnostic, see below)
    model: str | None           # API model ID from the source session (informational)
```

### `TextMessage` and `ToolCallMessage`

```python
@dataclass
class TextMessage:
    role: Literal["user", "assistant"]
    text: str
    timestamp: datetime | None

@dataclass
class ToolCallMessage:
    name: str        # MUST be a StandardToolName constant (see below)
    input: dict      # tool arguments
    result: str      # serialised result string
    timestamp: datetime | None
```

### `StandardToolName`

`StandardToolName` is a class of string constants that form the canonical tool name vocabulary. `ToolCallMessage.name` MUST always be one of these constants after `read_conversation()` returns:

| Constant | Value |
|---|---|
| `StandardToolName.READ` | `"Read"` |
| `StandardToolName.WRITE` | `"Write"` |
| `StandardToolName.EDIT` | `"Edit"` |
| `StandardToolName.MULTI_EDIT` | `"MultiEdit"` |
| `StandardToolName.BASH` | `"Bash"` |
| `StandardToolName.GLOB` | `"Glob"` |
| `StandardToolName.GREP` | `"Grep"` |
| `StandardToolName.WEB_FETCH` | `"WebFetch"` |
| `StandardToolName.WEB_SEARCH` | `"WebSearch"` |
| `StandardToolName.NOTEBOOK_READ` | `"NotebookRead"` |
| `StandardToolName.NOTEBOOK_EDIT` | `"NotebookEdit"` |
| `StandardToolName.TODO_WRITE` | `"TodoWrite"` |
| `StandardToolName.TODO_READ` | `"TodoRead"` |
| `StandardToolName.EXIT_PLAN_MODE` | `"ExitPlanMode"` |

If a source tool uses a native tool name that has no standard equivalent, pass it through unchanged (i.e. keep `name` as the native string). The destination adapter should fall back gracefully for unrecognised names.

**Do not extend `StandardToolName` with tool-specific entries** — the constants represent operations that are generically meaningful across coding assistants. If a tool has a unique operation with no analogue, the native name passthrough is the right approach.

### `plan_content`

`plan_content` holds raw markdown text representing the structured plan the AI built before or during the conversation. It is entirely tool-agnostic — just a string.

- **On read**: extract whatever your tool stores as a "plan" and place the raw markdown in this field.
- **On write**: persist the markdown in whatever way your tool's UI will recognise it as a plan.
- **Claude Code** stores plans as `~/.claude/plans/{slug}.md` and discovers them via a `slug` field in the JSONL record.
- **If your tool has no plan concept**, leave `plan_content` as `None` on read and silently skip it on write.

---

## The `ToolAdapter` ABC (`tools/base.py`)

Create a subclass and implement all five methods:

```python
from agent_migrator.tools.base import ToolAdapter, ToolNetworkError

class MyToolAdapter(ToolAdapter):
    name = "My Tool"       # shown in the TUI
    tool_id = "my_tool"    # used in ConversationInfo.source_tool; must be unique

    def is_available(self) -> bool:
        # Return True if this tool is installed / its storage exists on this machine.
        return Path("...").exists()

    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        # Return conversations for project_path, sorted newest-first.
        # Must be fast — avoid loading full message content here.
        ...

    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        # Load the full conversation. Map all tool names to StandardToolName constants.
        ...

    def write_conversation(
        self,
        conv: Conversation,
        project_path: Path,
        *,
        use_local_backend: bool = False,
    ) -> str:
        # Persist the conversation atomically. Return the new conversation ID.
        # See "Atomic writes" and "use_local_backend" sections below.
        ...

    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        # Delete the conversation. Must be idempotent (no error if already gone).
        ...
```

### Registering the adapter

Add your adapter to the list in `cli.py`:

```python
all_adapters = [CursorAdapter(), ClaudeCodeAdapter(), MyToolAdapter()]
```

---

## Tool Name Mapping Contract

### On `read_conversation`

Your adapter receives native tool call records from disk. Before inserting them into `turns`, map each tool name to the closest `StandardToolName` constant:

```python
_MY_TOOL_TO_STANDARD: dict[str, str] = {
    "open_file":   StandardToolName.READ,
    "modify_file": StandardToolName.EDIT,
    "run_cmd":     StandardToolName.BASH,
    # etc.
}

name = _MY_TOOL_TO_STANDARD.get(native_name, native_name)  # passthrough if unknown
```

### On `write_conversation`

Your adapter receives `ToolCallMessage` instances whose `.name` is a standard constant (or an unrecognised passthrough). Map them back to your tool's native representation:

```python
_STANDARD_TO_MY_TOOL: dict[str, str] = {
    StandardToolName.READ:  "open_file",
    StandardToolName.EDIT:  "modify_file",
    StandardToolName.BASH:  "run_cmd",
    # etc.
}

native_name = _STANDARD_TO_MY_TOOL.get(tc.name, tc.name)
```

---

## `use_local_backend` and `ToolNetworkError`

Some tools support two write paths: uploading conversation history to a remote server (which makes it available across machines and in the cloud UI), and writing directly to local storage (which works offline but may limit functionality).

**If your tool has a remote upload path:**

1. By default (`use_local_backend=False`), attempt the remote upload.
2. If the upload fails for any network or auth reason, raise `ToolNetworkError` (or a subclass).
3. When `use_local_backend=True`, skip the upload and write locally.

```python
class MyToolUploadError(ToolNetworkError):
    """Raised when the remote upload to My Tool's server fails."""

def write_conversation(self, conv, project_path, *, use_local_backend=False):
    if not use_local_backend:
        try:
            self._upload_to_server(conv)
            return new_id
        except Exception as exc:
            raise MyToolUploadError(str(exc)) from exc
    # local fallback path
    self._write_local(conv, project_path)
    return new_id
```

`cli.py` catches `ToolNetworkError` (the base class) and offers the user a chance to retry with `use_local_backend=True`. It does not need to know which adapter or subclass is involved.

**If your tool has no remote path**, simply ignore `use_local_backend` — it defaults to `False` and the base class signature accepts it without any action required on your part.

---

## Atomic Write Requirement

`write_conversation` must leave no partial state on failure. The standard patterns:

**File-based storage** — write to a temp file first, then rename atomically:

```python
tmp = final_path.with_suffix(".tmp")
try:
    with open(tmp, "w") as f:
        # write everything
    tmp.replace(final_path)      # atomic on the same filesystem
except Exception:
    if tmp.exists():
        tmp.unlink()
    raise
```

**SQLite-based storage** — wrap all inserts in a single transaction:

```python
con = sqlite3.connect(db_path)
try:
    with con:          # auto-commits on exit, rolls back on exception
        con.execute(...)
        con.execute(...)
finally:
    con.close()
```

---

## Exception Hierarchy

```
Exception
└── ToolNetworkError                  (base.py)
    └── ServerUploadError             (cursor.py — Cursor-specific)
    └── YourToolUploadError           (your_tool.py — add here)
```

Only subclass `ToolNetworkError` for failures where `use_local_backend=True` is a meaningful retry. Other errors (file not found, corrupt data, etc.) should propagate as standard Python exceptions.

---

## Verification Checklist

After implementing your adapter, test these scenarios manually:

1. **Detection**: run `uv run agent-migrator` from a project directory — your tool should appear in the source/destination list.
2. **List**: conversations for the current project are listed with correct names, dates, and message counts.
3. **Read → Write (round-trip)**: migrate a multi-turn conversation with tool calls from your tool to Claude Code (or Cursor); open the result and confirm the history is readable.
4. **Write → Read (round-trip)**: migrate from Claude Code (or Cursor) into your tool; open the result in your tool's UI and confirm the history appears.
5. **Atomic write**: kill the process mid-write (or simulate an error); confirm no partial files or DB entries remain.
6. **`use_local_backend`** (if applicable): trigger a server upload failure; confirm `ToolNetworkError` is raised; accept the fallback prompt; confirm the local path succeeds.
7. **`plan_content`**: migrate a conversation that includes a plan; confirm the plan appears in the destination tool.
8. **Unknown tool names**: if the source contains tool calls not in `StandardToolName`, confirm they pass through without crashing.
