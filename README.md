# agent-migrator

A CLI tool for migrating conversation history between AI coding tools. Supports bidirectional migration between **Cursor** and **Claude Code**, with an interactive CLI for selecting conversations and tracking progress.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

**Install globally** (recommended — makes `agent-migrator` available on your PATH):

```bash
uv tool install agent-migrator
```

**Or run without installing:**

```bash
uvx agent-migrator
```

---

## Usage

Run from inside a project directory:

```bash
agent-migrator
```

Or pass a path explicitly:

```bash
agent-migrator /path/to/project
```

The CLI walks you through:

1. **Tool selection** — choose which tool to migrate from and which to migrate to.
2. **Conversation selection** — pick individual conversations or migrate all of them. Conversations are sorted most-recent-first and show the name, date, and size.
3. **Migration progress** — each conversation is processed in sequence. Press `Ctrl+C` at any time to cancel; any in-progress conversation is rolled back.
4. **Summary** — a results table shows what succeeded and any errors.

**Note:** For Cursor, the project path must be a directory that has been opened as a workspace in Cursor at least once. Cursor tracks conversations per workspace, so use the exact subdirectory you opened in Cursor (e.g. `my-repo/feature-branch`), not a parent directory.

---

## What gets migrated

### Cursor → Claude Code

| Feature | Status |
|---|---|
| Text messages (user and assistant) | ✓ |
| Tool calls (Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch) | ✓ |
| Bash mode commands and output | ✓ |
| Plan (most recent plan associated with the conversation) | ✓ |

### Claude Code → Cursor

| Feature | Status |
|---|---|
| Text messages (user and assistant) | ✓ |
| Tool calls (Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, MCP) | ✓ |
| Bash mode commands and output | ✓ |
| Plan | Partial — the plan document is migrated and visible to the Cursor agent as context, but Cursor does not treat it as a native plan in its plan mode UI |

---

## How it works

### Storage formats

**Cursor** stores conversations in a SQLite database at:
- Windows: `%APPDATA%\Cursor\User\globalStorage\state.vscdb`
- macOS: `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb`

Each conversation is a `composerData:<id>` record containing ordered bubble IDs and a `conversationState` protobuf that references content-addressed blobs. Individual messages are stored as `bubbleId:<composerId>:<bubbleId>` records (UI rendering) and `agentKv:blob:<sha256>` records (agent context). Tool calls use a `capabilityType: 15` marker with a `toolFormerData` object.

To associate a conversation with a project, Cursor writes a `workspace.json` file in a per-workspace directory under `workspaceStorage/`.

**Claude Code** stores conversations as JSONL files (one JSON object per line) at:
- `~/.claude/projects/<encoded-path>/<session-uuid>.jsonl`

The path encoding replaces path separators and colons with dashes (e.g. `C:/Users/me/project` becomes `C--Users-me-project`). Each record has a `type` field (`user`, `assistant`, etc.) and records are chained via `uuid`/`parentUuid` fields. Tool calls are split across two records: an assistant record with a `tool_use` block and a following user record with a `tool_result` block.

### Migration

Conversations are read into a tool-agnostic normalized format (`TextMessage` and `ToolCallMessage` turns), then written out in the destination format:

- **Cursor → Claude Code:** Each Cursor tool bubble is split into a `tool_use` assistant record and a `tool_result` user record. Plans are read from Cursor's plan registry and written as markdown files in `~/.claude/plans/`.

- **Claude Code → Cursor:** Paired `tool_use`/`tool_result` records are merged into single tool bubbles. The conversation history is uploaded to Cursor's server via the `ConvertOALToNAL` endpoint, which creates server-owned blobs for the `conversationState`. This ensures the Cursor agent has full prior context when resuming the conversation, with all models available. If the server upload fails (e.g. user not logged in), the CLI offers a local fallback using the `"claude-code"` agent backend — this writes a session JSONL to `~/.cursor-exp/` that provides full context but restricts model selection to Anthropic models.

Writes are atomic: Claude Code sessions are written to a `.tmp` file and renamed on success; Cursor writes use a SQLite transaction. If anything fails or the user cancels, the partial output is deleted.

### Extensibility

Each tool is implemented as a `ToolAdapter` subclass in `agent_migrator/tools/`. Adding support for a new tool means implementing the adapter interface (`list_conversations`, `read_conversation`, `write_conversation`, `delete_conversation`) and registering it in `cli.py`.
