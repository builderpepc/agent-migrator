# agent-migrator

A CLI tool for migrating conversation history between AI coding tools. Supports bidirectional migration between **Cursor** and **Claude Code**, with an interactive terminal UI for selecting conversations and tracking progress.

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

The tool will launch an interactive TUI that walks you through:

1. **Tool selection** — choose which tool to migrate from and which to migrate to.
2. **Conversation selection** — pick individual conversations or migrate all of them. Conversations are sorted most-recent-first and show the name, date, and size.
3. **Migration progress** — a progress bar and live log show each conversation as it is processed. Press `Ctrl+C` at any time to cancel; any in-progress conversation is rolled back.
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
| Tool calls (Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch) | ✓ |
| Bash mode commands and output | ✓ |
| Plan | Partial — the plan document and todos are migrated, but Cursor does not treat the conversation as being in Plan Mode |

---

## How it works

### Storage formats

**Cursor** stores conversations in a SQLite database at:
- Windows: `%APPDATA%\Cursor\User\globalStorage\state.vscdb`
- macOS: `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb`

Each conversation is a `composerData:<id>` record listing the ordered bubble IDs. Individual messages are stored as `bubbleId:<composerId>:<bubbleId>` records. Tool calls (file edits, terminal commands, searches) are stored as separate bubbles with a `capabilityType: 15` marker and a `toolFormerData` object containing the call and its result.

To associate a conversation with a project, Cursor writes a `workspace.json` file in a per-workspace directory under `workspaceStorage/`. The tool uses this to find which conversations belong to a given path.

**Claude Code** stores conversations as JSONL files (one JSON object per line) at:
- `~/.claude/projects/<encoded-path>/<session-uuid>.jsonl`

The path encoding replaces path separators and colons with dashes (e.g. `C:/Users/me/project` becomes `C--Users-me-project`). Each record has a `type` field (`user`, `assistant`, etc.) and records are chained via `uuid`/`parentUuid` fields. Tool calls are split across two records: an assistant record containing a `tool_use` block, and a following user record containing a `tool_result` block.

### Migration

Conversations are read into a tool-agnostic normalized format (`TextMessage` and `ToolCallMessage` turns), then written out in the destination format:

- **Cursor to Claude Code:** Each Cursor tool bubble (`capabilityType: 15`) is split into a `tool_use` assistant record and a `tool_result` user record. Text bubbles become plain `user`/`assistant` records.
- **Claude Code to Cursor:** Paired `tool_use`/`tool_result` records are merged into a single `capabilityType: 15` bubble. Text records become type-1 (user) or type-2 (assistant) bubbles.

Writes are atomic: Claude Code sessions are written to a `.tmp` file and renamed on success; Cursor writes use a SQLite transaction. If anything fails or the user cancels, the partial output is deleted.

### Extensibility

Each tool is implemented as a `ToolAdapter` subclass in `agent_migrator/tools/`. Adding support for a new tool means implementing the adapter interface (`list_conversations`, `read_conversation`, `write_conversation`, `delete_conversation`) and registering it in `cli.py`.
