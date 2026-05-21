# agent-migrator

A CLI tool for migrating conversation history between AI coding tools.

## Supported Coding Agents

- **Claude Code** (Anthropic)
- **Codex** (OpenAI)
- **Gemini CLI** (Google)
- **Cursor**

All pairs are supported bidirectionally.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

**Install globally** (recommended):

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

1. **Tool selection** — choose source and destination.
2. **Conversation selection** — pick conversations to migrate, sorted most-recent-first.
3. **Migration progress** — each conversation is processed in sequence; `Ctrl+C` cancels and rolls back.
4. **Summary** — a results table shows successes and errors.

> **Cursor:** the project path must be a directory that has been opened as a workspace in Cursor at least once.

---

## What gets migrated

All pairs support:

| Feature | Status |
|---|---|
| Text messages (user and assistant) | ✓ |
| Tool calls (Read, Write, Edit, Bash, Glob, and equivalents) | ✓ |
| Plan (presented natively in the destination tool's plan UI) | ✓ |

**Claude Code → Cursor** is the one exception: plans are migrated as context documents but are not surfaced in Cursor's native plan mode UI.

**Claude Code → Cursor** also requires a server upload to Cursor's `ConvertOALToNAL` endpoint to make all models available. If that fails (e.g. not logged in), the CLI offers a local fallback that provides full context but restricts model selection to Anthropic models.

---

## How it works

Conversations are read into a tool-agnostic normalized format (`TextMessage` and `ToolCallMessage` turns) and written out in the destination format. Each tool is implemented as a `AgentAdapter` subclass in `agent_migrator/tools/`. This decouples all source/destination pairs — adding a new tool requires only a new adapter; all existing migration paths continue to work unchanged.

### Extensibility

To add support for a new coding agent, implement the `AgentAdapter` interface in `agent_migrator/tools/` and register it in `cli.py`:

```python
class MyAgentAdapter(AgentAdapter):
    name = "My Agent"
    tool_id = "myagent"

    def is_available(self) -> bool: ...
    def list_conversations(self, project_path: Path) -> list[ConversationInfo]: ...
    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation: ...
    def write_conversation(self, conv: Conversation, project_path: Path, *, use_local_backend: bool = False) -> str: ...
    def delete_conversation(self, conv_id: str, project_path: Path) -> None: ...
```

### Storage locations

| Tool | Location |
|---|---|
| **Claude Code** | `~/.claude/projects/<encoded-path>/<session>.jsonl` |
| **Codex** | `~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl` |
| **Gemini CLI** | `~/.gemini/tmp/<project-slug>/chats/session-<timestamp>-<id>.jsonl` |
| **Cursor** | `%APPDATA%/Cursor/User/globalStorage/state.vscdb` (SQLite) |
