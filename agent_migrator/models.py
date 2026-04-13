from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Union


class StandardToolName:
    """
    Canonical tool names used as the interchange format in ToolCallMessage.name.

    Every adapter's read_conversation() MUST map its native tool names to these
    constants before returning a Conversation. Every adapter's write_conversation()
    receives ToolCallMessage instances whose .name is one of these constants (or an
    unrecognised string, which adapters should pass through unchanged).
    """
    READ           = "Read"
    WRITE          = "Write"
    EDIT           = "Edit"
    MULTI_EDIT     = "MultiEdit"
    BASH           = "Bash"
    GLOB           = "Glob"
    GREP           = "Grep"
    WEB_FETCH      = "WebFetch"
    WEB_SEARCH     = "WebSearch"
    NOTEBOOK_READ  = "NotebookRead"
    NOTEBOOK_EDIT  = "NotebookEdit"
    TODO_WRITE     = "TodoWrite"
    TODO_READ      = "TodoRead"
    EXIT_PLAN_MODE = "ExitPlanMode"


@dataclass
class ConversationInfo:
    """Lightweight metadata used to populate the TUI conversation list."""
    id: str                  # composerId (Cursor) or session UUID (Claude Code)
    name: str                # human-readable display name
    updated_at: datetime
    created_at: datetime
    message_count: int       # approximate count of text turns
    size_bytes: int
    source_tool: str         # "cursor" | "claude_code"


@dataclass
class TextMessage:
    role: Literal["user", "assistant"]
    text: str
    timestamp: datetime | None = None


@dataclass
class ToolCallMessage:
    name: str
    input: dict
    result: str              # serialized result string
    timestamp: datetime | None = None


MessageTurn = Union[TextMessage, ToolCallMessage]


@dataclass
class Conversation:
    info: ConversationInfo
    turns: list[MessageTurn] = field(default_factory=list)
    plan_content: str | None = None
    # Raw markdown plan text (tool-agnostic). Each adapter is responsible for
    # persisting and restoring this in whatever way its tool stores plans.
    # Claude Code: stored as ~/.claude/plans/{slug}.md
    # Other tools: write to and read from their equivalent plan storage.
    model: str | None = None          # API model ID used (e.g. "claude-sonnet-4-6")


@dataclass
class MigrationResult:
    succeeded: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, new conv_id)
    failed: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, error message)
    cancelled: bool = False
