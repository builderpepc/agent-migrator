from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Union


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
    plan_content: str | None = None  # raw markdown plan text (no YAML frontmatter)


@dataclass
class MigrationResult:
    succeeded: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, new conv_id)
    failed: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, error message)
    cancelled: bool = False
