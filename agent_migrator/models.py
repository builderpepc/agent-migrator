from __future__ import annotations

import re
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
    source_agent: str        # "cursor" | "claude_code"


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


_PROPOSED_PLAN_RE = re.compile(r'<proposed_plan>', re.IGNORECASE)
_PROPOSED_PLAN_BLOCK_RE = re.compile(
    r'\s*<proposed_plan>.*?</proposed_plan>\s*', re.DOTALL | re.IGNORECASE
)


def _strip_proposed_plan_tag(text: str) -> str:
    """Remove <proposed_plan>...</proposed_plan> blocks from assistant text.

    Used when writing to tools that represent plans via a dedicated tool call
    (ExitPlanMode) rather than inline XML — leaving the raw tags would show
    as literal text in those tools' UIs.
    """
    return _PROPOSED_PLAN_BLOCK_RE.sub("\n", text).strip()


def inject_exit_plan_mode(
    turns: list["MessageTurn"],
    plan_content: str,
) -> list["MessageTurn"]:
    """
    Ensure an ExitPlanMode ToolCallMessage appears in the turns when a plan
    is present.  If one already exists, the list is returned unchanged.
    Otherwise the new turn is inserted immediately after the first assistant
    TextMessage that contains a <proposed_plan> tag (or appended at the end
    if no such message is found).

    Any <proposed_plan>...</proposed_plan> block is stripped from the
    assistant text that precedes the injected ExitPlanMode, since the plan
    content is already carried by ExitPlanMode.input.plan.

    This is called by write_conversation() in each adapter so that the
    destination tool renders a plan panel rather than plain assistant text.
    """
    # Already has an ExitPlanMode — nothing to do.
    if any(
        isinstance(t, ToolCallMessage) and t.name == StandardToolName.EXIT_PLAN_MODE
        for t in turns
    ):
        return turns

    # Build the result string in native CC format so CC renders the plan
    # in historical sessions (renderToolResultMessage reads from this string).
    epm_result = (
        "User has approved your plan. You can now start coding."
        " Start with updating your todo list if applicable"
        f"\n\n## Approved Plan:\n{plan_content}"
    )
    epm: "MessageTurn" = ToolCallMessage(
        name=StandardToolName.EXIT_PLAN_MODE,
        input={"plan": plan_content},
        result=epm_result,
    )

    # Find the assistant turn that presented the proposed plan.
    for i, turn in enumerate(turns):
        if (
            isinstance(turn, TextMessage)
            and turn.role == "assistant"
            and _PROPOSED_PLAN_RE.search(turn.text)
        ):
            result = list(turns)
            # Replace the matching turn with a version that has the XML stripped.
            stripped_text = _strip_proposed_plan_tag(turn.text)
            result[i] = TextMessage(
                role=turn.role,
                text=stripped_text,
                timestamp=turn.timestamp,
            )
            result.insert(i + 1, epm)
            return result

    # Fallback: no <proposed_plan> marker found — append at end.
    return list(turns) + [epm]


@dataclass
class MigrationResult:
    succeeded: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, new conv_id)
    failed: list[tuple[ConversationInfo, str]] = field(default_factory=list)
    # (original ConversationInfo, error message)
    cancelled: bool = False
