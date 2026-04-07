from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agent_migrator.models import Conversation, ConversationInfo


class ToolAdapter(ABC):
    """
    Abstract base class for an AI coding tool's conversation storage.

    To add support for a new tool:
    1. Subclass ToolAdapter and implement all abstract methods.
    2. Add an instance of the subclass to the adapter list in cli.py.
    """

    #: Human-readable display name shown in the TUI (e.g. "Cursor")
    name: str

    #: Stable identifier used in ConversationInfo.source_tool (e.g. "cursor")
    tool_id: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this tool's storage directory exists on this system."""

    @abstractmethod
    def list_conversations(self, project_path: Path) -> list[ConversationInfo]:
        """
        Return all conversations for *project_path*, sorted newest-first.

        This must be fast — it is called from a background worker while the TUI
        is already running.  Implementations should avoid loading full message
        content; only read the metadata needed for ConversationInfo.
        """

    @abstractmethod
    def read_conversation(self, conv_id: str, project_path: Path) -> Conversation:
        """Load the full Conversation (all turns) for *conv_id*."""

    @abstractmethod
    def write_conversation(
        self, conv: Conversation, project_path: Path, **kwargs
    ) -> str:
        """
        Persist *conv* as a new conversation for *project_path*.

        Implementations must write atomically: use a temp file / SQLite
        transaction so that an exception leaves no partial state.

        Returns the new conversation ID assigned by this tool.
        """

    @abstractmethod
    def delete_conversation(self, conv_id: str, project_path: Path) -> None:
        """
        Delete the conversation identified by *conv_id*.

        Used for rollback when the user presses Ctrl+C mid-migration or when
        an error occurs after a partial write.  Must be idempotent.
        """
