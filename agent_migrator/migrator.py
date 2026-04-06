from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent_migrator.models import ConversationInfo, MigrationResult
from agent_migrator.tools.base import ToolAdapter


class MigrationEngine:
    def migrate_one(
        self,
        source: ToolAdapter,
        dest: ToolAdapter,
        conv_info: ConversationInfo,
        project_path: Path,
        on_progress: Callable[[str], None],
    ) -> str:
        """
        Migrate a single conversation from *source* to *dest*.

        Returns the new conversation ID assigned by the destination tool.
        Raises on any failure; the destination adapter's write_conversation
        is responsible for leaving no partial state on error.
        """
        on_progress(f"Reading: {conv_info.name}")
        conv = source.read_conversation(conv_info.id, project_path)

        on_progress(f"Writing: {conv_info.name}")
        new_id = dest.write_conversation(conv, project_path)

        return new_id

    def migrate_many(
        self,
        source: ToolAdapter,
        dest: ToolAdapter,
        conversations: list[ConversationInfo],
        project_path: Path,
        on_progress: Callable[[str], None],
        on_complete: Callable[[ConversationInfo, str], None],
        on_error: Callable[[ConversationInfo, Exception], None],
        is_cancelled: Callable[[], bool],
    ) -> MigrationResult:
        """
        Migrate all *conversations* one by one.

        Callbacks are invoked from whichever thread this runs in (the caller is
        responsible for thread-safety when updating UI state).

        *is_cancelled* is polled before each conversation; returning True aborts
        the loop and marks the result as cancelled.
        """
        result = MigrationResult()

        for conv_info in conversations:
            if is_cancelled():
                result.cancelled = True
                break

            try:
                new_id = self.migrate_one(
                    source, dest, conv_info, project_path, on_progress
                )
                result.succeeded.append((conv_info, new_id))
                on_complete(conv_info, new_id)
            except Exception as exc:
                result.failed.append((conv_info, str(exc)))
                on_error(conv_info, exc)

        return result
