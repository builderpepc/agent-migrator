from __future__ import annotations

import argparse
import sys
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _humanize(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} GB"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-migrator",
        description="Migrate conversation history between AI coding tools.",
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help="Path to the project directory (default: current directory).",
    )
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve() if args.project_path else Path.cwd().resolve()

    if not project_path.exists():
        console.print(f"[red]Error:[/red] path does not exist: {project_path}")
        sys.exit(1)

    from agent_migrator.tools.claude_code import ClaudeCodeAdapter
    from agent_migrator.tools.cursor import CursorAdapter

    all_adapters = [CursorAdapter(), ClaudeCodeAdapter()]
    available = [a for a in all_adapters if a.is_available()]

    if len(available) < 2:
        names = ", ".join(a.name for a in available) if available else "none"
        console.print(
            f"[red]Error:[/red] at least 2 supported tools must be installed.\n"
            f"Detected: {names}"
        )
        sys.exit(1)

    console.print(f"\n[bold]agent-migrator[/bold]  —  project: [cyan]{project_path}[/cyan]\n")

    # Pick source
    src_name = questionary.select(
        "Source tool:",
        choices=[a.name for a in available],
    ).ask()
    if src_name is None:
        sys.exit(0)
    src = next(a for a in available if a.name == src_name)

    # Pick destination (exclude source)
    dst_choices = [a.name for a in available if a.tool_id != src.tool_id]
    if len(dst_choices) == 1:
        dst = next(a for a in available if a.tool_id != src.tool_id)
        console.print(f"Destination tool: {dst.name}")
    else:
        dst_name = questionary.select(
            "Destination tool:",
            choices=dst_choices,
        ).ask()
        if dst_name is None:
            sys.exit(0)
        dst = next(a for a in available if a.name == dst_name)

    console.print(f"\nLoading conversations from [bold]{src.name}[/bold]...")
    convs = src.list_conversations(project_path)

    if not convs:
        console.print(
            f"\n[yellow]No conversations found for this project in {src.name}.[/yellow]\n"
            "For Cursor, open the exact subdirectory that was opened as a workspace in Cursor."
        )
        sys.exit(0)

    # Build labelled choices for checkbox prompt
    def _label(c) -> str:
        date = c.updated_at.strftime("%Y-%m-%d")
        size = _humanize(c.size_bytes)
        return f"{c.name[:45]:<45}  {date}  {size:>7}  ({c.message_count} turns)"

    choices = [questionary.Choice(title=_label(c), value=c) for c in convs]

    console.print()
    selected_convs = questionary.checkbox(
        "Select conversations to migrate (Space to toggle, Enter to confirm):",
        choices=choices,
    ).ask()

    if not selected_convs:
        console.print("[yellow]Nothing selected. Exiting.[/yellow]")
        sys.exit(0)

    console.print(
        f"\nMigrating [bold]{len(selected_convs)}[/bold] conversation(s) "
        f"from [bold]{src.name}[/bold] → [bold]{dst.name}[/bold]...\n"
    )

    from agent_migrator.migrator import MigrationEngine
    from agent_migrator.models import MigrationResult

    engine = MigrationEngine()
    results = MigrationResult()

    # Track whether the user has already chosen to use the local fallback backend
    # for all remaining conversations (so we don't ask repeatedly).
    use_local_fallback = False
    use_local_fallback_decided = False

    try:
        for idx, conv_info in enumerate(selected_convs, 1):
            label = f"  [{idx}/{len(selected_convs)}] {conv_info.name[:50]}"
            console.print(f"{label} ...", end="")
            try:
                if use_local_fallback:
                    # User already chose fallback for remaining conversations.
                    conv = src.read_conversation(conv_info.id, project_path)
                    new_id = dst.write_conversation(conv, project_path, use_local_backend=True)
                else:
                    new_id = engine.migrate_one(src, dst, conv_info, project_path, lambda _: None)
                results.succeeded.append((conv_info, new_id))
                console.print(" [green]done[/green]")
            except Exception as e:
                # Check if this is a server upload failure that can fall back.
                from agent_migrator.tools.cursor import ServerUploadError
                if isinstance(e, ServerUploadError) and not use_local_fallback_decided:
                    use_local_fallback_decided = True
                    console.print(f" [yellow]server upload failed[/yellow]")
                    console.print(
                        f"\n  [yellow]Could not upload conversation history to Cursor's server:[/yellow]"
                        f"\n  {e}"
                        f"\n\n  The fallback uses Cursor's local agent backend, which provides full"
                        f"\n  conversation context but restricts model selection to Anthropic models"
                        f"\n  (Sonnet/Opus) for migrated conversations.\n"
                    )
                    fallback = questionary.confirm(
                        "Use local fallback backend for this and remaining conversations?",
                        default=True,
                    ).ask()
                    if fallback:
                        use_local_fallback = True
                        # Retry this conversation with the fallback.
                        try:
                            conv = src.read_conversation(conv_info.id, project_path)
                            new_id = dst.write_conversation(conv, project_path, use_local_backend=True)
                            results.succeeded.append((conv_info, new_id))
                            console.print(f"  {label} ... [green]done (local backend)[/green]")
                            continue
                        except Exception as e2:
                            results.failed.append((conv_info, str(e2)))
                            console.print(f"  {label} ... [red]failed: {e2}[/red]")
                            continue
                    else:
                        results.failed.append((conv_info, str(e)))
                elif isinstance(e, ServerUploadError) and use_local_fallback:
                    # Already decided to use fallback — retry silently.
                    try:
                        conv = src.read_conversation(conv_info.id, project_path)
                        new_id = dst.write_conversation(conv, project_path, use_local_backend=True)
                        results.succeeded.append((conv_info, new_id))
                        console.print(" [green]done (local backend)[/green]")
                        continue
                    except Exception as e2:
                        results.failed.append((conv_info, str(e2)))
                        console.print(f" [red]failed: {e2}[/red]")
                else:
                    results.failed.append((conv_info, str(e)))
                    console.print(f" [red]failed: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")

    # Summary
    console.print()
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("status", width=2)
    summary_table.add_column("name")
    for conv, _ in results.succeeded:
        summary_table.add_row("[green]✓[/green]", conv.name)
    for conv, err in results.failed:
        summary_table.add_row("[red]✗[/red]", f"{conv.name}  [dim]({err})[/dim]")

    border = "green" if not results.failed else "yellow"
    title = f"{len(results.succeeded)} succeeded, {len(results.failed)} failed"
    console.print(Panel(summary_table, title=title, border_style=border))
