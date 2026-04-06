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

    try:
        for idx, conv_info in enumerate(selected_convs, 1):
            label = f"  [{idx}/{len(selected_convs)}] {conv_info.name[:50]}"
            console.print(f"{label} ...", end="")
            try:
                new_id = engine.migrate_one(src, dst, conv_info, project_path, lambda _: None)
                results.succeeded.append((conv_info, new_id))
                console.print(" [green]done[/green]")
            except Exception as e:
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
