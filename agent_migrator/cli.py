from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_AGENT_ALIASES: dict[str, str] = {
    "claude-code": "claude_code",
    "claude_code":  "claude_code",
    "codex":        "codex",
    "cursor":       "cursor",
    "gemini":       "gemini",
}


def _humanize(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} GB"


def _load_adapters():
    from agent_migrator.agents.claude_code import ClaudeCodeAdapter
    from agent_migrator.agents.codex import CodexAdapter
    from agent_migrator.agents.cursor import CursorAdapter
    from agent_migrator.agents.gemini import GeminiAdapter

    return [CursorAdapter(), ClaudeCodeAdapter(), CodexAdapter(), GeminiAdapter()]


def _find_adapter(agent_alias: str, adapters):
    """Resolve an agent alias to an adapter. Exits 1 with JSON error if not found."""
    agent_id = _AGENT_ALIASES.get(agent_alias)
    if agent_id is None:
        known = ", ".join(sorted(set(_AGENT_ALIASES.keys())))
        _json_error(f"Unknown agent '{agent_alias}'. Known agents: {known}")
    adapter = next((a for a in adapters if a.agent_id == agent_id), None)
    if adapter is None or not adapter.is_available():
        _json_error(f"Agent '{agent_alias}' is not installed or not available on this system.")
    return adapter


def _json_error(msg: str) -> None:
    """Print a JSON error object to stderr and exit 1."""
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(1)


def _conv_info_dict(c) -> dict:
    return {
        "id":            c.id,
        "name":          c.name,
        "updated_at":    c.updated_at.isoformat(),
        "created_at":    c.created_at.isoformat(),
        "message_count": c.message_count,
        "size_bytes":    c.size_bytes,
    }


def _run_list(args, adapters) -> None:
    project_path = (args.project_path or Path.cwd()).resolve()
    if not project_path.exists():
        _json_error(f"Path does not exist: {project_path}")

    source = _find_adapter(args.source_agent, adapters)
    convs = source.list_conversations(project_path)
    print(json.dumps([_conv_info_dict(c) for c in convs]))


def _run_move(args, adapters) -> None:
    from agent_migrator.agents.base import AgentNetworkError

    project_path = (args.project_path or Path.cwd()).resolve()
    if not project_path.exists():
        _json_error(f"Path does not exist: {project_path}")

    source = _find_adapter(args.source_agent, adapters)
    dest = _find_adapter(args.dest_agent, adapters)

    all_convs = source.list_conversations(project_path)

    if args.conv_id:
        matching = [c for c in all_convs if c.id == args.conv_id]
        if not matching:
            _json_error(f"Conversation '{args.conv_id}' not found in {source.name} for this project.")
        conv_infos = matching
    else:
        conv_infos = all_convs

    if not conv_infos:
        _json_error(f"No conversations found in {source.name} for this project.")

    results = []
    for conv_info in conv_infos:
        conv = source.read_conversation(conv_info.id, project_path)
        try:
            new_id = dest.write_conversation(conv, project_path)
            results.append({"source_id": conv_info.id, "destination_id": new_id, "name": conv_info.name})
        except AgentNetworkError as e:
            if args.allow_cursor_fallback:
                new_id = dest.write_conversation(conv, project_path, use_local_backend=True)
                results.append({
                    "source_id": conv_info.id, "destination_id": new_id,
                    "name": conv_info.name, "used_fallback": True,
                })
            else:
                _json_error(str(e))

    print(json.dumps(results))


def _run_interactive(args, adapters) -> None:
    project_path = Path(args.project_path).resolve() if args.project_path else Path.cwd().resolve()

    if not project_path.exists():
        console.print(f"[red]Error:[/red] path does not exist: {project_path}")
        sys.exit(1)

    available = [a for a in adapters if a.is_available()]

    if len(available) < 2:
        names = ", ".join(a.name for a in available) if available else "none"
        console.print(
            f"[red]Error:[/red] at least 2 supported agents must be installed.\n"
            f"Detected: {names}"
        )
        sys.exit(1)

    console.print(f"\n[bold]agent-migrator[/bold]  —  project: [cyan]{project_path}[/cyan]\n")

    # Pick source
    src_name = questionary.select(
        "Source agent:",
        choices=[a.name for a in available],
    ).ask()
    if src_name is None:
        sys.exit(0)
    src = next(a for a in available if a.name == src_name)

    # Pick destination (exclude source)
    dst_choices = [a.name for a in available if a.agent_id != src.agent_id]
    if len(dst_choices) == 1:
        dst = next(a for a in available if a.agent_id != src.agent_id)
        console.print(f"Destination agent: {dst.name}")
    else:
        dst_name = questionary.select(
            "Destination agent:",
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

    use_local_fallback = False
    use_local_fallback_decided = False

    try:
        for idx, conv_info in enumerate(selected_convs, 1):
            label = f"  [{idx}/{len(selected_convs)}] {conv_info.name[:50]}"
            console.print(f"{label} ...", end="")
            try:
                if use_local_fallback:
                    conv = src.read_conversation(conv_info.id, project_path)
                    new_id = dst.write_conversation(conv, project_path, use_local_backend=True)
                else:
                    new_id = engine.migrate_one(src, dst, conv_info, project_path, lambda _: None)
                results.succeeded.append((conv_info, new_id))
                console.print(" [green]done[/green]")
            except Exception as e:
                from agent_migrator.agents.base import AgentNetworkError
                if isinstance(e, AgentNetworkError) and not use_local_fallback_decided:
                    use_local_fallback_decided = True
                    console.print(f" [yellow]server upload failed[/yellow]")
                    console.print(
                        f"\n  [yellow]Could not upload conversation history to the destination's server:[/yellow]"
                        f"\n  {e}"
                        f"\n\n  The fallback uses the destination's local backend, which provides full"
                        f"\n  conversation context but may restrict model selection for migrated conversations.\n"
                    )
                    fallback = questionary.confirm(
                        "Use local fallback backend for this and remaining conversations?",
                        default=True,
                    ).ask()
                    if fallback:
                        use_local_fallback = True
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
                elif isinstance(e, AgentNetworkError) and use_local_fallback:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-migrator",
        description="Migrate conversation history between AI coding agents.",
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help="Project directory for interactive mode (default: cwd).",
    )

    sub = parser.add_subparsers(dest="command")

    # list subcommand
    lp = sub.add_parser("list", help="List conversations for a project (outputs JSON).")
    lp.add_argument("--from", dest="source_agent", required=True, metavar="AGENT",
                    help="Source agent (claude-code, codex, cursor, gemini).")
    lp.add_argument("--dir", dest="project_path", type=Path, default=None, metavar="PATH",
                    help="Project directory (default: cwd).")

    # move subcommand
    mp = sub.add_parser("move", help="Migrate conversations (outputs JSON).")
    mp.add_argument("--from", dest="source_agent", required=True, metavar="AGENT",
                    help="Source agent.")
    mp.add_argument("--to", dest="dest_agent", required=True, metavar="AGENT",
                    help="Destination agent.")
    mp.add_argument("--id", dest="conv_id", default=None, metavar="ID",
                    help="Source conversation ID. Omit to migrate all conversations.")
    mp.add_argument("--dir", dest="project_path", type=Path, default=None, metavar="PATH",
                    help="Project directory (default: cwd).")
    mp.add_argument("--allow-cursor-fallback", action="store_true",
                    help="Fall back to local backend if Cursor API upload fails.")

    args = parser.parse_args()
    adapters = _load_adapters()

    if args.command == "list":
        _run_list(args, adapters)
    elif args.command == "move":
        _run_move(args, adapters)
    else:
        _run_interactive(args, adapters)
