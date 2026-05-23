"""
StopFailure hook for the cc-failover Claude Code plugin.

Invoked by Claude Code when a session ends due to an API-level failure.
Reads hook input from stdin (JSON), migrates the session to a configured
fallback agent, and sends a desktop notification with the result.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from notifypy import Notify


def _notify(title: str, message: str) -> None:
    n = Notify()
    n.title = title
    n.message = message
    n.send(block=False)


def main() -> None:
    hook_input = json.loads(sys.stdin.read())
    session_id = hook_input.get("session_id")
    cwd = hook_input.get("cwd", ".")
    error = hook_input.get("error", "unknown")

    # Resolve config. Claude Code is expected to expose userConfig values as
    # CLAUDE_PLUGIN_CONFIG_<KEY> environment variables when running hooks.
    # Falls back to a config.json in CLAUDE_PLUGIN_DATA if the env vars are absent
    # (e.g. when the plugin is loaded via --plugin-dir without a persistent install).
    destination = os.environ.get("CLAUDE_PLUGIN_CONFIG_DESTINATION")
    trigger_raw = os.environ.get("CLAUDE_PLUGIN_CONFIG_TRIGGER_ERRORS")
    trigger_errors: set[str] | None = set(trigger_raw.split(",")) if trigger_raw else None

    if not destination or trigger_errors is None:
        data_dir = Path(os.environ.get("CLAUDE_PLUGIN_DATA", ""))
        config_path = data_dir / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            destination = destination or cfg.get("destination")
            trigger_errors = trigger_errors or set(
                cfg.get("trigger_errors", ["rate_limit", "billing_error", "server_error"])
            )

    if not destination:
        _notify(
            "Claude Code Failover",
            "Plugin not configured — run: claude plugin configure cc-failover",
        )
        return

    trigger_errors = trigger_errors or {"rate_limit", "billing_error", "server_error"}

    if error not in trigger_errors:
        return

    if not session_id:
        _notify(
            "Claude Code Failover",
            f"StopFailure ({error}) — no session ID in hook payload, cannot migrate.",
        )
        return

    _notify("Claude Code Failover", f"StopFailure ({error}) — migrating session to {destination}…")

    try:
        result = subprocess.run(
            [
                "agent-migrator", "move",
                "--from", "claude-code",
                "--to", destination,
                "--id", session_id,
                "--dir", cwd,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        migration = json.loads(result.stdout)[0]
        dest_id = migration["destination_id"]
        _notify(
            "Claude Code Failover",
            f"Migrated to {destination}.\nSession ID: {dest_id[:8]}…",
        )
    except subprocess.CalledProcessError as exc:
        try:
            err_msg = json.loads(exc.stderr).get("error", exc.stderr)
        except (json.JSONDecodeError, AttributeError):
            err_msg = exc.stderr or str(exc)
        _notify("Claude Code Failover", f"Migration failed: {err_msg}")
    except FileNotFoundError:
        _notify(
            "Claude Code Failover",
            "agent-migrator not found — install with: uv tool install agent-migrator",
        )


if __name__ == "__main__":
    main()
