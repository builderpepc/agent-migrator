# cc-failover

A Claude Code plugin that automatically migrates your current session to a fallback coding agent when Claude hits a `StopFailure` event (rate limit, billing error, server outage, etc.).

Because Claude cannot respond during a StopFailure, the plugin uses desktop notifications ([notify-py](https://pypi.org/project/notify_py/)) to tell you what happened and where your session was migrated.

## Prerequisites

- **agent-migrator** — `uv tool install agent-migrator` (includes notify-py automatically)
- **Supported platforms** — Windows 10/11, macOS 10.10+, Linux (libnotify)

## Installation

**From the GitHub repo (recommended):**

```
claude plugin marketplace add builderpepc/agent-migrator --sparse .claude-plugin cc-failover-plugin
claude plugin install cc-failover
```

**From a local clone:**

```
claude plugin marketplace add ./agent-migrator
claude plugin install cc-failover
```

## Enable

```
claude plugin enable cc-failover
```

You will be prompted for:

| Field | Description |
|---|---|
| **Destination tool** | Agent to migrate to: `codex`, `gemini`, or `cursor` |
| **Error types** | Which error codes should trigger migration (see below) |

## Supported error types

| Code | Meaning |
|---|---|
| `rate_limit` | Hit token/request rate limit |
| `billing_error` | Account billing issue |
| `server_error` | Anthropic API outage or 5xx |
| `max_output_tokens` | Response exceeded output token limit |
| `authentication_failed` | API key or auth failure |

Default triggers: `rate_limit`, `billing_error`, `server_error`.

## What happens on failure

1. Claude Code fires a `StopFailure` hook with the session ID and error code.
2. The plugin checks whether the error type is configured to trigger migration.
3. If so, it runs `agent-migrator move --from claude-code --to <destination> --id <session-id>`.
4. A desktop notification reports success (with the destination session ID) or the reason for failure.

Note: Claude Code's `StopFailure` hook is notification-only — the plugin cannot resume the session automatically. Open the destination tool and resume the migrated session from there.
