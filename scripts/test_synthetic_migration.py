"""
Synthetic migration test.

Creates a canonical multi-turn conversation that exercises every important
turn type, then writes it to every supported format so the user can
manually verify how each tool renders the conversation.

Run:
    python scripts/test_synthetic_migration.py

Outputs:
    CC:     ~/.claude/projects/<encoded test dir>/<session-id>.jsonl
    Codex:  ~/.codex/sessions/…/rollout-…-<session-id>.jsonl
    Cursor: <test dir>/.specstory/history/<session-id>.json

The synthetic conversation is associated with the test directory:
    C:/Users/troyh/Documents/dev/agent-migrator-synthetic-test
(created by this script if it doesn't exist)
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_migrator.models import (
    Conversation,
    ConversationInfo,
    StandardToolName,
    TextMessage,
    ToolCallMessage,
)
from agent_migrator.tools.claude_code import ClaudeCodeAdapter
from agent_migrator.tools.codex import CodexAdapter
from agent_migrator.tools.cursor import CursorAdapter

TEST_DIR = Path("C:/Users/troyh/Documents/dev/agent-migrator-synthetic-test")
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build a rich synthetic conversation
# ---------------------------------------------------------------------------

def ts(offset_s: int = 0) -> datetime:
    base = datetime(2026, 4, 13, 10, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return base + timedelta(seconds=offset_s)


PLAN_CONTENT = """\
# Synthetic Demo Plan

## Goal
Build a minimal demo script that exercises every tool type.

## Steps
1. Read the existing project structure
2. Create a new helper file
3. Edit the main entry point
4. Run tests and verify output
"""

SYNTHETIC_TURNS = [
    # ── Turn 0: user kicks off task ──────────────────────────────────────────
    TextMessage(
        role="user",
        text="Audit the project and create a plan, then implement it.",
        timestamp=ts(0),
    ),

    # ── Turn 1: assistant explains, then Bash ────────────────────────────────
    TextMessage(
        role="assistant",
        text="I'll start by surveying the project layout.",
        timestamp=ts(1),
    ),
    ToolCallMessage(
        name=StandardToolName.BASH,
        input={"command": "ls -la"},
        result="total 12\ndrwxr-xr-x  2 user user 4096 Apr 13 10:00 .\ndrwxr-xr-x 10 user user 4096 Apr 13 10:00 ..\n-rw-r--r--  1 user user  220 Apr 13 10:00 main.py",
        timestamp=ts(2),
    ),

    # ── Turn 2: parallel Bash calls ──────────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Let me check the file contents and git status simultaneously.",
        timestamp=ts(3),
    ),
    ToolCallMessage(
        name=StandardToolName.BASH,
        input={"command": "cat main.py"},
        result="# Entry point\nprint('hello world')\n",
        timestamp=ts(4),
    ),
    ToolCallMessage(
        name=StandardToolName.BASH,
        input={"command": "git status --short"},
        result="M  main.py\n?? helper.py\n",
        timestamp=ts(4),
    ),

    # ── Turn 3: Read tool ────────────────────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Reading the main file for details.",
        timestamp=ts(5),
    ),
    ToolCallMessage(
        name=StandardToolName.READ,
        input={"file_path": "C:/Users/troyh/Documents/dev/agent-migrator-synthetic-test/main.py"},
        result="# Entry point\nprint('hello world')\n",
        timestamp=ts(6),
    ),

    # ── Turn 4: Glob + Grep ──────────────────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Searching for Python files and looking for TODO markers.",
        timestamp=ts(7),
    ),
    ToolCallMessage(
        name=StandardToolName.GLOB,
        input={"pattern": "**/*.py"},
        result="main.py\nhelper.py\n",
        timestamp=ts(8),
    ),
    ToolCallMessage(
        name=StandardToolName.GREP,
        input={"pattern": "TODO", "path": "."},
        result="main.py:3:# TODO: add logging\n",
        timestamp=ts(8),
    ),

    # ── Turn 5: assistant writes plan, user approves ─────────────────────────
    TextMessage(
        role="assistant",
        text=(
            "Here is my plan:\n\n"
            "<proposed_plan>\n"
            + PLAN_CONTENT.strip()
            + "\n</proposed_plan>\n\n"
            "Ready to proceed when you are."
        ),
        timestamp=ts(9),
    ),
    TextMessage(
        role="user",
        text="Looks good, go ahead.",
        timestamp=ts(10),
    ),

    # ── Turn 6: Write tool (new file) ────────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Creating helper.py now.",
        timestamp=ts(11),
    ),
    ToolCallMessage(
        name=StandardToolName.WRITE,
        input={
            "file_path": "C:/Users/troyh/Documents/dev/agent-migrator-synthetic-test/helper.py",
            "content": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n",
        },
        result="File written successfully.",
        timestamp=ts(12),
    ),

    # ── Turn 7: Edit tool (modify existing file) ─────────────────────────────
    TextMessage(
        role="assistant",
        text="Updating main.py to use the new helper.",
        timestamp=ts(13),
    ),
    ToolCallMessage(
        name=StandardToolName.EDIT,
        input={
            "file_path": "C:/Users/troyh/Documents/dev/agent-migrator-synthetic-test/main.py",
            "old_string": "# Entry point\nprint('hello world')\n",
            "new_string": "# Entry point\nfrom helper import greet\nprint(greet('world'))\n",
        },
        result="File updated successfully.",
        timestamp=ts(14),
    ),

    # ── Turn 8: verify with Bash ─────────────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Running the updated script to verify.",
        timestamp=ts(15),
    ),
    ToolCallMessage(
        name=StandardToolName.BASH,
        input={"command": "python main.py"},
        result="Hello, world!\n",
        timestamp=ts(16),
    ),

    # ── Turn 9: assistant final summary ─────────────────────────────────────
    TextMessage(
        role="assistant",
        text="Done! The implementation is complete and tests pass.",
        timestamp=ts(17),
    ),

    # ── Turn 10: user Bash Mode (user-initiated shell command) ───────────────
    # Simulates the <user_shell_command> pattern from Codex Bash Mode.
    # In our intermediate format, these become a ToolCallMessage with role-like
    # semantics — stored as a Bash tool call without a preceding assistant turn.
    ToolCallMessage(
        name=StandardToolName.BASH,
        input={"command": "ls -la"},
        result="total 16\n-rw-r--r-- 1 user user  72 Apr 13 10:00 helper.py\n-rw-r--r-- 1 user user  80 Apr 13 10:00 main.py\n",
        timestamp=ts(18),
    ),
]

info = ConversationInfo(
    id="synthetic-001",
    name="Synthetic Migration Test",
    source_tool="test",
    updated_at=ts(18),
    created_at=ts(0),
    message_count=len(SYNTHETIC_TURNS),
    size_bytes=0,
)

conv = Conversation(
    info=info,
    turns=SYNTHETIC_TURNS,
    plan_content=PLAN_CONTENT,
    model="claude-sonnet-4-6",
)

# ---------------------------------------------------------------------------
# Write to every available format
# ---------------------------------------------------------------------------

passed = 0
failed = 0
written: dict[str, str] = {}


def run(name: str, fn):
    global passed, failed
    try:
        result = fn()
        print(f"  [OK]   {name}: {result}")
        passed += 1
        return result
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1
        return None


print(f"\n=== Writing synthetic conversation to: {TEST_DIR} ===\n")

# Claude Code
cc = ClaudeCodeAdapter()
cc_id = run("CC write", lambda: cc.write_conversation(conv, TEST_DIR))
if cc_id:
    written["CC"] = cc_id
    from agent_migrator.tools.claude_code import encode_project_path, _projects_dir
    encoded = encode_project_path(TEST_DIR.resolve())
    jsonl = _projects_dir() / encoded / f"{cc_id}.jsonl"
    print(f"         -> {jsonl}")

# Codex
codex = CodexAdapter()
if codex.is_available():
    codex_id = run("Codex write", lambda: codex.write_conversation(conv, TEST_DIR))
    if codex_id:
        written["Codex"] = codex_id
        from agent_migrator.tools.codex import _find_rollout_file
        rollout = _find_rollout_file(codex_id)
        print(f"         -> {rollout}")
else:
    print("  [SKIP] Codex (not available)")

# Cursor
cursor = CursorAdapter()
if cursor.is_available():
    try:
        cursor_id = cursor.write_conversation(conv, TEST_DIR)
        print(f"  [OK]   Cursor write: {cursor_id}")
        passed += 1
        written["Cursor"] = cursor_id
    except RuntimeError as e:
        if "No Cursor workspace found" in str(e):
            print(f"  [SKIP] Cursor write: directory not yet opened in Cursor")
            print(f"         Open {TEST_DIR} in Cursor once, then re-run this script")
        else:
            # Cursor upload may fail — try local backend
            print(f"  [INFO] Cursor write failed ({e}), trying local backend...")
            cursor_id = run("Cursor write (local)", lambda: cursor.write_conversation(conv, TEST_DIR, use_local_backend=True))
            if cursor_id:
                written["Cursor"] = cursor_id
    except Exception as e:
        print(f"  [INFO] Cursor write failed ({e}), trying local backend...")
        cursor_id = run("Cursor write (local)", lambda: cursor.write_conversation(conv, TEST_DIR, use_local_backend=True))
        if cursor_id:
            written["Cursor"] = cursor_id
else:
    print("  [SKIP] Cursor (not available)")

# ---------------------------------------------------------------------------
# Verify round-trips (read back what was written)
# ---------------------------------------------------------------------------

print(f"\n=== Round-trip verification ===\n")

if "CC" in written:
    rt = run("CC round-trip read", lambda: cc.read_conversation(written["CC"], TEST_DIR))
    if rt:
        n_text = sum(1 for t in rt.turns if isinstance(t, TextMessage))
        n_tool = sum(1 for t in rt.turns if isinstance(t, ToolCallMessage))
        tool_names = {t.name for t in rt.turns if isinstance(t, ToolCallMessage)}
        print(f"         turns={len(rt.turns)} (text={n_text}, tool={n_tool})")
        print(f"         tool names: {sorted(tool_names)}")
        print(f"         plan_content: {bool(rt.plan_content)}")

if "Codex" in written:
    rt = run("Codex round-trip read", lambda: codex.read_conversation(written["Codex"], TEST_DIR))
    if rt:
        n_text = sum(1 for t in rt.turns if isinstance(t, TextMessage))
        n_tool = sum(1 for t in rt.turns if isinstance(t, ToolCallMessage))
        tool_names = {t.name for t in rt.turns if isinstance(t, ToolCallMessage)}
        print(f"         turns={len(rt.turns)} (text={n_text}, tool={n_tool})")
        print(f"         tool names: {sorted(tool_names)}")
        print(f"         plan_content: {bool(rt.plan_content)}")

# ---------------------------------------------------------------------------
# Summary and cleanup prompt
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Results: {passed} OK, {failed} failed")
print(f"\nSessions written:")
for tool, sid in written.items():
    print(f"  {tool}: {sid}")

print(f"""
Manual verification steps:
  CC:    Open Claude Code in {TEST_DIR}
         The conversation 'Synthetic Migration Test' should appear with:
         - Tool calls (Bash, Read, Write, Edit, Glob, Grep) visible as collapsible blocks
         - A plan visible in the plan panel
  Codex: Open Codex in {TEST_DIR}
         The session should show all turns including tool calls and plan
""")

print("Run this script again with --cleanup to delete the test sessions:")
print("  python scripts/test_synthetic_migration.py --cleanup")

if "--cleanup" in sys.argv:
    print("\n=== Cleanup ===")
    if "CC" in written:
        run("CC delete", lambda: cc.delete_conversation(written["CC"], TEST_DIR))
    if "Codex" in written and codex.is_available():
        run("Codex delete", lambda: codex.delete_conversation(written["Codex"], TEST_DIR))
    print("Cleanup done.")

if failed:
    sys.exit(1)
