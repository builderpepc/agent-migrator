"""
Test script for CodexAdapter — calls adapter methods directly (no interactive CLI).

The native Codex session (019d84ae-...) is READ ONLY in these tests.
Write-path tests create new files which are cleaned up after each test.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow running from repo root: python scripts/test_codex_adapter.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_migrator.models import StandardToolName, TextMessage, ToolCallMessage
from agent_migrator.tools.claude_code import ClaudeCodeAdapter
from agent_migrator.tools.codex import CodexAdapter

CODEX_DEMO_PROJECT = Path("C:/Users/troyh/Documents/dev/codex-demo")
NATIVE_SESSION_ID = "019d84ae-774f-7e02-8e71-f46a1509334a"

passed = 0
failed = 0


def test(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1


# ── 1. Detection ─────────────────────────────────────────────────────────────
print("\n=== 1. Detection ===")

def t_detection():
    assert CodexAdapter().is_available(), "CodexAdapter.is_available() returned False"

test("is_available", t_detection)


# ── 2. List conversations ─────────────────────────────────────────────────────
print("\n=== 2. List conversations ===")

adapter = CodexAdapter()
convs = adapter.list_conversations(CODEX_DEMO_PROJECT)
print(f"  Found {len(convs)} conversation(s) for {CODEX_DEMO_PROJECT}")
for c in convs:
    print(f"    id={c.id}  name={c.name!r}  msgs={c.message_count}  updated={c.updated_at.date()}")

def t_list_finds_native():
    assert any(NATIVE_SESSION_ID in c.id for c in convs), (
        f"Native session {NATIVE_SESSION_ID} not found in list"
    )

def t_list_has_name():
    native = next(c for c in convs if NATIVE_SESSION_ID in c.id)
    assert native.name and native.name != NATIVE_SESSION_ID, (
        f"Name not extracted (got {native.name!r})"
    )

def t_list_message_count():
    native = next(c for c in convs if NATIVE_SESSION_ID in c.id)
    assert native.message_count > 0, "message_count is 0"

test("finds native session", t_list_finds_native)
test("extracts display name", t_list_has_name)
test("non-zero message count", t_list_message_count)


# ── 3. Read conversation ──────────────────────────────────────────────────────
print("\n=== 3. Read conversation ===")

conv = adapter.read_conversation(NATIVE_SESSION_ID, CODEX_DEMO_PROJECT)
print(f"  Turns: {len(conv.turns)}")
print(f"  Text turns: {sum(1 for t in conv.turns if isinstance(t, TextMessage))}")
print(f"  Tool turns: {sum(1 for t in conv.turns if isinstance(t, ToolCallMessage))}")
print(f"  Plan content: {bool(conv.plan_content)} ({len(conv.plan_content or '')} chars)")
print(f"  Model: {conv.model!r}")

tool_names = {t.name for t in conv.turns if isinstance(t, ToolCallMessage)}
print(f"  Tool names in session: {tool_names}")

def t_read_has_user_turns():
    assert any(isinstance(t, TextMessage) and t.role == "user" for t in conv.turns)

def t_read_has_assistant_turns():
    assert any(isinstance(t, TextMessage) and t.role == "assistant" for t in conv.turns)

def t_read_plan_content():
    assert conv.plan_content is not None, "plan_content is None"
    assert "FieldOps" in conv.plan_content, f"Expected 'FieldOps' in plan, got: {conv.plan_content[:100]}"

def t_read_bash_tool_calls():
    assert any(
        isinstance(t, ToolCallMessage) and t.name == StandardToolName.BASH
        for t in conv.turns
    ), "No Bash tool calls found"

def t_read_no_system_messages():
    for turn in conv.turns:
        if isinstance(turn, TextMessage):
            assert not turn.text.strip().startswith("<environment_context"), (
                f"System-injected content leaked into turns: {turn.text[:80]!r}"
            )

def t_read_tool_results_populated():
    # Tool calls should have non-empty results matched from output records
    tool_calls_with_results = [
        t for t in conv.turns
        if isinstance(t, ToolCallMessage) and t.result
    ]
    assert len(tool_calls_with_results) > 0, "No tool call results were populated"

test("has user turns", t_read_has_user_turns)
test("has assistant turns", t_read_has_assistant_turns)
test("plan_content extracted", t_read_plan_content)
test("Bash tool calls", t_read_bash_tool_calls)
test("no system content in turns", t_read_no_system_messages)
test("tool results populated", t_read_tool_results_populated)


# ── 4. Codex -> Claude Code ───────────────────────────────────────────────────
print("\n=== 4. Codex -> Claude Code ===")

cc_adapter = ClaudeCodeAdapter()
cc_new_id = None

def t_codex_to_cc_write():
    global cc_new_id
    cc_new_id = cc_adapter.write_conversation(conv, CODEX_DEMO_PROJECT)
    assert cc_new_id, "write_conversation returned empty ID"

def t_codex_to_cc_file_exists():
    assert cc_new_id, "write skipped"
    from agent_migrator.tools.claude_code import encode_project_path, _projects_dir
    encoded = encode_project_path(CODEX_DEMO_PROJECT.resolve())
    jsonl = _projects_dir() / encoded / f"{cc_new_id}.jsonl"
    assert jsonl.exists(), f"Expected JSONL at {jsonl}"

def t_codex_to_cc_roundtrip():
    assert cc_new_id, "write skipped"
    rt_conv = cc_adapter.read_conversation(cc_new_id, CODEX_DEMO_PROJECT)
    assert len(rt_conv.turns) > 0, "Round-trip conversation has no turns"
    assert rt_conv.plan_content is not None, "Round-trip plan_content is None"
    assert "FieldOps" in rt_conv.plan_content

test("write_conversation", t_codex_to_cc_write)
test("JSONL file created", t_codex_to_cc_file_exists)
test("round-trip preserves turns and plan", t_codex_to_cc_roundtrip)

# Cleanup CC file
if cc_new_id:
    try:
        cc_adapter.delete_conversation(cc_new_id, CODEX_DEMO_PROJECT)
        print(f"  (cleaned up CC session {cc_new_id})")
    except Exception as e:
        print(f"  (cleanup failed: {e})")


# ── 5. Claude Code -> Codex ───────────────────────────────────────────────────
print("\n=== 5. Claude Code -> Codex ===")

# Find a native CC session from any project (prefer non-today sessions)
from agent_migrator.tools.claude_code import _projects_dir
import os

cc_source_conv = None
cc_source_project = None

projects_dir = _projects_dir()
if projects_dir.exists():
    for project_dir in sorted(projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not project_dir.is_dir():
            continue
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for jf in jsonl_files:
            # Skip files modified today
            mtime = jf.stat().st_mtime
            from datetime import date
            if date.fromtimestamp(mtime) == date.today():
                continue
            try:
                test_conv = cc_adapter.read_conversation(jf.stem, Path(project_dir.name))
                if len(test_conv.turns) > 2:
                    cc_source_conv = test_conv
                    cc_source_project = Path(project_dir.name)
                    print(f"  Using CC session: {jf.stem[:20]}... from {project_dir.name}")
                    break
            except Exception:
                continue
        if cc_source_conv:
            break

codex_new_id = None

def t_cc_to_codex_write():
    global codex_new_id
    if cc_source_conv is None:
        print("  (skipped: no suitable CC source session found)")
        return
    codex_new_id = adapter.write_conversation(cc_source_conv, CODEX_DEMO_PROJECT)
    assert codex_new_id, "write_conversation returned empty ID"

def t_cc_to_codex_file_exists():
    if codex_new_id is None:
        print("  (skipped)")
        return
    rollout_file = None
    from agent_migrator.tools.codex import _find_rollout_file
    rollout_file = _find_rollout_file(codex_new_id)
    assert rollout_file and rollout_file.exists(), f"Rollout file not found for {codex_new_id}"
    print(f"  Rollout file: {rollout_file.name}")

def t_cc_to_codex_session_meta():
    if codex_new_id is None:
        print("  (skipped)")
        return
    from agent_migrator.tools.codex import _find_rollout_file, _read_session_meta
    rollout_file = _find_rollout_file(codex_new_id)
    meta = _read_session_meta(rollout_file)
    assert meta, "session_meta not written"
    assert meta.get("id") == codex_new_id, f"session_meta.id mismatch: {meta.get('id')}"
    assert meta.get("originator") == "agent-migrator", "originator not set"

def t_cc_to_codex_roundtrip():
    if codex_new_id is None:
        print("  (skipped)")
        return
    rt_conv = adapter.read_conversation(codex_new_id, CODEX_DEMO_PROJECT)
    assert len(rt_conv.turns) > 0, "Round-trip has no turns"
    print(f"  Round-trip turns: {len(rt_conv.turns)}")

test("write CC -> Codex", t_cc_to_codex_write)
test("rollout file created", t_cc_to_codex_file_exists)
test("session_meta correct", t_cc_to_codex_session_meta)
test("round-trip readable", t_cc_to_codex_roundtrip)

# Cleanup
if codex_new_id:
    try:
        adapter.delete_conversation(codex_new_id, CODEX_DEMO_PROJECT)
        print(f"  (cleaned up Codex session {codex_new_id})")
    except Exception as e:
        print(f"  (cleanup failed: {e})")


# ── 6. Atomic write (failure cleanup) ────────────────────────────────────────
print("\n=== 6. Atomic write / cleanup ===")

def t_no_tmp_on_failure():
    import tempfile, shutil
    from agent_migrator.tools.codex import _sessions_dir
    # Use a read-only temp dir to force failure after tmp creation
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Make the target dir read-only so replace() fails
        dummy_project = tmp_dir / "dummy"
        dummy_project.mkdir()
        new_id = None
        try:
            new_id = adapter.write_conversation(conv, dummy_project)
        except Exception:
            pass  # expected to fail
        # Check no .tmp files remain in sessions dir
        tmp_files = list(_sessions_dir().rglob("*.tmp"))
        assert not tmp_files, f"Leftover .tmp files: {tmp_files}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if new_id:
            adapter.delete_conversation(new_id, dummy_project)

test("no leftover .tmp on failure", t_no_tmp_on_failure)


# ── 7. Delete (idempotency) ───────────────────────────────────────────────────
print("\n=== 7. Delete ===")

def t_delete_idempotent():
    # Write a throwaway session then delete it twice
    temp_id = adapter.write_conversation(conv, CODEX_DEMO_PROJECT)
    adapter.delete_conversation(temp_id, CODEX_DEMO_PROJECT)
    adapter.delete_conversation(temp_id, CODEX_DEMO_PROJECT)  # second delete: should not raise

test("delete is idempotent", t_delete_idempotent)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
