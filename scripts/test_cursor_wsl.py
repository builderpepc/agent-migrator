"""
Test script for CursorAdapter WSL Remote workspace detection.

Covers the path/URI canonicalisation helpers and the end-to-end
_find_workspace_dir() lookup against a synthetic workspaceStorage tree.
No real Cursor data is read or written.

Run from repo root:  python scripts/test_cursor_wsl.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_migrator.agents import cursor as C

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


# ── 1. WSL folder URI -> key ──────────────────────────────────────────────────
print("\n=== 1. _wsl_uri_to_key ===")

def t_uri_plain():
    k = C._wsl_uri_to_key("vscode-remote://wsl+Ubuntu/home/user/project")
    assert k == "wsl:ubuntu:/home/user/project", k

def t_uri_encoded_plus():
    k = C._wsl_uri_to_key("vscode-remote://wsl%2BUbuntu/home/user/project")
    assert k == "wsl:ubuntu:/home/user/project", k

def t_uri_versioned_distro():
    k = C._wsl_uri_to_key("vscode-remote://wsl+Ubuntu-22.04/home/me/app")
    assert k == "wsl:ubuntu-22.04:/home/me/app", k

def t_uri_trailing_slash():
    k = C._wsl_uri_to_key("vscode-remote://wsl+Ubuntu/home/user/project/")
    assert k == "wsl:ubuntu:/home/user/project", k

def t_uri_encoded_space():
    k = C._wsl_uri_to_key("vscode-remote://wsl+Ubuntu/home/user/my%20project")
    assert k == "wsl:ubuntu:/home/user/my project", k

def t_uri_non_wsl_remote_returns_none():
    # ssh-remote and containers are not WSL — must not match.
    assert C._wsl_uri_to_key("vscode-remote://ssh-remote+box/home/user") is None
    assert C._wsl_uri_to_key("file:///c%3A/Users/me/proj") is None

test("plain wsl+Ubuntu URI", t_uri_plain)
test("percent-encoded wsl%2BUbuntu URI", t_uri_encoded_plus)
test("versioned distro name", t_uri_versioned_distro)
test("trailing slash trimmed", t_uri_trailing_slash)
test("percent-encoded space in path", t_uri_encoded_space)
test("non-WSL remote / file URI -> None", t_uri_non_wsl_remote_returns_none)


# ── 2. WSL UNC path -> key ────────────────────────────────────────────────────
print("\n=== 2. _wsl_path_to_key ===")

def t_path_localhost():
    k = C._wsl_path_to_key(Path(r"\\wsl.localhost\Ubuntu\home\user\project"))
    assert k == "wsl:ubuntu:/home/user/project", k

def t_path_dollar():
    k = C._wsl_path_to_key(Path(r"\\wsl$\Ubuntu\home\user\project"))
    assert k == "wsl:ubuntu:/home/user/project", k

def t_path_extended_unc():
    # Path.resolve() may yield the \\?\UNC\ extended-length form.
    k = C._wsl_path_to_key(Path(r"\\?\UNC\wsl.localhost\Ubuntu\home\user\project"))
    assert k == "wsl:ubuntu:/home/user/project", k

def t_path_non_wsl_returns_none():
    assert C._wsl_path_to_key(Path(r"C:\Users\me\proj")) is None

test("\\\\wsl.localhost\\... UNC path", t_path_localhost)
test("\\\\wsl$\\... UNC path", t_path_dollar)
test("\\\\?\\UNC\\... extended-length path", t_path_extended_unc)
test("non-WSL local path -> None", t_path_non_wsl_returns_none)


# ── 3. UNC path and remote URI agree ──────────────────────────────────────────
print("\n=== 3. UNC path <-> remote URI agreement ===")

def t_round_trip():
    uri_key = C._wsl_uri_to_key("vscode-remote://wsl+Ubuntu/home/user/project")
    # Both common UNC spellings (and casing) must produce the same key.
    for p in (
        r"\\wsl.localhost\Ubuntu\home\user\project",
        r"\\wsl$\Ubuntu\home\user\project",
        r"\\WSL.LOCALHOST\ubuntu\home\user\project",
    ):
        assert C._wsl_path_to_key(Path(p)) == uri_key, p

test("UNC variants match the remote URI key", t_round_trip)


# ── 4. _find_workspace_dir end-to-end (synthetic storage) ─────────────────────
print("\n=== 4. _find_workspace_dir (synthetic workspaceStorage) ===")

def t_find_wsl_workspace():
    with tempfile.TemporaryDirectory() as td:
        ws_root = Path(td) / "workspaceStorage"
        # A decoy local workspace + the real WSL one.
        decoy = ws_root / "aaaa"
        decoy.mkdir(parents=True)
        (decoy / "workspace.json").write_text(
            json.dumps({"folder": "file:///c%3A/Users/me/other"}), encoding="utf-8"
        )
        target = ws_root / "bbbb"
        target.mkdir(parents=True)
        (target / "workspace.json").write_text(
            json.dumps({"folder": "vscode-remote://wsl+Ubuntu/home/user/project"}),
            encoding="utf-8",
        )

        orig = C._workspace_storage_dir
        C._workspace_storage_dir = lambda: ws_root
        try:
            found = C._find_workspace_dir(Path(r"\\wsl.localhost\Ubuntu\home\user\project"))
        finally:
            C._workspace_storage_dir = orig
        assert found == target, found

def t_find_no_match_returns_none():
    with tempfile.TemporaryDirectory() as td:
        ws_root = Path(td) / "workspaceStorage"
        e = ws_root / "aaaa"
        e.mkdir(parents=True)
        (e / "workspace.json").write_text(
            json.dumps({"folder": "vscode-remote://wsl+Ubuntu/home/user/project"}),
            encoding="utf-8",
        )
        orig = C._workspace_storage_dir
        C._workspace_storage_dir = lambda: ws_root
        try:
            found = C._find_workspace_dir(Path(r"\\wsl.localhost\Ubuntu\home\user\different"))
        finally:
            C._workspace_storage_dir = orig
        assert found is None, found

test("finds WSL workspace by UNC target path", t_find_wsl_workspace)
test("no false match for different WSL path", t_find_no_match_returns_none)


print(f"\n{'='*50}\nResult: {passed} passed, {failed} failed\n{'='*50}")
sys.exit(1 if failed else 0)
