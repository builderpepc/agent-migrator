import json
import os
import sys
from pathlib import Path
from agent_migrator.tools.claude_code import ClaudeCodeAdapter
from agent_migrator.tools.gemini_cli import GeminiCliAdapter

def compare_tool_calls(native_tool, migrated_tool, tool_type):
    print(f"\n--- Comparing {tool_type} Tool Call ---")
    keys_to_check = ["name", "displayName", "description", "status", "kind", "renderOutputAsMarkdown"]
    
    for key in keys_to_check:
        native_val = native_tool.get(key)
        migrated_val = migrated_tool.get(key)
        status = "MATCH" if native_val == migrated_val else "MISMATCH"
        print(f"[{status}] {key}: Native='{native_val}', Migrated='{migrated_val}'")

    # Deep check for resultDisplay (where diffs/ansi live)
    native_res = native_tool.get("resultDisplay")
    migrated_res = migrated_tool.get("resultDisplay")
    
    if isinstance(native_res, dict) and isinstance(migrated_res, dict):
        print(f"Checking resultDisplay keys...")
        for k in set(native_res.keys()) | set(migrated_res.keys()):
            nv = native_res.get(k)
            mv = migrated_res.get(k)
            if k == "fileDiff":
                # Check for headers
                has_header = mv and ("Index:" in mv or "@@" in mv)
                print(f"  - {k}: Native length={len(str(nv))}, Migrated length={len(str(mv))} (Has header: {has_header})")
            elif k == "diffStat":
                print(f"  - {k}: Native={nv}, Migrated={mv}")
            else:
                print(f"  - {k}: Native={nv}, Migrated={mv}")
    else:
        status = "MATCH" if native_res == migrated_res else "MISMATCH"
        print(f"[{status}] resultDisplay (raw): Native='{native_res}', Migrated='{migrated_res}'")

def main():
    project_path = Path("../WorkspaceAgent").resolve()
    cc_adapter = ClaudeCodeAdapter()
    gemini_adapter = GeminiCliAdapter()

    # 1. Find the latest Claude Code session
    cc_sessions = cc_adapter.list_conversations(project_path)
    if not cc_sessions:
        print("Error: No Claude Code sessions found in WorkspaceAgent.")
        return
    
    latest_cc = cc_sessions[0]
    print(f"Found Claude Code session: {latest_cc.name} ({latest_cc.id})")
    
    # 2. Read it
    conv = cc_adapter.read_conversation(latest_cc.id, project_path)
    
    # 3. Migrate it
    migrated_filename = gemini_adapter.write_conversation(conv, project_path)
    migrated_path = _get_chats_dir(project_path) / migrated_filename
    print(f"Migrated to: {migrated_path}")

    # 4. Load native reference
    native_path = Path(r"C:\Users\troyh\.gemini\tmp\725fc5d5b07a88a3c7ebe6a13fc72d9e8f32a2a8c42df40e533be0fbf85bd883\chats\session-2026-04-12T04-30-bde99a86.json")
    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)
    
    # 5. Load migrated data
    with open(migrated_path, "r", encoding="utf-8") as f:
        migrated_data = json.load(f)

    # 6. Extract tools for comparison
    native_tools = [t for m in native_data['messages'] if 'toolCalls' in m for t in m['toolCalls']]
    migrated_tools = [t for m in migrated_data['messages'] if 'toolCalls' in m for t in m['toolCalls']]

    # Compare first Edit/Write tool
    n_edit = next((t for t in native_tools if t['name'] in ('replace', 'write_file', 'Edit')), None)
    m_edit = next((t for t in migrated_tools if t['name'] in ('replace', 'write_file', 'Edit')), None)
    
    if n_edit and m_edit:
        compare_tool_calls(n_edit, m_edit, "Edit")
    else:
        print(f"Warning: Could not find Edit tool calls to compare. Native count: {len(native_tools)}, Migrated count: {len(migrated_tools)}")

    # Compare first Shell tool
    n_shell = next((t for t in native_tools if t['name'] in ('run_shell_command', 'Shell')), None)
    m_shell = next((t for t in migrated_tools if t['name'] in ('run_shell_command', 'Shell')), None)
    
    if n_shell and m_shell:
        compare_tool_calls(n_shell, m_shell, "Shell")

def _get_chats_dir(project_path):
    # Temporary helper to access internal adapter logic for the script
    from agent_migrator.tools.gemini_cli import _get_chats_dir as gcd
    return gcd(project_path)

if __name__ == "__main__":
    main()
