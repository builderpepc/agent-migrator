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

    # Deep check for resultDisplay
    native_res = native_tool.get("resultDisplay")
    migrated_res = migrated_tool.get("resultDisplay")
    
    # CRITICAL: Check if migrated_res is a string (failure) or a dict (success)
    print(f"resultDisplay Type: Native={type(native_res)}, Migrated={type(migrated_res)}")
    
    if isinstance(native_res, dict) and isinstance(migrated_res, dict):
        print(f"Checking resultDisplay keys...")
        for k in set(native_res.keys()) | set(migrated_res.keys()):
            nv = native_res.get(k)
            mv = migrated_res.get(k)
            status = "MATCH" if type(nv) == type(mv) else "MISMATCH"
            print(f"  - {k} ({status}): Native type={type(nv)}, Migrated type={type(mv)}")
    else:
        status = "MATCH" if native_res == migrated_res else "MISMATCH"
        if isinstance(migrated_res, str) and migrated_res.startswith("{"):
            print(f"[FAIL] resultDisplay is a STRING containing a dict. Must be a real DICT.")
        print(f"[{status}] resultDisplay (raw): Native='{native_res}', Migrated='{migrated_res}'")

def main():
    project_path = Path("../WorkspaceAgent").resolve()
    cc_adapter = ClaudeCodeAdapter()
    gemini_adapter = GeminiCliAdapter()

    # 1. Find the latest Claude Code session
    cc_sessions = cc_adapter.list_conversations(project_path)
    if not cc_sessions:
        print("Error: No Claude Code sessions found.")
        return
    
    latest_cc = cc_sessions[0]
    print(f"Found Claude Code session: {latest_cc.name}")
    
    # 2. Read and Migrate
    conv = cc_adapter.read_conversation(latest_cc.id, project_path)
    migrated_filename = gemini_adapter.write_conversation(conv, project_path)
    migrated_path = _get_chats_dir(project_path) / migrated_filename

    # 3. Load files
    native_path = Path(r"C:\Users\troyh\.gemini\tmp\725fc5d5b07a88a3c7ebe6a13fc72d9e8f32a2a8c42df40e533be0fbf85bd883\chats\session-2026-04-12T04-30-bde99a86.json")
    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)
    with open(migrated_path, "r", encoding="utf-8") as f:
        migrated_data = json.load(f)

    # 4. Extract tools
    native_tools = [t for m in native_data['messages'] if 'toolCalls' in m for t in m['toolCalls']]
    migrated_tools = [t for m in migrated_data['messages'] if 'toolCalls' in m for t in m['toolCalls']]

    n_edit = next((t for t in native_tools if t['name'] == 'write_file'), None)
    # Find any migrated edit tool
    m_edit = next((t for t in migrated_tools if t['name'] in ('replace', 'write_file')), None)
    
    if n_edit and m_edit:
        compare_tool_calls(n_edit, m_edit, "Edit")
    else:
        print("Could not find tools to compare.")

def _get_chats_dir(project_path):
    from agent_migrator.tools.gemini_cli import _get_chats_dir as gcd
    return gcd(project_path)

if __name__ == "__main__":
    main()
