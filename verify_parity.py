import json
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from agent_migrator.tools.claude_code import ClaudeCodeAdapter, encode_project_path
from agent_migrator.models import Conversation, TextMessage, ToolCallMessage

def main():
    cc_adapter = ClaudeCodeAdapter()
    project_path = Path(".").resolve()
    
    # 1. Find a native CC session from history
    sessions = cc_adapter.list_conversations(project_path)
    native_session = None
    for s in sessions:
        if s.created_at.date() < datetime(2026, 4, 11).date():
            native_session = s
            break
    
    if not native_session:
        print("No native CC sessions found before today.")
        return

    print(f"Testing with native session: {native_session.id} ({native_session.name})")
    
    # 2. Read into intermediary format
    conv = cc_adapter.read_conversation(native_session.id, project_path)
    
    bash_turns = [t for t in conv.turns if isinstance(t, ToolCallMessage) and t.name == "Bash"]
    if not bash_turns:
        print("No Bash turns found in this native session.")
        return
    
    print(f"Found {len(bash_turns)} Bash turns in intermediary format.")
    first_bash = bash_turns[0]
    print(f"Intermediary Name: {first_bash.name}")
    print(f"Intermediary Input: {first_bash.input}")
    print(f"Intermediary Result (repr): {repr(first_bash.result)[:200]}...")
    
    # 3. Write back to a new CC session
    new_id = cc_adapter.write_conversation(conv, project_path)
    print(f"Wrote back to new session: {new_id}")
    
    # 4. Compare the raw JSONL records
    encoded = encode_project_path(project_path)
    native_jsonl = Path.home() / ".claude" / "projects" / encoded / f"{native_session.id}.jsonl"
    migrated_jsonl = Path.home() / ".claude" / "projects" / encoded / f"{new_id}.jsonl"
    
    def get_bash_result_rec(path):
        with open(path, "r", encoding="utf-8") as f:
            # We need to find the tool_result record for a Bash tool.
            # First map tuid -> name from assistant records.
            tool_uses = {}
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "assistant":
                    content = rec.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for b in content:
                            if b.get("type") == "tool_use":
                                tool_uses[b["id"]] = b["name"]
                if rec.get("type") == "user":
                    content = rec.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for b in content:
                            if b.get("type") == "tool_result":
                                tuid = b.get("tool_use_id")
                                if tool_uses.get(tuid) == "Bash":
                                    return rec
        return None

    native_rec = get_bash_result_rec(native_jsonl)
    migrated_rec = get_bash_result_rec(migrated_jsonl)
    
    print("\n--- Comparing Raw JSONL Records for Bash Tool Result ---")
    if native_rec:
        print("NATIVE Record (snippet):")
        # Just show the tool_result block and toolUseResult
        tr = next(b for b in native_rec["message"]["content"] if b["type"] == "tool_result")
        print(f"  tool_result.content: {repr(tr['content'])[:100]}...")
        print(f"  toolUseResult: {json.dumps(native_rec.get('toolUseResult'), indent=2)}")
    
    if migrated_rec:
        print("\nMIGRATED Record (snippet):")
        tr = next(b for b in migrated_rec["message"]["content"] if b["type"] == "tool_result")
        print(f"  tool_result.content: {repr(tr['content'])[:100]}...")
        print(f"  toolUseResult: {json.dumps(migrated_rec.get('toolUseResult'), indent=2)}")

if __name__ == "__main__":
    main()
