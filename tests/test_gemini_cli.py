import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from agent_migrator.models import Conversation, ConversationInfo, TextMessage, ToolCallMessage
from agent_migrator.tools.gemini_cli import GeminiCliAdapter, _get_project_hash, _normalize_path


class TestGeminiCliAdapter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.gemini_home = Path(self.test_dir) / ".gemini"
        os.environ["GEMINI_CLI_HOME"] = str(self.gemini_home)
        
        # Mock project root (with .git)
        self.project_root = Path(self.test_dir) / "my-project"
        self.project_root.mkdir()
        (self.project_root / ".git").mkdir()
        
        # Subdirectory (where user might run the tool)
        self.project_sub = self.project_root / "src"
        self.project_sub.mkdir()
        
        # Mock projects.json registry
        self.project_id = "custom-slug"
        self.gemini_home.mkdir(parents=True)
        registry = {
            "projects": {
                _normalize_path(str(self.project_root)): self.project_id
            }
        }
        (self.gemini_home / "projects.json").write_text(json.dumps(registry))
        
        self.chats_dir = self.gemini_home / "tmp" / self.project_id / "chats"
        self.chats_dir.mkdir(parents=True)
        
        # Create a mock JSON session file (monolithic)
        self.session_id = "test-session-uuid"
        self.session_data = {
            "sessionId": self.session_id,
            "projectHash": _get_project_hash(self.project_root),
            "startTime": "2026-04-11T10:00:00Z",
            "lastUpdated": "2026-04-11T10:05:00Z",
            "summary": "Test Conversation",
            "kind": "main",
            "messages": [
                {
                    "id": "msg1",
                    "timestamp": "2026-04-11T10:00:00Z",
                    "type": "user",
                    "content": [{"text": "Hello Gemini"}]
                }
            ]
        }
        self.session_file = self.chats_dir / f"session-2026-04-11T10-00-{self.session_id[:8]}.json"
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if "GEMINI_CLI_HOME" in os.environ:
            del os.environ["GEMINI_CLI_HOME"]

    def test_list_conversations_from_subfolder(self):
        """Should find the root and look up the slug in projects.json."""
        adapter = GeminiCliAdapter()
        convs = adapter.list_conversations(self.project_sub)
        
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].name, "Test Conversation")

    def test_write_conversation_uses_registry(self):
        """Should write to the custom-slug folder, not a hash folder."""
        adapter = GeminiCliAdapter()
        info = ConversationInfo(
            id="new-id",
            name="New Migration",
            updated_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            message_count=1,
            size_bytes=0,
            source_tool="cursor"
        )
        new_conv = Conversation(
            info=info,
            turns=[TextMessage(role="user", text="Migrated Text")]
        )
        
        filename = adapter.write_conversation(new_conv, self.project_sub)
        self.assertTrue((self.chats_dir / filename).exists())
        self.assertTrue(filename.endswith(".json"))
        
        # Verify JSON content
        with open(self.chats_dir / filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["kind"], "main")
            self.assertEqual(data["projectHash"], _get_project_hash(self.project_root))

    def test_write_conversation_maps_tools(self):
        """Should map CC tools to Gemini CLI 0.37.1 canonical names and kinds."""
        adapter = GeminiCliAdapter()
        info = ConversationInfo(
            id="tool-test", name="Tool Test",
            updated_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            message_count=1, size_bytes=0, source_tool="claude_code"
        )
        new_conv = Conversation(
            info=info,
            turns=[
                ToolCallMessage(name="run_shell_command", input={"command": "whoami"}, result="laptop-user"),
                ToolCallMessage(name="read_file", input={"file_path": "test.py"}, result="print('hello')"),
                ToolCallMessage(name="replace", input={"file_path": "main.py"}, result="--- diff ---")
            ]
        )
        
        filename = adapter.write_conversation(new_conv, self.project_sub)
        with open(self.chats_dir / filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            tools = data["messages"][0]["toolCalls"]
            self.assertEqual(len(tools), 3)
            
            # Shell tool mapping
            self.assertEqual(tools[0]["name"], "run_shell_command")
            self.assertEqual(tools[0]["kind"], "execute")
            self.assertTrue(tools[0]["description"].startswith("whoami [current working directory"))
            
            # Read tool mapping
            self.assertEqual(tools[1]["name"], "read_file")
            self.assertEqual(tools[1]["kind"], "read")
            self.assertEqual(tools[1]["description"], "test.py")
            
            # Edit tool mapping
            self.assertEqual(tools[2]["name"], "replace")
            self.assertEqual(tools[2]["kind"], "edit")
            self.assertEqual(tools[2]["description"], "main.py")

if __name__ == "__main__":
    unittest.main()
