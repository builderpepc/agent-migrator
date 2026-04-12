import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from agent_migrator.models import Conversation, ConversationInfo, TextMessage
from agent_migrator.tools.gemini_cli import GeminiCliAdapter, _get_project_hash


class TestGeminiCliAdapter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.gemini_home = Path(self.test_dir) / ".gemini"
        os.environ["GEMINI_CLI_HOME"] = str(self.gemini_home)
        
        self.project_path = Path(self.test_dir) / "my-project"
        self.project_path.mkdir()
        
        self.project_hash = _get_project_hash(self.project_path)
        self.chats_dir = self.gemini_home / "tmp" / self.project_hash / "chats"
        self.chats_dir.mkdir(parents=True)
        
        # Create a mock session file
        self.session_id = "test-session-uuid"
        self.session_data = {
            "sessionId": self.session_id,
            "projectHash": self.project_hash,
            "startTime": "2026-04-11T10:00:00Z",
            "lastUpdated": "2026-04-11T10:05:00Z",
            "summary": "Test Conversation",
            "messages": [
                {
                    "id": "msg1",
                    "timestamp": "2026-04-11T10:00:00Z",
                    "type": "user",
                    "content": [{"text": "Hello Gemini"}]
                },
                {
                    "id": "msg2",
                    "timestamp": "2026-04-11T10:01:00Z",
                    "type": "gemini",
                    "content": [{"text": "Hello User"}]
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

    def test_list_conversations(self):
        adapter = GeminiCliAdapter()
        convs = adapter.list_conversations(self.project_path)
        
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].name, "Test Conversation")
        self.assertEqual(convs[0].message_count, 2)

    def test_read_conversation(self):
        adapter = GeminiCliAdapter()
        convs = adapter.list_conversations(self.project_path)
        conv = adapter.read_conversation(convs[0].id, self.project_path)
        
        self.assertEqual(len(conv.turns), 2)
        self.assertIsInstance(conv.turns[0], TextMessage)
        self.assertEqual(conv.turns[0].text, "Hello Gemini")
        self.assertEqual(conv.turns[1].text, "Hello User")

    def test_write_conversation(self):
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
        
        filename = adapter.write_conversation(new_conv, self.project_path)
        self.assertTrue((self.chats_dir / filename).exists())
        
        # Verify written content
        with open(self.chats_dir / filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["summary"], "New Migration")
            self.assertEqual(data["messages"][0]["content"][0]["text"], "Migrated Text")

if __name__ == "__main__":
    unittest.main()
