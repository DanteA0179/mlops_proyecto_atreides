"""Unit tests for ConversationManager."""

import pytest
from src.api.services.conversation_manager import ConversationManager


class TestConversationManager:
    """Tests for ConversationManager service."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = ConversationManager(max_messages=5)

    def test_add_message(self):
        """Test adding a message to conversation."""
        self.manager.add_message("test-id", "user", "Hello")

        context = self.manager.get_context("test-id")
        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert "timestamp" in context[0]

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there"),
            ("user", "How are you?"),
            ("assistant", "I'm good"),
        ]

        for role, content in messages:
            self.manager.add_message("test-id", role, content)

        context = self.manager.get_context("test-id")
        assert len(context) == 4

    def test_get_context_empty_conversation(self):
        """Test getting context for non-existent conversation."""
        context = self.manager.get_context("non-existent")
        assert context == []

    def test_max_messages_limit(self):
        """Test that context is limited to max_messages."""
        # Add more messages than max_messages
        for i in range(10):
            self.manager.add_message("test-id", "user", f"Message {i}")

        context = self.manager.get_context("test-id")
        # Should only return last 5 messages (max_messages=5)
        assert len(context) == 5
        assert context[0]["content"] == "Message 5"
        assert context[4]["content"] == "Message 9"

    def test_clear_conversation(self):
        """Test clearing a conversation."""
        self.manager.add_message("test-id", "user", "Hello")
        self.manager.add_message("test-id", "assistant", "Hi")

        self.manager.clear_conversation("test-id")

        context = self.manager.get_context("test-id")
        assert context == []

    def test_clear_non_existent_conversation(self):
        """Test clearing non-existent conversation (should not raise error)."""
        self.manager.clear_conversation("non-existent")
        # Should not raise any exception

    def test_multiple_conversations(self):
        """Test managing multiple conversations simultaneously."""
        self.manager.add_message("conv-1", "user", "Message 1-1")
        self.manager.add_message("conv-2", "user", "Message 2-1")
        self.manager.add_message("conv-1", "assistant", "Message 1-2")
        self.manager.add_message("conv-2", "assistant", "Message 2-2")

        context1 = self.manager.get_context("conv-1")
        context2 = self.manager.get_context("conv-2")

        assert len(context1) == 2
        assert len(context2) == 2
        assert context1[0]["content"] == "Message 1-1"
        assert context2[0]["content"] == "Message 2-1"

    def test_get_all_conversations(self):
        """Test getting all conversations."""
        self.manager.add_message("conv-1", "user", "Message 1")
        self.manager.add_message("conv-2", "user", "Message 2")

        all_convs = self.manager.get_all_conversations()
        assert len(all_convs) == 2
        assert "conv-1" in all_convs
        assert "conv-2" in all_convs

    def test_message_roles(self):
        """Test different message roles."""
        roles = ["user", "assistant", "system"]

        for role in roles:
            self.manager.add_message(f"test-{role}", role, f"Message from {role}")

        for role in roles:
            context = self.manager.get_context(f"test-{role}")
            assert context[0]["role"] == role

    def test_message_content_preservation(self):
        """Test that message content is preserved exactly."""
        content = "This is a message with special chars: @#$%^&*()"
        self.manager.add_message("test-id", "user", content)

        context = self.manager.get_context("test-id")
        assert context[0]["content"] == content

    def test_context_order(self):
        """Test that messages are returned in chronological order."""
        for i in range(5):
            self.manager.add_message("test-id", "user", f"Message {i}")

        context = self.manager.get_context("test-id")

        for i, msg in enumerate(context):
            assert msg["content"] == f"Message {i}"
