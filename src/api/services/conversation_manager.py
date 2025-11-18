"""Conversation history and context management."""

import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context.

    Uses in-memory storage (dict) for MVP.
    Future enhancement: Redis or database for persistence.
    """

    def __init__(self, max_messages: int = 10):
        """
        Initialize ConversationManager.

        Parameters
        ----------
        max_messages : int
            Maximum number of messages to keep in context
        """
        self.conversations: dict[str, list[dict]] = defaultdict(list)
        self.max_messages = max_messages

    def get_context(self, conversation_id: str) -> list[dict]:
        """
        Get conversation context.

        Retrieves the most recent messages for the given conversation.

        Parameters
        ----------
        conversation_id : str
            Conversation UUID

        Returns
        -------
        list[dict]
            Last N messages in format [{"role": "user", "content": "..."}]

        Examples
        --------
        >>> manager = ConversationManager()
        >>> manager.add_message("test-id", "user", "Hello")
        >>> context = manager.get_context("test-id")
        >>> len(context)
        1
        """
        messages = self.conversations.get(conversation_id, [])
        return messages[-self.max_messages:]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> None:
        """
        Add message to conversation history.

        Parameters
        ----------
        conversation_id : str
            Conversation UUID
        role : str
            Message role: "user", "assistant", or "system"
        content : str
            Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.conversations[conversation_id].append(message)

        # Trim to max_messages
        if len(self.conversations[conversation_id]) > self.max_messages * 2:
            self.conversations[conversation_id] = \
                self.conversations[conversation_id][-self.max_messages * 2:]

        logger.debug(f"Added message to conversation {conversation_id}: {role}")

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear conversation history.

        Parameters
        ----------
        conversation_id : str
            Conversation UUID to clear
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")

    def get_all_conversations(self) -> dict[str, list[dict]]:
        """
        Get all conversations.

        Returns
        -------
        dict[str, list[dict]]
            All conversation histories (for debugging)
        """
        return dict(self.conversations)
