"""Request models for copilot endpoint."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ConversationMessage(BaseModel):
    """Single message in conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: str | None = None


class ChatRequest(BaseModel):
    """Request model for /copilot/chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query in natural language",
        examples=["What will be the consumption tomorrow at 10am with Medium load?"]
    )

    conversation_id: str | None = Field(
        default=None,
        description="UUID for conversation tracking. Auto-generated if not provided."
    )

    context: list[ConversationMessage] | None = Field(
        default=None,
        max_length=10,
        description="Previous messages for context (max 10)"
    )

    @model_validator(mode='after')
    def validate_and_set_conversation_id(self):
        """Generate UUID for conversation_id if not provided."""
        if self.conversation_id is None or self.conversation_id == '':
            self.conversation_id = str(uuid.uuid4())
        else:
            # Validate UUID format
            try:
                uuid.UUID(self.conversation_id)
            except ValueError:
                raise ValueError(f"Invalid conversation_id format: {self.conversation_id}")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What will be the consumption tomorrow at 10am?",
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
                }
            ]
        }
    }
