"""
Conversation Models
==================

Data models for conversation management and context tracking.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Message types in conversations."""
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class Message(BaseModel):
    """Individual message in a conversation."""
    content: str
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    """Conversation session with message history."""
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None
        
    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
        
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }