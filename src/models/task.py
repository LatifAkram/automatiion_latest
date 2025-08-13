"""
Task Models
==========

Data models for task definitions and execution tracking.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of tasks that can be executed."""
    WEB_AUTOMATION = "web_automation"
    API_CALL = "api_call"
    DOM_EXTRACTION = "dom_extraction"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    SEARCH = "search"
    GENERAL = "general"


class TaskStatus(str, Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Task(BaseModel):
    """Task definition and metadata."""
    id: str
    name: str
    type: TaskType
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 0
    max_retries: int = 3
    priority: str = "medium"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }