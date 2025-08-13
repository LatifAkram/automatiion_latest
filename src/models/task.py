"""
Task Model
=========

Task definition and status tracking for workflow execution.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskType(str, Enum):
    """Task types for different automation scenarios."""
    WEB_NAVIGATION = "web_navigation"
    DATA_EXTRACTION = "data_extraction"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    DATA_PROCESSING = "data_processing"
    EMAIL_SEND = "email_send"
    DATABASE_QUERY = "database_query"
    IMAGE_PROCESSING = "image_processing"
    DOCUMENT_PROCESSING = "document_processing"
    SOCIAL_MEDIA = "social_media"
    ECOMMERCE = "ecommerce"
    BANKING = "banking"
    TRAVEL = "travel"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    CUSTOM = "custom"


class Task(BaseModel):
    """Task definition for workflow execution."""
    
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(default="", description="Task description")
    type: TaskType = Field(..., description="Task type")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    
    # Execution parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Results
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Performance metrics
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Task tags")
    priority: int = Field(default=1, description="Task priority (1-10)")
    
    class Config:
        use_enum_values = True
        
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        
    def is_running(self) -> bool:
        """Check if task is running."""
        return self.status == TaskStatus.RUNNING
        
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries
        
    def mark_started(self):
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()
        
    def mark_completed(self, result: Optional[Dict[str, Any]] = None):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
            
    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
            
    def mark_retrying(self):
        """Mark task for retry."""
        self.status = TaskStatus.RETRYING
        self.retry_count += 1
        self.started_at = None
        self.completed_at = None
        self.error = None