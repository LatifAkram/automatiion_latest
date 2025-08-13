"""
Execution Models
==============

Models for execution results and performance tracking.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionLog(BaseModel):
    """Individual execution log entry."""
    
    id: str = Field(..., description="Log entry ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Log timestamp")
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR, DEBUG)")
    message: str = Field(..., description="Log message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Log context")
    
    class Config:
        use_enum_values = True


class ExecutionResult(BaseModel):
    """Result of workflow execution."""
    
    workflow_id: str = Field(..., description="Workflow ID")
    success: bool = Field(..., description="Execution success status")
    status: ExecutionStatus = Field(..., description="Execution status")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    duration: float = Field(default=0.0, description="Execution duration in seconds")
    
    # Results
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Execution steps")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Performance metrics
    total_tasks: int = Field(default=0, description="Total tasks executed")
    successful_tasks: int = Field(default=0, description="Successfully completed tasks")
    failed_tasks: int = Field(default=0, description="Failed tasks")
    
    # Data
    data_extracted: Dict[str, Any] = Field(default_factory=dict, description="Extracted data")
    files_processed: List[str] = Field(default_factory=list, description="Processed files")
    api_calls_made: int = Field(default=0, description="Number of API calls")
    
    # Logs
    logs: List[ExecutionLog] = Field(default_factory=list, description="Execution logs")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        
    def add_step(self, step: Dict[str, Any]):
        """Add execution step."""
        self.steps.append(step)
        
    def add_error(self, error: str):
        """Add error message."""
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add warning message."""
        self.warnings.append(warning)
        
    def add_log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Add execution log."""
        log = ExecutionLog(
            id=f"log_{len(self.logs)}_{datetime.utcnow().timestamp()}",
            level=level,
            message=message,
            context=context or {}
        )
        self.logs.append(log)
        
    def mark_completed(self, success: bool = True):
        """Mark execution as completed."""
        self.completed_at = datetime.utcnow()
        self.duration = (self.completed_at - self.started_at).total_seconds()
        self.success = success
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        
    def mark_failed(self, error: str):
        """Mark execution as failed."""
        self.add_error(error)
        self.mark_completed(success=False)
        
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.successful_tasks / self.total_tasks) * 100
        
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "status": self.status.value,
            "duration": self.duration,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.get_success_rate(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "api_calls": self.api_calls_made,
            "files_processed": len(self.files_processed)
        }