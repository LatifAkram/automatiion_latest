"""
Execution Models
===============

Data models for task execution, results, and logging.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Execution status for tasks and steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionStep(BaseModel):
    """Individual execution step."""
    task_id: str
    task_name: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    success: bool = False
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExecutionResult(BaseModel):
    """Complete execution result."""
    workflow_id: str
    success: bool
    steps: List[ExecutionStep] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_step(self, step: ExecutionStep):
        """Add an execution step."""
        self.steps.append(step)
        if step.success:
            self.success = self.success and True
        else:
            self.success = False
            if step.error:
                self.errors.append(step.error)
                
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.success = False
        
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
        
    def calculate_duration(self):
        """Calculate total execution duration."""
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        elif self.steps:
            start_times = [step.start_time for step in self.steps if step.start_time]
            end_times = [step.end_time for step in self.steps if step.end_time]
            
            if start_times and end_times:
                self.start_time = min(start_times)
                self.end_time = max(end_times)
                self.duration = (self.end_time - self.start_time).total_seconds()
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.calculate_duration()
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "steps_count": len(self.steps),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "duration": self.duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "artifacts_count": len(self.artifacts),
            "created_at": self.created_at.isoformat()
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExecutionLog(BaseModel):
    """Execution log for tracking workflow execution."""
    workflow_id: str
    steps: List[ExecutionStep] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    def add_step(self, step: ExecutionStep):
        """Add an execution step."""
        self.steps.append(step)
        if not self.start_time:
            self.start_time = step.start_time
        self.end_time = step.end_time
        
        if not step.success and step.error:
            self.errors.append(step.error)
            
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
        
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return len(self.errors) == 0 and all(step.success for step in self.steps)
        
    def get_progress(self) -> float:
        """Get execution progress as percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for step in self.steps if step.status == ExecutionStatus.COMPLETED)
        return (completed / len(self.steps)) * 100.0
        
    def get_duration(self) -> float:
        """Get total execution duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return self.duration
        
    def to_execution_result(self) -> ExecutionResult:
        """Convert to execution result."""
        result = ExecutionResult(
            workflow_id=self.workflow_id,
            success=self.is_successful(),
            steps=self.steps.copy(),
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.get_duration()
        )
        return result
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics for workflows and tasks."""
    workflow_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    success_rate: float = 0.0
    last_execution: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def update_metrics(self, execution_result: ExecutionResult):
        """Update metrics with new execution result."""
        self.total_executions += 1
        self.last_execution = execution_result.created_at
        
        if execution_result.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            
        # Update duration metrics
        duration = execution_result.duration
        if self.total_executions == 1:
            self.avg_duration = duration
            self.min_duration = duration
            self.max_duration = duration
        else:
            self.avg_duration = ((self.avg_duration * (self.total_executions - 1)) + duration) / self.total_executions
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            
        # Update success rate
        self.success_rate = self.successful_executions / self.total_executions
        self.updated_at = datetime.utcnow()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }