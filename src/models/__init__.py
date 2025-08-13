"""
Data models for the Multi-Agent Automation Platform.

This package contains the data structures and models:
- Workflow definitions and execution
- Task management and status tracking
- Execution results and performance metrics
- Conversation and message handling
"""

from .workflow import Workflow, WorkflowStatus, WorkflowExecution, WorkflowStep
from .execution import ExecutionResult, PerformanceMetrics
from .conversation import Conversation, Message, MessageType
from .task import TaskStatus, TaskType

__all__ = [
    "Workflow",
    "WorkflowStatus", 
    "WorkflowExecution",
    "WorkflowStep",
    "ExecutionResult",
    "PerformanceMetrics",
    "Conversation",
    "Message",
    "MessageType",
    "TaskStatus",
    "TaskType"
]