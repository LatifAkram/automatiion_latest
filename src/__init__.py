"""
Multi-Agent Automation Platform
==============================

A comprehensive, autonomous, adaptive multi-agent automation platform
capable of executing ultra-complex workflows across diverse domains.
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Automation Team"
__description__ = "Advanced automation platform with AI-powered agents"

# Core exports
from .core.orchestrator import MultiAgentOrchestrator
from .core.config import Config
from .core.database import DatabaseManager
from .core.vector_store import VectorStore
from .core.audit import AuditLogger
from .core.ai_provider import AIProvider

# Agent exports
from .agents.planner import PlannerAgent
from .agents.executor import ExecutionAgent
from .agents.conversational import ConversationalAgent
from .agents.search import SearchAgent
from .agents.dom_extractor import DOMExtractionAgent

# Model exports
from .models.workflow import Workflow, WorkflowStep, WorkflowStatus
from .models.task import Task, TaskStatus, TaskType
from .models.execution import ExecutionResult, ExecutionLog
from .models.conversation import Conversation, Message, MessageType

# Utility exports
from .utils.media_capture import MediaCapture
from .utils.selector_drift import SelectorDriftDetector
from .utils.logger import setup_logging

__all__ = [
    # Core
    "MultiAgentOrchestrator",
    "Config",
    "DatabaseManager",
    "VectorStore",
    "AuditLogger",
    "AIProvider",
    
    # Agents
    "PlannerAgent",
    "ExecutionAgent",
    "ConversationalAgent",
    "SearchAgent",
    "DOMExtractionAgent",
    
    # Models
    "Workflow",
    "WorkflowStep",
    "WorkflowStatus",
    "Task",
    "TaskStatus",
    "TaskType",
    "ExecutionResult",
    "ExecutionLog",
    "Conversation",
    "Message",
    "MessageType"
]