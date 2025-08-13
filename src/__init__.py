"""
Autonomous Multi-Agent Automation Platform
=========================================

A comprehensive, autonomous, adaptive, multi-agent automation platform
capable of executing ultra-complex workflows across diverse domains.

This package contains the core components:
- Multi-agent orchestration
- AI-powered planning and execution
- Web automation with self-healing
- Search and data extraction
- Conversational AI with reasoning
- Vector-based learning and memory
- Enterprise compliance and audit
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Automation Platform"
__description__ = "Autonomous Multi-Agent Automation Platform"

# Core exports
from .core.orchestrator import MultiAgentOrchestrator
from .core.config import Config
from .core.database import DatabaseManager
from .core.vector_store import VectorStore
from .core.ai_provider import AIProvider
from .core.audit import AuditLogger

# Agent exports
from .agents.planner import PlannerAgent
from .agents.executor import ExecutionAgent
from .agents.conversational import ConversationalAgent
from .agents.search import SearchAgent
from .agents.dom_extractor import DOMExtractionAgent

# Model exports
from .models.workflow import Workflow, WorkflowStatus, WorkflowExecution
from .models.execution import ExecutionResult, PerformanceMetrics
from .models.conversation import Conversation, Message, MessageType
from .models.task import TaskStatus, TaskType

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
    "AIProvider",
    "AuditLogger",
    
    # Agents
    "PlannerAgent",
    "ExecutionAgent", 
    "ConversationalAgent",
    "SearchAgent",
    "DOMExtractionAgent",
    
    # Models
    "Workflow",
    "WorkflowStatus",
    "WorkflowExecution", 
    "ExecutionResult",
    "PerformanceMetrics",
    "Conversation",
    "Message",
    "MessageType",
    "TaskStatus",
    "TaskType",
    
    # Utils
    "MediaCapture",
    "SelectorDriftDetector",
    "setup_logging"
]