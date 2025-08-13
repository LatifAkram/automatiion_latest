"""
Core components for the Multi-Agent Automation Platform.

This package contains the fundamental building blocks:
- Configuration management
- Database operations
- Vector store for embeddings
- AI provider abstraction
- Audit logging
- Multi-agent orchestration
"""

from .config import Config
from .database import Database
from .vector_store import VectorStore
from .ai_provider import AIProvider
from .audit import AuditLogger
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    "Config",
    "Database", 
    "VectorStore",
    "AIProvider",
    "AuditLogger",
    "MultiAgentOrchestrator"
]