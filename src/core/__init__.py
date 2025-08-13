"""
Core Package
===========

Core components of the multi-agent automation platform.
"""

# Core components
from .config import Config
from .database import DatabaseManager
from .vector_store import VectorStore
from .audit import AuditLogger
from .ai_provider import AIProvider

__all__ = [
    'Config',
    'DatabaseManager', 
    'VectorStore',
    'AuditLogger',
    'AIProvider'
]