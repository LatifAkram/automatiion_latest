"""
API components for the Multi-Agent Automation Platform.

This package contains the REST API server and endpoints:
- FastAPI server setup
- Workflow management endpoints
- Agent interaction endpoints
- Status and monitoring endpoints
"""

from .server import start_api_server

__all__ = [
    "start_api_server"
]