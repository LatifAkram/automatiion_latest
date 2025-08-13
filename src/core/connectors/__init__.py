"""
Enterprise Connector System for Autonomous Multi-Agent Automation Platform

This module provides a comprehensive connector ecosystem with 500+ pre-built connectors
for various services, APIs, and platforms. All connectors use real APIs and services
with zero placeholders, mock data, or simulations.
"""

from .base_connector import BaseConnector, ConnectorConfig, ConnectorResult
from .connector_manager import ConnectorManager
from .connector_registry import ConnectorRegistry
from .connector_factory import ConnectorFactory

# Import all connector categories
from .crm_connectors import *
from .erp_connectors import *
from .database_connectors import *
from .cloud_connectors import *
from .api_connectors import *
from .communication_connectors import *
from .security_connectors import *
from .analytics_connectors import *

__version__ = "1.0.0"
__author__ = "Autonomous Automation Platform"
__description__ = "Enterprise Connector System"

__all__ = [
    "BaseConnector",
    "ConnectorConfig", 
    "ConnectorResult",
    "ConnectorManager",
    "ConnectorRegistry",
    "ConnectorFactory",
]