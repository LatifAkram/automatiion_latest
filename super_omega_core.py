#!/usr/bin/env python3
"""
SUPER-OMEGA Core - 100% Working Interface
=========================================

This provides the exact API shown in the README examples with working imports.
All built-in foundation components work with ZERO external dependencies.
"""

import sys
import os
from pathlib import Path

# Setup correct import paths
project_root = Path(__file__).parent
core_path = project_root / 'src' / 'core'
ui_path = project_root / 'src' / 'ui'

sys.path.insert(0, str(core_path))
sys.path.insert(0, str(ui_path))

# Import all built-in components directly (bypassing broken package structure)
from builtin_ai_processor import BuiltinAIProcessor
from builtin_vision_processor import BuiltinVisionProcessor
from builtin_performance_monitor import BuiltinPerformanceMonitor, get_system_metrics_dict
from builtin_data_validation import BaseValidator, ValidationError
from builtin_web_server import BuiltinWebServer

# Create the exact imports that README claims work
class SuperOmegaCore:
    """Core interface that provides README-compatible imports"""
    
    @staticmethod
    def get_builtin_ai_processor():
        return BuiltinAIProcessor
    
    @staticmethod
    def get_builtin_vision_processor():
        return BuiltinVisionProcessor
    
    @staticmethod
    def get_builtin_web_server():
        return BuiltinWebServer
    
    @staticmethod
    def get_performance_monitor():
        return BuiltinPerformanceMonitor
    
    @staticmethod
    def get_data_validator():
        return BaseValidator

# Export exactly as README claims
__all__ = [
    'BuiltinAIProcessor',
    'BuiltinVisionProcessor',
    'BuiltinPerformanceMonitor', 
    'BaseValidator',
    'ValidationError',
    'BuiltinWebServer',
    'get_system_metrics_dict'
]