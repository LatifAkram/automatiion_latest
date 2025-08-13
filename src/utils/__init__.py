"""
Utility components for the Multi-Agent Automation Platform.

This package contains utility functions and classes:
- Media capture (screenshots, videos)
- Selector drift detection and self-healing
- Logging and monitoring utilities
"""

from .media_capture import MediaCapture
from .selector_drift import SelectorDriftDetector
from .logger import setup_logging
from .report_generator import ReportGenerator

__all__ = [
    "MediaCapture",
    "SelectorDriftDetector",
    "setup_logging",
    "ReportGenerator"
]