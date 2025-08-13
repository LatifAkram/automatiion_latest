"""
Interactive Training System for Autonomous Multi-Agent Automation Platform

This module provides a comprehensive training system with real interactive tutorials,
progress tracking, skill assessment, and personalized learning paths. All training
uses real platform features with zero placeholders, mock data, or simulations.
"""

from .training_manager import TrainingManager
from .course_manager import CourseManager
from .progress_tracker import ProgressTracker
from .skill_assessor import SkillAssessor
from .interactive_tutorial import InteractiveTutorial
from .certification_system import CertificationSystem

__version__ = "1.0.0"
__author__ = "Autonomous Automation Platform"
__description__ = "Interactive Training System"

__all__ = [
    "TrainingManager",
    "CourseManager", 
    "ProgressTracker",
    "SkillAssessor",
    "InteractiveTutorial",
    "CertificationSystem",
]