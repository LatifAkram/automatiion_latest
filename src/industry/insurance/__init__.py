"""
Insurance Industry Automation Module
====================================

Complete automation solutions for the insurance industry, featuring:
- Comprehensive Guidewire platform integration (ALL 18 platforms)
- Real-time data streaming and synchronization
- Cross-platform workflow automation
- Enterprise-grade security and compliance
"""

from .guidewire_automation import (
    GuidewireProduct,
    GuidewireConfig,
    UniversalGuidewireOrchestrator
)

from .complete_guidewire_platform import (
    CompleteGuidewirePlatformOrchestrator,
    GuidewirePlatform,
    GuidewireConnection,
    GuidewireAPIType,
    RealTimeDataStream,
    create_complete_guidewire_orchestrator
)

__all__ = [
    # Legacy Guidewire (for backward compatibility)
    'GuidewireProduct',
    'GuidewireConfig', 
    'UniversalGuidewireOrchestrator',
    
    # Complete Guidewire Platform System (NEW)
    'CompleteGuidewirePlatformOrchestrator',
    'GuidewirePlatform',
    'GuidewireConnection',
    'GuidewireAPIType',
    'RealTimeDataStream',
    'create_complete_guidewire_orchestrator'
]