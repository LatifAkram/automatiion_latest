#!/usr/bin/env python3
"""Test startup script to identify issues."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

async def test_startup():
    """Test startup components."""
    try:
        print("🔧 Testing imports...")
        
        from src.core.config import Config
        print("✅ Config imported")
        
        config = Config()
        print("✅ Config created")
        
        from src.core.orchestrator import MultiAgentOrchestrator
        print("✅ Orchestrator imported")
        
        print("🔧 Testing orchestrator initialization...")
        orchestrator = MultiAgentOrchestrator(config)
        print("✅ Orchestrator created")
        
        await orchestrator.initialize()
        print("✅ Orchestrator initialized")
        
        print("🎉 SUCCESS: All components working!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_startup())
    if result:
        print("🚀 Platform ready for testing!")
    else:
        print("🔧 Need to fix startup issues")