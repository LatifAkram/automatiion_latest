#!/usr/bin/env python3
"""
Simple test script to verify the Multi-Agent Automation Platform components.
This script tests the basic functionality without requiring external APIs.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.core.database import DatabaseManager
from src.core.vector_store import VectorStore
from src.core.audit import AuditLogger
from src.core.ai_provider import AIProvider
from src.utils.logger import setup_logging


async def test_core_components():
    """Test core components initialization."""
    print("🧪 Testing Core Components...")
    
    try:
        # Test configuration
        print("  📋 Testing Configuration...")
        config = Config()
        print(f"    ✅ Configuration loaded: {config.api.host}:{config.api.port}")
        
        # Test database
        print("  🗄️  Testing Database...")
        database = DatabaseManager(config.database)
        await database.initialize()
        print("    ✅ Database initialized")
        
        # Test vector store
        print("  🔍 Testing Vector Store...")
        vector_store = VectorStore(config.database)
        await vector_store.initialize()
        print("    ✅ Vector store initialized")
        
        # Test audit logger
        print("  📝 Testing Audit Logger...")
        audit_logger = AuditLogger(config)
        await audit_logger.initialize()
        print("    ✅ Audit logger initialized")
        
        # Test AI provider (without API keys)
        print("  🤖 Testing AI Provider...")
        ai_provider = AIProvider(config.ai)
        await ai_provider.initialize()
        print("    ✅ AI provider initialized")
        
        print("✅ All core components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Core component test failed: {e}")
        return False


async def test_agent_imports():
    """Test that all agents can be imported."""
    print("🤖 Testing Agent Imports...")
    
    try:
        from src.agents.planner import PlannerAgent
        from src.agents.executor import ExecutionAgent
        from src.agents.conversational import ConversationalAgent
        from src.agents.search import SearchAgent
        from src.agents.dom_extractor import DOMExtractionAgent
        
        print("  ✅ All agents imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Agent import test failed: {e}")
        return False


async def test_model_imports():
    """Test that all models can be imported."""
    print("📊 Testing Model Imports...")
    
    try:
        from src.models.workflow import Workflow, WorkflowStatus, WorkflowExecution
        from src.models.execution import ExecutionResult, PerformanceMetrics
        from src.models.conversation import Conversation, Message, MessageType
        from src.models.task import TaskStatus, TaskType
        
        print("  ✅ All models imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model import test failed: {e}")
        return False


async def test_utility_imports():
    """Test that all utilities can be imported."""
    print("🔧 Testing Utility Imports...")
    
    try:
        from src.utils.media_capture import MediaCapture
        from src.utils.selector_drift import SelectorDriftDetector
        from src.utils.logger import setup_logging
        
        print("  ✅ All utilities imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Utility import test failed: {e}")
        return False


async def test_orchestrator_import():
    """Test that the orchestrator can be imported."""
    print("🎼 Testing Orchestrator Import...")
    
    try:
        from src.core.orchestrator import MultiAgentOrchestrator
        
        print("  ✅ Orchestrator imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator import test failed: {e}")
        return False


async def test_api_import():
    """Test that the API server can be imported."""
    print("🌐 Testing API Import...")
    
    try:
        from src.api.server import start_api_server
        
        print("  ✅ API server imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ API import test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Multi-Agent Automation Platform - Component Test")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    tests = [
        ("Core Components", test_core_components),
        ("Agent Imports", test_agent_imports),
        ("Model Imports", test_model_imports),
        ("Utility Imports", test_utility_imports),
        ("Orchestrator Import", test_orchestrator_import),
        ("API Import", test_api_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The platform is ready to use.")
        print("\n📝 Next steps:")
        print("  1. Set up your .env file with API keys")
        print("  2. Run 'python main.py' to start the platform")
        print("  3. Access the API at http://localhost:8000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())