#!/usr/bin/env python3
"""
Simple Platform Test
==================

Direct test of core platform functionality.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_core_components():
    """Test core components directly."""
    logger.info("🚀 Testing Core Components...")
    
    results = {}
    
    # Test 1: Configuration
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from core.config import Config
        
        config = Config()
        logger.info("✅ Configuration loaded successfully")
        results["Configuration"] = {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        results["Configuration"] = {"success": False, "error": str(e)}
    
    # Test 2: Database Manager
    try:
        from core.database import DatabaseManager
        
        db_manager = DatabaseManager(config.database)
        await db_manager.initialize()
        logger.info("✅ Database manager initialized successfully")
        results["Database"] = {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        results["Database"] = {"success": False, "error": str(e)}
    
    # Test 3: Vector Store
    try:
        from core.vector_store import VectorStore
        
        vector_store = VectorStore(config.database)
        await vector_store.initialize()
        logger.info("✅ Vector store initialized successfully")
        results["Vector Store"] = {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        results["Vector Store"] = {"success": False, "error": str(e)}
    
    # Test 4: AI Provider
    try:
        from core.ai_provider import AIProvider
        
        ai_provider = AIProvider(config.ai)
        await ai_provider.initialize()
        logger.info("✅ AI provider initialized successfully")
        results["AI Provider"] = {"success": True}
        
    except Exception as e:
        logger.error(f"❌ AI provider test failed: {e}")
        results["AI Provider"] = {"success": False, "error": str(e)}
    
    # Test 5: Audit Logger
    try:
        from core.audit import AuditLogger
        
        audit_logger = AuditLogger(config)
        await audit_logger.initialize()
        logger.info("✅ Audit logger initialized successfully")
        results["Audit Logger"] = {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Audit logger test failed: {e}")
        results["Audit Logger"] = {"success": False, "error": str(e)}
    
    return results


async def test_agents():
    """Test agent components."""
    logger.info("🤖 Testing Agent Components...")
    
    results = {}
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from core.config import Config
        from agents.search import SearchAgent
        
        config = Config()
        search_agent = SearchAgent(config.search, None)
        await search_agent.initialize()
        
        # Test search functionality
        search_results = await search_agent.search(
            query="Python automation",
            max_results=3,
            sources=["duckduckgo"]
        )
        
        logger.info(f"✅ Search agent test successful: {len(search_results)} results")
        results["Search Agent"] = {"success": True, "results": len(search_results)}
        
    except Exception as e:
        logger.error(f"❌ Search agent test failed: {e}")
        results["Search Agent"] = {"success": False, "error": str(e)}
    
    return results


async def generate_report(results):
    """Generate test report."""
    logger.info(f"\n{'='*60}")
    logger.info("📊 PLATFORM TEST REPORT")
    logger.info(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get("success"))
    failed_tests = total_tests - passed_tests
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    logger.info(f"\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result.get("success") else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result.get("success"):
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Assessment
    logger.info(f"\n🎯 PLATFORM ASSESSMENT:")
    if passed_tests >= 4:
        logger.info("🟢 EXCELLENT: Core platform is working well")
    elif passed_tests >= 3:
        logger.info("🟡 GOOD: Platform has good foundation with some issues")
    elif passed_tests >= 2:
        logger.info("🟠 FAIR: Platform has basic functionality but needs work")
    else:
        logger.info("🔴 POOR: Platform has critical issues")
    
    logger.info(f"\n{'='*60}")


async def main():
    """Main test function."""
    logger.info("🚀 Starting Platform Testing...")
    
    # Test core components
    core_results = await test_core_components()
    
    # Test agents
    agent_results = await test_agents()
    
    # Combine results
    all_results = {**core_results, **agent_results}
    
    # Generate report
    await generate_report(all_results)


if __name__ == "__main__":
    asyncio.run(main())