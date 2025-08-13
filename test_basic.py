#!/usr/bin/env python3
"""
Basic Platform Test
==================

Direct test of individual components.
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


async def test_configuration():
    """Test configuration loading."""
    logger.info("🔧 Testing Configuration...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test config import
        from core.config import Config
        
        config = Config()
        logger.info("✅ Configuration loaded successfully")
        
        # Test config properties
        logger.info(f"Database path: {config.database.db_path}")
        logger.info(f"Vector DB path: {config.database.vector_db_path}")
        logger.info(f"API host: {config.api.host}")
        logger.info(f"API port: {config.api.port}")
        
        return {"success": True, "config": config}
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_database():
    """Test database functionality."""
    logger.info("🗄️ Testing Database...")
    
    try:
        from core.config import Config
        from core.database import DatabaseManager
        
        config = Config()
        db_manager = DatabaseManager(config.database)
        await db_manager.initialize()
        
        logger.info("✅ Database initialized successfully")
        
        # Test basic operations
        test_workflow = {
            "id": "test_workflow_001",
            "name": "Test Workflow",
            "description": "Test workflow for validation",
            "domain": "testing",
            "status": "planning",
            "created_at": datetime.utcnow().isoformat(),
            "parameters": {"test": True},
            "tags": ["test", "validation"]
        }
        
        await db_manager.save_workflow(test_workflow)
        logger.info("✅ Workflow saved successfully")
        
        # Retrieve workflow
        workflow = await db_manager.get_workflow("test_workflow_001")
        if workflow:
            logger.info(f"✅ Workflow retrieved: {workflow.name}")
        else:
            logger.warning("⚠️ Workflow not found after save")
        
        return {"success": True, "workflow_id": "test_workflow_001"}
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_vector_store():
    """Test vector store functionality."""
    logger.info("🔍 Testing Vector Store...")
    
    try:
        from core.config import Config
        from core.vector_store import VectorStore
        
        config = Config()
        vector_store = VectorStore(config.database)
        await vector_store.initialize()
        
        logger.info("✅ Vector store initialized successfully")
        
        # Test storing a plan
        test_plan = {
            "workflow_id": "test_plan_001",
            "analysis": {"domain": "testing"},
            "tasks": [{"name": "Test Task", "type": "test"}],
            "estimated_duration": 60,
            "success_probability": 0.9
        }
        
        await vector_store.store_plan("test_plan_001", test_plan)
        logger.info("✅ Plan stored successfully")
        
        # Test statistics
        stats = await vector_store.get_statistics()
        logger.info(f"✅ Vector store statistics: {stats}")
        
        return {"success": True, "stats": stats}
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_ai_provider():
    """Test AI provider functionality."""
    logger.info("🤖 Testing AI Provider...")
    
    try:
        from core.config import Config
        from core.ai_provider import AIProvider
        
        config = Config()
        ai_provider = AIProvider(config.ai)
        await ai_provider.initialize()
        
        logger.info("✅ AI provider initialized successfully")
        
        # Test response generation
        response = await ai_provider.generate_response(
            prompt="Hello, this is a test message.",
            max_tokens=50,
            temperature=0.7
        )
        
        logger.info(f"✅ AI response generated: {response[:100]}...")
        
        return {"success": True, "response": response}
        
    except Exception as e:
        logger.error(f"❌ AI provider test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_search_agent():
    """Test search agent functionality."""
    logger.info("🔎 Testing Search Agent...")
    
    try:
        from core.config import Config
        from agents.search import SearchAgent
        
        config = Config()
        search_agent = SearchAgent(config.search, None)  # No audit logger for test
        await search_agent.initialize()
        
        logger.info("✅ Search agent initialized successfully")
        
        # Test search functionality
        results = await search_agent.search(
            query="Python automation",
            max_results=3,
            sources=["duckduckgo"]
        )
        
        logger.info(f"✅ Search completed: {len(results)} results found")
        
        if results:
            logger.info(f"First result: {results[0].get('title', 'No title')[:50]}...")
        
        return {"success": True, "results_count": len(results)}
        
    except Exception as e:
        logger.error(f"❌ Search agent test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_dom_extraction():
    """Test DOM extraction functionality."""
    logger.info("🌐 Testing DOM Extraction...")
    
    try:
        from core.config import Config
        from agents.dom_extractor import DOMExtractionAgent
        
        config = Config()
        dom_agent = DOMExtractionAgent(config.automation, None)  # No audit logger for test
        await dom_agent.initialize()
        
        logger.info("✅ DOM extraction agent initialized successfully")
        
        # Test DOM extraction
        data = await dom_agent.extract_data(
            url="https://example.com",
            selectors={
                "title": "h1",
                "content": "p",
                "links": "a[href]"
            }
        )
        
        logger.info(f"✅ DOM extraction completed: {len(data)} fields extracted")
        
        if data:
            logger.info(f"Extracted data keys: {list(data.keys())}")
        
        return {"success": True, "data_fields": len(data)}
        
    except Exception as e:
        logger.error(f"❌ DOM extraction test failed: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Run all basic tests."""
    logger.info("🚀 Starting Basic Platform Tests...")
    
    tests = [
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Vector Store", test_vector_store),
        ("AI Provider", test_ai_provider),
        ("Search Agent", test_search_agent),
        ("DOM Extraction", test_dom_extraction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result.get("success"):
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: EXCEPTION - {e}")
            results[test_name] = {"success": False, "error": str(e)}
            
        await asyncio.sleep(1)
    
    # Generate report
    logger.info(f"\n{'='*60}")
    logger.info("📊 BASIC TEST REPORT")
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
    if passed_tests >= 5:
        logger.info("🟢 EXCELLENT: Core platform is working well")
    elif passed_tests >= 4:
        logger.info("🟡 GOOD: Platform has good foundation with minor issues")
    elif passed_tests >= 3:
        logger.info("🟠 FAIR: Platform has basic functionality but needs work")
    else:
        logger.info("🔴 POOR: Platform has critical issues")
    
    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())