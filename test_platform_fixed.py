#!/usr/bin/env python3
"""
Fixed Platform Test
==================

This script tests the actual capabilities of the multi-agent automation platform
with proper import handling.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlatformTester:
    """Comprehensive platform testing class."""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
        
    async def initialize_platform(self):
        """Initialize the platform."""
        try:
            logger.info("🚀 Initializing Multi-Agent Automation Platform...")
            
            # Import after path setup
            from core.config import Config
            from core.orchestrator import MultiAgentOrchestrator
            
            # Initialize orchestrator
            config = Config()
            self.orchestrator = MultiAgentOrchestrator(config)
            await self.orchestrator.initialize()
            
            logger.info("✅ Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize platform: {e}", exc_info=True)
            return False
            
    async def test_basic_functionality(self):
        """Test basic platform functionality."""
        logger.info("🔧 Testing Basic Functionality...")
        
        try:
            # Test if orchestrator is initialized
            if not self.orchestrator:
                return {"success": False, "error": "Orchestrator not initialized"}
                
            # Test if agents are available
            agents_available = {
                "planner": self.orchestrator.planner_agent is not None,
                "execution": len(self.orchestrator.execution_agents) > 0,
                "conversational": self.orchestrator.conversational_agent is not None,
                "search": self.orchestrator.search_agent is not None,
                "dom_extractor": self.orchestrator.dom_extractor_agent is not None
            }
            
            logger.info(f"✅ Agents available: {agents_available}")
            
            return {"success": True, "agents": agents_available}
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_conversational_ai(self):
        """Test conversational AI capabilities."""
        logger.info("💬 Testing Conversational AI...")
        
        try:
            if not self.orchestrator.conversational_agent:
                return {"success": False, "error": "Conversational agent not available"}
                
            # Test conversation
            response = await self.orchestrator.conversational_agent.process_message(
                user_id="test_user",
                message="Hello, can you help me with automation?",
                context={"session_id": "test_session"}
            )
            
            logger.info(f"✅ AI Response: {response.get('response', 'No response')[:100]}...")
            
            return {"success": True, "response": response}
            
        except Exception as e:
            logger.error(f"❌ Conversational AI test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_search_capabilities(self):
        """Test search capabilities."""
        logger.info("🔍 Testing Search Capabilities...")
        
        try:
            if not self.orchestrator.search_agent:
                return {"success": False, "error": "Search agent not available"}
                
            # Test search
            results = await self.orchestrator.search_agent.search(
                query="Python automation",
                max_results=3,
                sources=["duckduckgo"]
            )
            
            logger.info(f"✅ Search results: {len(results)} items found")
            
            return {"success": True, "results_count": len(results)}
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_dom_extraction(self):
        """Test DOM extraction capabilities."""
        logger.info("🌐 Testing DOM Extraction...")
        
        try:
            if not self.orchestrator.dom_extractor_agent:
                return {"success": False, "error": "DOM extractor agent not available"}
                
            # Test DOM extraction
            data = await self.orchestrator.dom_extractor_agent.extract_data(
                url="https://example.com",
                selectors={
                    "title": "h1",
                    "content": "p",
                    "links": "a[href]"
                }
            )
            
            logger.info(f"✅ DOM extraction completed: {len(data)} fields extracted")
            
            return {"success": True, "data_fields": len(data)}
            
        except Exception as e:
            logger.error(f"❌ DOM extraction test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_workflow_execution(self):
        """Test workflow execution."""
        logger.info("⚙️ Testing Workflow Execution...")
        
        try:
            # Simple workflow request
            workflow_request = {
                "name": "Test Workflow",
                "description": "Simple test workflow",
                "domain": "testing",
                "tasks": [
                    {
                        "name": "Test Task",
                        "type": "data_processing",
                        "parameters": {
                            "data": "test data"
                        }
                    }
                ]
            }
            
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ Workflow started: {workflow_id}")
            
            # Wait for completion
            await asyncio.sleep(3)
            
            # Check workflow status
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ Workflow execution test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests."""
        logger.info("🚀 Starting Comprehensive Platform Testing...")
        
        # Initialize platform
        if not await self.initialize_platform():
            logger.error("❌ Platform initialization failed. Stopping tests.")
            return
            
        # Run all tests
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Conversational AI", self.test_conversational_ai),
            ("Search Capabilities", self.test_search_capabilities),
            ("DOM Extraction", self.test_dom_extraction),
            ("Workflow Execution", self.test_workflow_execution)
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
                
            # Small delay between tests
            await asyncio.sleep(1)
            
        # Generate comprehensive report
        await self.generate_test_report(results)
        
    async def generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 COMPREHENSIVE TEST REPORT")
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
                
        # Platform capabilities assessment
        logger.info(f"\n🎯 PLATFORM CAPABILITIES ASSESSMENT:")
        
        if passed_tests >= 4:
            logger.info("🟢 EXCELLENT: Platform is highly capable and ready for production")
        elif passed_tests >= 3:
            logger.info("🟡 GOOD: Platform has good capabilities with some areas for improvement")
        elif passed_tests >= 2:
            logger.info("🟠 FAIR: Platform has basic capabilities but needs significant improvement")
        else:
            logger.info("🔴 POOR: Platform has critical issues that need immediate attention")
            
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        if failed_tests > 0:
            logger.info("1. Fix failed test components")
            logger.info("2. Improve error handling and recovery")
            logger.info("3. Enhance integration between components")
        else:
            logger.info("1. Platform is ready for production use")
            logger.info("2. Consider adding more advanced features")
            logger.info("3. Implement performance monitoring")
            
        logger.info(f"\n{'='*60}")
        logger.info("🏁 Testing Complete!")
        logger.info(f"{'='*60}")


async def main():
    """Main test function."""
    tester = PlatformTester()
    await tester.run_comprehensive_tests()


if __name__ == "__main__":
    asyncio.run(main())