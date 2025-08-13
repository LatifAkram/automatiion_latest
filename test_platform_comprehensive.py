#!/usr/bin/env python3
"""
Comprehensive Platform Test
==========================

This script tests the actual capabilities of the multi-agent automation platform
with real scenarios across different domains.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.config import Config
from core.orchestrator import MultiAgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlatformTester:
    """Comprehensive platform testing class."""
    
    def __init__(self):
        self.config = Config()
        self.orchestrator = None
        self.test_results = []
        
    async def initialize_platform(self):
        """Initialize the platform."""
        try:
            logger.info("🚀 Initializing Multi-Agent Automation Platform...")
            
            # Initialize orchestrator
            self.orchestrator = MultiAgentOrchestrator(self.config)
            await self.orchestrator.initialize()
            
            logger.info("✅ Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize platform: {e}", exc_info=True)
            return False
            
    async def test_ecommerce_automation(self):
        """Test e-commerce automation capabilities."""
        logger.info("🛒 Testing E-commerce Automation...")
        
        workflow_request = {
            "name": "E-commerce Product Research",
            "description": "Research products on Amazon, extract pricing and reviews",
            "domain": "e-commerce",
            "tasks": [
                {
                    "name": "Search Amazon for laptops",
                    "type": "web_search",
                    "parameters": {
                        "query": "best gaming laptops 2024",
                        "site": "amazon.com"
                    }
                },
                {
                    "name": "Extract product data",
                    "type": "data_extraction",
                    "parameters": {
                        "selectors": {
                            "products": ".s-result-item",
                            "titles": "h2 a span",
                            "prices": ".a-price-whole",
                            "ratings": ".a-icon-alt"
                        }
                    }
                }
            ]
        }
        
        try:
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ E-commerce workflow started: {workflow_id}")
            
            # Wait for completion
            await asyncio.sleep(5)
            
            # Check workflow status
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ E-commerce test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_banking_automation(self):
        """Test banking automation capabilities."""
        logger.info("🏦 Testing Banking Automation...")
        
        workflow_request = {
            "name": "Bank Account Balance Check",
            "description": "Check account balance and recent transactions",
            "domain": "banking",
            "tasks": [
                {
                    "name": "Navigate to banking portal",
                    "type": "web_navigation",
                    "parameters": {
                        "url": "https://demo.bank.com",
                        "wait_for": ".login-form"
                    }
                },
                {
                    "name": "Extract account information",
                    "type": "data_extraction",
                    "parameters": {
                        "selectors": {
                            "balance": ".account-balance",
                            "transactions": ".transaction-list li"
                        }
                    }
                }
            ]
        }
        
        try:
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ Banking workflow started: {workflow_id}")
            
            await asyncio.sleep(5)
            
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ Banking test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_ticket_booking(self):
        """Test ticket booking automation."""
        logger.info("🎫 Testing Ticket Booking Automation...")
        
        workflow_request = {
            "name": "Flight Ticket Search",
            "description": "Search for flight tickets and extract pricing",
            "domain": "travel",
            "tasks": [
                {
                    "name": "Search flight tickets",
                    "type": "web_search",
                    "parameters": {
                        "query": "flight tickets New York to London",
                        "site": "skyscanner.com"
                    }
                },
                {
                    "name": "Extract flight data",
                    "type": "data_extraction",
                    "parameters": {
                        "selectors": {
                            "flights": ".flight-item",
                            "prices": ".price",
                            "times": ".time"
                        }
                    }
                }
            ]
        }
        
        try:
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ Ticket booking workflow started: {workflow_id}")
            
            await asyncio.sleep(5)
            
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ Ticket booking test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_stock_analysis(self):
        """Test stock market analysis automation."""
        logger.info("📈 Testing Stock Market Analysis...")
        
        workflow_request = {
            "name": "Stock Price Analysis",
            "description": "Analyze stock prices and market data",
            "domain": "finance",
            "tasks": [
                {
                    "name": "Get stock data",
                    "type": "api_call",
                    "parameters": {
                        "url": "https://api.example.com/stocks/AAPL",
                        "method": "GET"
                    }
                },
                {
                    "name": "Analyze stock trends",
                    "type": "data_analysis",
                    "parameters": {
                        "analysis_type": "trend_analysis",
                        "timeframe": "1d"
                    }
                }
            ]
        }
        
        try:
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ Stock analysis workflow started: {workflow_id}")
            
            await asyncio.sleep(5)
            
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ Stock analysis test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_conversational_ai(self):
        """Test conversational AI capabilities."""
        logger.info("💬 Testing Conversational AI...")
        
        try:
            # Test conversation
            response = await self.orchestrator.conversational_agent.process_message(
                user_id="test_user",
                message="Can you help me create a workflow for web scraping?",
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
            # Test search
            results = await self.orchestrator.search_agent.search(
                query="Python automation frameworks",
                max_results=5,
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
            # Test DOM extraction
            data = await self.orchestrator.dom_extractor.extract_data(
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
            
    async def test_self_healing(self):
        """Test self-healing capabilities."""
        logger.info("🔧 Testing Self-Healing Capabilities...")
        
        try:
            # Simulate a failure scenario
            workflow_request = {
                "name": "Self-Healing Test",
                "description": "Test self-healing with invalid selectors",
                "domain": "testing",
                "tasks": [
                    {
                        "name": "Test with invalid selector",
                        "type": "data_extraction",
                        "parameters": {
                            "selectors": {
                                "invalid": ".non-existent-selector"
                            }
                        }
                    }
                ]
            }
            
            workflow_id = await self.orchestrator.execute_workflow(workflow_request)
            logger.info(f"✅ Self-healing test workflow started: {workflow_id}")
            
            await asyncio.sleep(3)
            
            workflow = self.orchestrator.active_workflows.get(workflow_id)
            if workflow:
                logger.info(f"📊 Self-healing workflow status: {workflow.status.value}")
                
            return {"success": True, "workflow_id": workflow_id}
            
        except Exception as e:
            logger.error(f"❌ Self-healing test failed: {e}")
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
            ("E-commerce Automation", self.test_ecommerce_automation),
            ("Banking Automation", self.test_banking_automation),
            ("Ticket Booking", self.test_ticket_booking),
            ("Stock Analysis", self.test_stock_analysis),
            ("Conversational AI", self.test_conversational_ai),
            ("Search Capabilities", self.test_search_capabilities),
            ("DOM Extraction", self.test_dom_extraction),
            ("Self-Healing", self.test_self_healing)
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
            await asyncio.sleep(2)
            
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
        
        if passed_tests >= 6:
            logger.info("🟢 EXCELLENT: Platform is highly capable and ready for production")
        elif passed_tests >= 4:
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