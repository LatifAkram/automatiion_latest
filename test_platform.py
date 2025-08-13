#!/usr/bin/env python3
"""
Comprehensive Platform Test
==========================

This script demonstrates the full capabilities of the Autonomous Multi-Agent
Automation Platform, including complex workflows, search, extraction, and automation.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

# Import platform components
from src.core.orchestrator import MultiAgentOrchestrator
from src.core.config import Config
from src.utils.logger import setup_logging


async def test_basic_workflow():
    """Test basic workflow execution."""
    print("\n" + "="*60)
    print("TEST 1: Basic E-commerce Workflow")
    print("="*60)
    
    workflow_request = {
        "name": "E-commerce Product Research",
        "description": "Research product information from multiple e-commerce sites",
        "domain": "ecommerce",
        "tasks": [
            {
                "name": "Search for product",
                "type": "search",
                "parameters": {
                    "query": "wireless headphones best 2024",
                    "sources": ["google", "bing", "duckduckgo"]
                }
            },
            {
                "name": "Extract product data",
                "type": "dom_extraction",
                "parameters": {
                    "urls": ["https://www.amazon.com", "https://www.bestbuy.com"],
                    "selectors": {
                        "product_title": "h1.product-title, .product-name h1",
                        "product_price": ".price, [class*='price']",
                        "product_rating": ".rating, [class*='rating']"
                    }
                }
            },
            {
                "name": "Process and analyze data",
                "type": "data_processing",
                "parameters": {
                    "operations": [
                        {"type": "filter", "field": "price", "condition": "less_than", "value": 200},
                        {"type": "sort", "field": "rating", "direction": "desc"},
                        {"type": "aggregate", "field": "price", "aggregate_type": "average"}
                    ]
                }
            }
        ]
    }
    
    return workflow_request


async def test_complex_automation():
    """Test complex web automation workflow."""
    print("\n" + "="*60)
    print("TEST 2: Complex Web Automation")
    print("="*60)
    
    workflow_request = {
        "name": "News Article Automation",
        "description": "Automate news article reading and analysis",
        "domain": "news",
        "tasks": [
            {
                "name": "Navigate to news site",
                "type": "web_automation",
                "parameters": {
                    "url": "https://news.ycombinator.com",
                    "actions": [
                        {"type": "navigate", "url": "https://news.ycombinator.com"},
                        {"type": "wait", "selector": ".athing", "timeout": 5000},
                        {"type": "screenshot", "name": "homepage"}
                    ]
                }
            },
            {
                "name": "Extract article links",
                "type": "dom_extraction",
                "parameters": {
                    "url": "https://news.ycombinator.com",
                    "selectors": {
                        "article_links": "a.storylink",
                        "article_titles": ".storylink",
                        "article_scores": ".score"
                    }
                }
            },
            {
                "name": "Read top articles",
                "type": "web_automation",
                "parameters": {
                    "actions": [
                        {"type": "click", "selector": "a.storylink:first-child"},
                        {"type": "wait", "selector": "body", "timeout": 3000},
                        {"type": "scroll", "direction": "down", "amount": 1000},
                        {"type": "screenshot", "name": "article_page"},
                        {"type": "extract", "selectors": {
                            "article_title": "h1, .title",
                            "article_content": ".content, article"
                        }}
                    ]
                }
            }
        ]
    }
    
    return workflow_request


async def test_search_and_analysis():
    """Test comprehensive search and analysis workflow."""
    print("\n" + "="*60)
    print("TEST 3: Search and Analysis Workflow")
    print("="*60)
    
    workflow_request = {
        "name": "Technology Research",
        "description": "Comprehensive technology research and analysis",
        "domain": "technology",
        "tasks": [
            {
                "name": "Search for AI trends",
                "type": "search",
                "parameters": {
                    "query": "artificial intelligence trends 2024",
                    "sources": ["google", "bing", "github", "stackoverflow", "reddit"]
                }
            },
            {
                "name": "Extract GitHub repositories",
                "type": "search",
                "parameters": {
                    "query": "machine learning python",
                    "sources": ["github"]
                }
            },
            {
                "name": "Analyze Stack Overflow questions",
                "type": "search",
                "parameters": {
                    "query": "python automation",
                    "sources": ["stackoverflow"]
                }
            },
            {
                "name": "Process search results",
                "type": "data_processing",
                "parameters": {
                    "operations": [
                        {"type": "filter", "field": "source", "condition": "in", "value": ["github", "stackoverflow"]},
                        {"type": "sort", "field": "timestamp", "direction": "desc"},
                        {"type": "aggregate", "field": "score", "aggregate_type": "average"}
                    ]
                }
            }
        ]
    }
    
    return workflow_request


async def test_api_integration():
    """Test API integration workflow."""
    print("\n" + "="*60)
    print("TEST 4: API Integration Workflow")
    print("="*60)
    
    workflow_request = {
        "name": "Weather Data Analysis",
        "description": "Fetch and analyze weather data from APIs",
        "domain": "data_analysis",
        "tasks": [
            {
                "name": "Fetch weather data",
                "type": "api_call",
                "parameters": {
                    "method": "GET",
                    "url": "https://api.openweathermap.org/data/2.5/weather",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "q": "London,UK",
                        "appid": "demo_key",
                        "units": "metric"
                    }
                }
            },
            {
                "name": "Process weather data",
                "type": "data_processing",
                "parameters": {
                    "operations": [
                        {"type": "transform", "field": "temperature", "transform_type": "convert_to_fahrenheit"},
                        {"type": "aggregate", "field": "humidity", "aggregate_type": "average"}
                    ]
                }
            },
            {
                "name": "Save results",
                "type": "file_operation",
                "parameters": {
                    "operation": "write",
                    "file_path": "weather_data.json",
                    "data": {"processed_weather": "data"}
                }
            }
        ]
    }
    
    return workflow_request


async def test_conversational_agent():
    """Test conversational agent capabilities."""
    print("\n" + "="*60)
    print("TEST 5: Conversational Agent")
    print("="*60)
    
    # Test conversation
    messages = [
        "Hello! I need help with web automation.",
        "Can you explain how to scrape data from a website?",
        "What are the best practices for handling dynamic content?",
        "How can I make my automation more robust?",
        "Can you help me debug a selector that's not working?"
    ]
    
    return messages


async def run_comprehensive_test():
    """Run comprehensive platform test."""
    print("🚀 Starting Comprehensive Platform Test")
    print("="*80)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize platform
        print("📋 Initializing platform...")
        config = Config()
        orchestrator = MultiAgentOrchestrator(config)
        await orchestrator.initialize()
        print("✅ Platform initialized successfully")
        
        # Test 1: Basic Workflow
        workflow1 = await test_basic_workflow()
        print(f"\n🔄 Executing workflow: {workflow1['name']}")
        
        result1 = await orchestrator.execute_workflow(workflow1)
        print(f"✅ Workflow 1 completed: {result1.success}")
        if result1.success:
            print(f"   Duration: {result1.duration:.2f}s")
            print(f"   Steps: {len(result1.steps)}")
            print(f"   Errors: {len(result1.errors)}")
        
        # Test 2: Complex Automation
        workflow2 = await test_complex_automation()
        print(f"\n🔄 Executing workflow: {workflow2['name']}")
        
        result2 = await orchestrator.execute_workflow(workflow2)
        print(f"✅ Workflow 2 completed: {result2.success}")
        if result2.success:
            print(f"   Duration: {result2.duration:.2f}s")
            print(f"   Steps: {len(result2.steps)}")
            print(f"   Errors: {len(result2.errors)}")
        
        # Test 3: Search and Analysis
        workflow3 = await test_search_and_analysis()
        print(f"\n🔄 Executing workflow: {workflow3['name']}")
        
        result3 = await orchestrator.execute_workflow(workflow3)
        print(f"✅ Workflow 3 completed: {result3.success}")
        if result3.success:
            print(f"   Duration: {result3.duration:.2f}s")
            print(f"   Steps: {len(result3.steps)}")
            print(f"   Errors: {len(result3.errors)}")
        
        # Test 4: API Integration
        workflow4 = await test_api_integration()
        print(f"\n🔄 Executing workflow: {workflow4['name']}")
        
        result4 = await orchestrator.execute_workflow(workflow4)
        print(f"✅ Workflow 4 completed: {result4.success}")
        if result4.success:
            print(f"   Duration: {result4.duration:.2f}s")
            print(f"   Steps: {len(result4.steps)}")
            print(f"   Errors: {len(result4.errors)}")
        
        # Test 5: Conversational Agent
        messages = await test_conversational_agent()
        print(f"\n💬 Testing conversational agent with {len(messages)} messages")
        
        for i, message in enumerate(messages, 1):
            print(f"\n   Message {i}: {message[:50]}...")
            response = await orchestrator.chat_with_agent(message)
            print(f"   Response: {response.response[:100]}...")
            print(f"   Confidence: {response.confidence:.2f}")
        
        # Get platform statistics
        print("\n" + "="*60)
        print("📊 PLATFORM STATISTICS")
        print("="*60)
        
        # Get agent status
        agent_status = orchestrator.get_agent_status()
        print(f"🤖 Agent Status:")
        for agent_name, status in agent_status.items():
            print(f"   {agent_name}: {'🟢 Active' if status.get('is_busy', False) else '🟡 Idle'}")
        
        # Get performance metrics
        performance = orchestrator.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Workflows: {performance.get('total_workflows', 0)}")
        print(f"   Successful Workflows: {performance.get('successful_workflows', 0)}")
        print(f"   Average Duration: {performance.get('average_duration', 0):.2f}s")
        print(f"   Success Rate: {performance.get('success_rate', 0):.1%}")
        
        # Get AI provider metrics
        ai_metrics = orchestrator.ai_provider.get_performance_metrics()
        print(f"\n🧠 AI Provider Metrics:")
        print(f"   Best Provider: {ai_metrics.get('best_provider', 'Unknown')}")
        print(f"   Available Providers: {len(ai_metrics.get('available_providers', []))}")
        print(f"   Fallback Count: {ai_metrics.get('fallback_count', 0)}")
        print(f"   Cache Size: {ai_metrics.get('cache_size', 0)}")
        
        # Get vector store statistics
        vector_stats = orchestrator.vector_store.get_statistics()
        print(f"\n🗄️ Vector Store Statistics:")
        for collection, count in vector_stats.items():
            if isinstance(count, int):
                print(f"   {collection}: {count} documents")
        
        # Get audit statistics
        audit_stats = orchestrator.audit_logger.get_audit_statistics()
        print(f"\n📋 Audit Statistics:")
        print(f"   Total Events: {audit_stats.get('total_events', 0)}")
        print(f"   PII Events: {audit_stats.get('pii_events', 0)}")
        print(f"   Recent Events (24h): {audit_stats.get('recent_events_24h', 0)}")
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Generate test report
        test_report = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "platform_version": "1.0.0",
            "test_results": {
                "workflow_1": {
                    "name": workflow1["name"],
                    "success": result1.success,
                    "duration": result1.duration,
                    "steps": len(result1.steps),
                    "errors": len(result1.errors)
                },
                "workflow_2": {
                    "name": workflow2["name"],
                    "success": result2.success,
                    "duration": result2.duration,
                    "steps": len(result2.steps),
                    "errors": len(result2.errors)
                },
                "workflow_3": {
                    "name": workflow3["name"],
                    "success": result3.success,
                    "duration": result3.duration,
                    "steps": len(result3.steps),
                    "errors": len(result3.errors)
                },
                "workflow_4": {
                    "name": workflow4["name"],
                    "success": result4.success,
                    "duration": result4.duration,
                    "steps": len(result4.steps),
                    "errors": len(result4.errors)
                }
            },
            "platform_statistics": {
                "agent_status": agent_status,
                "performance_metrics": performance,
                "ai_metrics": ai_metrics,
                "vector_stats": vector_stats,
                "audit_stats": audit_stats
            }
        }
        
        # Save test report
        report_path = Path("test_report.json")
        with open(report_path, "w") as f:
            json.dump(test_report, f, indent=2)
        
        print(f"📄 Test report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")
        
    finally:
        # Shutdown platform
        print("\n🔄 Shutting down platform...")
        await orchestrator.shutdown()
        print("✅ Platform shutdown complete")


async def test_specific_capability(capability: str):
    """Test a specific platform capability."""
    print(f"\n🧪 Testing Specific Capability: {capability}")
    print("="*60)
    
    setup_logging()
    config = Config()
    orchestrator = MultiAgentOrchestrator(config)
    
    try:
        await orchestrator.initialize()
        
        if capability == "search":
            # Test search capabilities
            search_results = await orchestrator.search_agent.comprehensive_search(
                "artificial intelligence trends 2024",
                sources=["google", "bing", "duckduckgo"]
            )
            print(f"✅ Search completed: {sum(len(r) for r in search_results.values())} total results")
            
        elif capability == "extraction":
            # Test extraction capabilities
            extraction_result = await orchestrator.dom_extractor.extract_from_url(
                "https://example.com",
                content_type="general"
            )
            print(f"✅ Extraction completed: {len(extraction_result)} fields extracted")
            
        elif capability == "automation":
            # Test automation capabilities
            automation_task = {
                "name": "Simple Navigation Test",
                "type": "web_automation",
                "parameters": {
                    "url": "https://example.com",
                    "actions": [
                        {"type": "navigate", "url": "https://example.com"},
                        {"type": "screenshot", "name": "test_screenshot"}
                    ]
                }
            }
            result = await orchestrator.execution_agent.execute_task(automation_task)
            print(f"✅ Automation completed: {result.get('success', False)}")
            
        elif capability == "conversation":
            # Test conversation capabilities
            response = await orchestrator.chat_with_agent("Hello! How can you help me with automation?")
            print(f"✅ Conversation completed: {response.confidence:.2f} confidence")
            
        else:
            print(f"❌ Unknown capability: {capability}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific capability
        capability = sys.argv[1]
        asyncio.run(test_specific_capability(capability))
    else:
        # Run comprehensive test
        asyncio.run(run_comprehensive_test())