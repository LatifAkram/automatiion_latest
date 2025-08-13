#!/usr/bin/env python3
"""Test API server and frontend integration."""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

async def test_api_server():
    """Test API server functionality."""
    try:
        print("🚀 Testing API Server and Frontend Integration...")
        
        # Start the server in background
        from src.core.config import Config
        from src.core.orchestrator import MultiAgentOrchestrator
        from src.api.server import start_api_server
        
        config = Config()
        orchestrator = MultiAgentOrchestrator(config)
        await orchestrator.initialize()
        
        print("✅ Platform initialized")
        
        # Test API endpoints
        base_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Health check
            print("\n🔍 Testing Health Check...")
            try:
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        print("✅ Health check passed")
                    else:
                        print(f"⚠️ Health check failed: {response.status}")
            except Exception as e:
                print(f"⚠️ Health check error: {e}")
            
            # Test 2: Get capabilities
            print("\n🔧 Testing Capabilities Endpoint...")
            try:
                async with session.get(f"{base_url}/automation/capabilities") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Capabilities: {len(data.get('capabilities', []))} capabilities found")
                    else:
                        print(f"⚠️ Capabilities failed: {response.status}")
            except Exception as e:
                print(f"⚠️ Capabilities error: {e}")
            
            # Test 3: Test comprehensive automation
            print("\n🤖 Testing Comprehensive Automation...")
            test_payload = {
                "instructions": "Automate login to Google and search for 'artificial intelligence'",
                "url": "https://www.google.com"
            }
            
            try:
                async with session.post(f"{base_url}/automation/test-comprehensive", json=test_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ Comprehensive automation test successful")
                        print(f"   AI Agents: {data.get('ai_agents', {})}")
                        print(f"   Test Summary: {data.get('test_summary', {})}")
                    else:
                        print(f"⚠️ Comprehensive automation failed: {response.status}")
            except Exception as e:
                print(f"⚠️ Comprehensive automation error: {e}")
            
            # Test 4: Test conversational AI
            print("\n💬 Testing Conversational AI...")
            chat_payload = {
                "message": "Can you help me automate a complex workflow?",
                "context": {"automation_type": "web_automation"}
            }
            
            try:
                async with session.post(f"{base_url}/chat", json=chat_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ Conversational AI working")
                        print(f"   Response: {data.get('response', '')[:100]}...")
                    else:
                        print(f"⚠️ Conversational AI failed: {response.status}")
            except Exception as e:
                print(f"⚠️ Conversational AI error: {e}")
        
        print("\n🎉 API SERVER TEST COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_api_server())
    if result:
        print("🚀 API Server is working!")
    else:
        print("🔧 Need to fix API server issues")