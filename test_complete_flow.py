#!/usr/bin/env python3
"""
Complete Flow Test
=================

Test the entire flow from frontend to backend with real automation instructions.
"""

import asyncio
import time
import json
import requests
import aiohttp
from typing import Dict, Any

async def test_complete_flow():
    """Test the complete frontend to backend flow."""
    
    print("🚀 TESTING COMPLETE FRONTEND TO BACKEND FLOW")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Backend Health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Backend Health: {health_data}")
        else:
            print(f"❌ Backend Health Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend Health Error: {e}")
        return False
    
    # Test 2: Frontend Health
    print("\n2️⃣ Testing Frontend Health...")
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend Health: Status {response.status_code}")
        else:
            print(f"❌ Frontend Health Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend Health Error: {e}")
    
    # Test 3: Capabilities Endpoint
    print("\n3️⃣ Testing Capabilities...")
    try:
        response = requests.get("http://localhost:8000/automation/capabilities", timeout=10)
        if response.status_code == 200:
            capabilities = response.json()
            print(f"✅ Capabilities: {len(capabilities.get('capabilities', []))} capabilities available")
        else:
            print(f"❌ Capabilities Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Capabilities Error: {e}")
    
    # Test 4: Simple Automation Test
    print("\n4️⃣ Testing Simple Automation...")
    test_data = {
        "automation_id": "flow_test_001",
        "instructions": "Navigate to Google and search for 'automation testing'",
        "url": "https://www.google.com",
        "generate_report": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/automation/intelligent",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    steps = data.get('steps', [])
                    screenshots = data.get('screenshots', [])
                    
                    print(f"✅ Simple Automation: {status}")
                    print(f"   Steps: {len(steps)}")
                    print(f"   Screenshots: {len(screenshots)}")
                    print(f"   AI Analysis: {data.get('ai_analysis', 'N/A')[:100]}...")
                else:
                    print(f"❌ Simple Automation Failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text[:200]}")
    except Exception as e:
        print(f"❌ Simple Automation Error: {e}")
    
    # Test 5: Chat Endpoint
    print("\n5️⃣ Testing Chat Endpoint...")
    chat_data = {
        "message": "Can you help me automate a complex workflow?",
        "context": {"automation_type": "web_automation"}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/chat",
                json=chat_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '')
                    print(f"✅ Chat Response: {len(response_text)} characters")
                    print(f"   Preview: {response_text[:100]}...")
                else:
                    print(f"❌ Chat Failed: {response.status}")
    except Exception as e:
        print(f"❌ Chat Error: {e}")
    
    # Test 6: Complex Automation Test
    print("\n6️⃣ Testing Complex Automation...")
    complex_test_data = {
        "automation_id": "flow_test_002",
        "instructions": "Navigate to Amazon, search for 'laptop', filter by 4+ stars, add first result to cart, and proceed to checkout",
        "url": "https://www.amazon.com",
        "generate_report": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/automation/intelligent",
                json=complex_test_data,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    steps = data.get('steps', [])
                    screenshots = data.get('screenshots', [])
                    
                    print(f"✅ Complex Automation: {status}")
                    print(f"   Steps: {len(steps)}")
                    print(f"   Screenshots: {len(screenshots)}")
                    
                    # Show step details
                    if steps:
                        print(f"   Step Details:")
                        for i, step in enumerate(steps[:3], 1):  # Show first 3 steps
                            print(f"     {i}. {step.get('action', 'unknown')}: {step.get('description', 'N/A')}")
                        if len(steps) > 3:
                            print(f"     ... and {len(steps) - 3} more steps")
                else:
                    print(f"❌ Complex Automation Failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text[:200]}")
    except Exception as e:
        print(f"❌ Complex Automation Error: {e}")
    
    # Test 7: Report Generation
    print("\n7️⃣ Testing Report Generation...")
    try:
        # First check available formats
        response = requests.get("http://localhost:8000/reports/formats", timeout=10)
        if response.status_code == 200:
            formats = response.json()
            print(f"✅ Report Formats: {formats.get('formats', [])}")
            
            # Test report generation
            report_data = {
                "automation_id": "flow_test_003",
                "instructions": "Test automation workflow",
                "url": "https://www.google.com",
                "status": "completed",
                "steps": [
                    {
                        "step": 1,
                        "action": "navigate",
                        "description": "Navigate to Google",
                        "status": "completed",
                        "duration": 2.5,
                        "screenshot": "screenshot_1.png"
                    }
                ],
                "screenshots": ["screenshot_1.png", "screenshot_2.png"],
                "execution_time": 15.5,
                "success_rate": 0.95
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/reports/generate",
                    json=report_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as report_response:
                    if report_response.status == 200:
                        report_result = await report_response.json()
                        report_paths = report_result.get('report_paths', [])
                        print(f"✅ Report Generation: {len(report_paths)} reports created")
                    else:
                        print(f"❌ Report Generation Failed: {report_response.status}")
        else:
            print(f"❌ Report Formats Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Report Generation Error: {e}")
    
    # Test 8: Browser Control
    print("\n8️⃣ Testing Browser Control...")
    try:
        # Test status
        response = requests.get("http://localhost:8000/automation/status", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Browser Status: {status_data.get('status', 'unknown')}")
            
            # Test close browser
            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/automation/close-browser") as close_response:
                    if close_response.status == 200:
                        close_data = await close_response.json()
                        print(f"✅ Browser Control: {close_data.get('message', 'unknown')}")
                    else:
                        print(f"❌ Browser Close Failed: {close_response.status}")
        else:
            print(f"❌ Browser Status Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Browser Control Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETE FLOW TEST FINISHED!")
    print("=" * 60)
    
    return True

async def main():
    """Main execution."""
    await test_complete_flow()

if __name__ == "__main__":
    asyncio.run(main())