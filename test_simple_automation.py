#!/usr/bin/env python3
"""
TEST SIMPLE RELIABLE AUTOMATION
===============================
Test the simple automation system that actually works
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

async def test_simple_automation():
    """Test simple automation"""
    print("🧪 TESTING SIMPLE RELIABLE AUTOMATION")
    print("=" * 50)
    
    try:
        from simple_reliable_automation import execute_simple_reliable_automation
        
        # Test YouTube automation
        print("🎯 Testing YouTube automation...")
        instruction = "open youtube and play trending songs 2025"
        
        result = await execute_simple_reliable_automation(instruction)
        
        print(f"✅ Result: {result}")
        
        if result.get('success'):
            print("🎉 SIMPLE AUTOMATION WORKS!")
            print(f"📊 Actions performed: {result.get('actions_performed', [])}")
            print(f"🌐 URL: {result.get('url', 'N/A')}")
            print(f"📸 Screenshot: {result.get('screenshot', 'N/A')}")
        else:
            print("❌ Simple automation failed")
            print(f"🔍 Error: {result.get('error', 'Unknown error')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_automation())
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if success else 'FAILED'}")