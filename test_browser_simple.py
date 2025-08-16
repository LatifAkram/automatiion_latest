#!/usr/bin/env python3
"""
Simple browser automation test to isolate the hanging issue
"""

import asyncio
import sys
import os
from playwright.async_api import async_playwright

async def simple_browser_test():
    """Test basic browser automation with timeouts"""
    print("🚀 Starting simple browser test...")
    
    try:
        print("1. Creating Playwright instance...")
        async with async_playwright() as p:
            print("✅ Playwright created")
            
            print("2. Launching browser...")
            browser = await asyncio.wait_for(
                p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                ),
                timeout=15.0
            )
            print("✅ Browser launched")
            
            print("3. Creating context...")
            context = await asyncio.wait_for(
                browser.new_context(),
                timeout=10.0
            )
            print("✅ Context created")
            
            print("4. Creating page...")
            page = await asyncio.wait_for(
                context.new_page(),
                timeout=10.0
            )
            print("✅ Page created")
            
            print("5. Navigating to Google...")
            await asyncio.wait_for(
                page.goto('https://www.google.com', wait_until='domcontentloaded'),
                timeout=20.0
            )
            print("✅ Navigation completed")
            
            print("6. Taking screenshot...")
            await page.screenshot(path='test_screenshot.png')
            print("✅ Screenshot taken")
            
            print("7. Closing browser...")
            await browser.close()
            print("✅ Browser closed")
            
            return {'success': True, 'message': 'Browser test completed successfully'}
            
    except asyncio.TimeoutError as e:
        print(f"❌ Timeout error: {e}")
        return {'success': False, 'error': 'Timeout during browser operations'}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🧪 SIMPLE BROWSER AUTOMATION TEST")
    print("=" * 50)
    
    result = asyncio.run(simple_browser_test())
    
    print("\n📊 RESULT:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Message: {result['message']}")
    else:
        print(f"Error: {result['error']}")
    
    print("\n🎯 Test completed!")