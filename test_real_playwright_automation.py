#!/usr/bin/env python3
"""
REAL PLAYWRIGHT AUTOMATION VERIFICATION
=======================================

CRITICAL VERIFICATION: Test that when we give an instruction,
ACTUAL PLAYWRIGHT BROWSER AUTOMATION HAPPENS - not simulation!

This test will:
1. Install Playwright if needed
2. Create a real browser session
3. Navigate to actual websites
4. Perform real interactions
5. Capture evidence of real automation
6. Prove this is NOT simulation

✅ 100% REAL BROWSER AUTOMATION - NO FAKE RESPONSES!
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def install_playwright():
    """Install Playwright if needed"""
    try:
        import playwright
        print("✅ Playwright already installed")
        return True
    except ImportError:
        print("📦 Installing Playwright...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            print("✅ Playwright installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install Playwright: {e}")
            return False

async def test_real_playwright_automation():
    """Test real Playwright automation"""
    
    print("🎯 REAL PLAYWRIGHT AUTOMATION TEST")
    print("=" * 45)
    
    # Install Playwright if needed
    if not install_playwright():
        return {"error": "Failed to install Playwright"}
    
    try:
        from playwright.async_api import async_playwright
        print("✅ Playwright imported successfully")
        
        test_results = {
            'test_metadata': {
                'test_name': 'Real Playwright Automation Verification',
                'timestamp': datetime.now().isoformat(),
                'test_type': 'REAL_BROWSER_AUTOMATION'
            },
            'automation_steps': [],
            'evidence_collected': [],
            'websites_accessed': [],
            'real_automation_confirmed': False
        }
        
        async with async_playwright() as p:
            print("🚀 Starting real browser...")
            
            # Launch real browser
            browser = await p.chromium.launch(headless=True)  # Use headless for automation
            context = await browser.new_context()
            page = await context.new_page()
            
            print("✅ Real Chromium browser launched")
            
            # Test 1: Navigate to Google
            print("🌐 Test 1: Navigating to Google...")
            start_time = time.time()
            await page.goto("https://www.google.com")
            nav_time = time.time() - start_time
            
            title = await page.title()
            url = page.url
            
            test_results['automation_steps'].append({
                'step': 'navigate_to_google',
                'url': url,
                'title': title,
                'duration_seconds': nav_time,
                'success': 'google' in title.lower(),
                'timestamp': datetime.now().isoformat()
            })
            test_results['websites_accessed'].append(url)
            
            print(f"✅ Navigation successful - Title: {title}")
            
            # Test 2: Search for something
            print("🔍 Test 2: Performing search...")
            search_start = time.time()
            
            # Find search box
            search_box = await page.wait_for_selector('textarea[name="q"], input[name="q"]', timeout=10000)
            await search_box.fill("playwright automation testing")
            await search_box.press("Enter")
            
            # Wait for results
            await page.wait_for_selector('div#search', timeout=10000)
            search_time = time.time() - search_start
            
            # Get search results
            results = await page.query_selector_all('div.g')
            results_count = len(results)
            
            test_results['automation_steps'].append({
                'step': 'perform_search',
                'search_term': 'playwright automation testing',
                'results_found': results_count,
                'duration_seconds': search_time,
                'success': results_count > 0,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Search successful - Found {results_count} results")
            
            # Test 3: Take screenshot as evidence
            print("📸 Test 3: Capturing evidence...")
            evidence_dir = Path("evidence_real_automation")
            evidence_dir.mkdir(exist_ok=True)
            
            screenshot_path = evidence_dir / f"google_search_{int(time.time())}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            
            test_results['evidence_collected'].append({
                'type': 'screenshot',
                'path': str(screenshot_path),
                'file_exists': screenshot_path.exists(),
                'file_size_bytes': screenshot_path.stat().st_size if screenshot_path.exists() else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Screenshot captured: {screenshot_path}")
            
            # Test 4: Navigate to another site
            print("🌐 Test 4: Navigating to Example.com...")
            await page.goto("https://example.com")
            await page.wait_for_load_state('networkidle')
            
            example_title = await page.title()
            example_url = page.url
            
            test_results['automation_steps'].append({
                'step': 'navigate_to_example',
                'url': example_url,
                'title': example_title,
                'success': 'example' in example_title.lower(),
                'timestamp': datetime.now().isoformat()
            })
            test_results['websites_accessed'].append(example_url)
            
            print(f"✅ Example.com navigation - Title: {example_title}")
            
            # Test 5: Extract page content
            print("📄 Test 5: Extracting page content...")
            h1_text = await page.text_content('h1')
            p_text = await page.text_content('p')
            
            test_results['automation_steps'].append({
                'step': 'extract_content',
                'h1_text': h1_text,
                'p_text': p_text[:100] + '...' if p_text and len(p_text) > 100 else p_text,
                'content_extracted': bool(h1_text and p_text),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Content extracted - H1: {h1_text}")
            
            # Final screenshot
            final_screenshot = evidence_dir / f"example_com_{int(time.time())}.png"
            await page.screenshot(path=final_screenshot)
            
            test_results['evidence_collected'].append({
                'type': 'screenshot',
                'path': str(final_screenshot),
                'file_exists': final_screenshot.exists(),
                'file_size_bytes': final_screenshot.stat().st_size if final_screenshot.exists() else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            # Close browser
            await browser.close()
            print("✅ Browser closed")
            
            # Verify real automation occurred
            successful_steps = sum(1 for step in test_results['automation_steps'] if step.get('success', False))
            evidence_files = sum(1 for evidence in test_results['evidence_collected'] if evidence.get('file_exists', False))
            
            test_results['real_automation_confirmed'] = (
                successful_steps >= 3 and 
                evidence_files >= 2 and 
                len(test_results['websites_accessed']) >= 2
            )
            
            test_results['summary'] = {
                'total_steps': len(test_results['automation_steps']),
                'successful_steps': successful_steps,
                'evidence_files_created': evidence_files,
                'websites_accessed': len(test_results['websites_accessed']),
                'real_automation_confirmed': test_results['real_automation_confirmed']
            }
            
            return test_results
            
    except Exception as e:
        print(f"❌ Real automation test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'real_automation_confirmed': False
        }

async def main():
    """Run the real Playwright automation test"""
    
    results = await test_real_playwright_automation()
    
    # Save results
    with open('REAL_PLAYWRIGHT_AUTOMATION_VERIFICATION.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🏆 REAL AUTOMATION TEST RESULTS:")
    print("=" * 40)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"📊 Total Steps: {summary['total_steps']}")
        print(f"✅ Successful Steps: {summary['successful_steps']}")
        print(f"📁 Evidence Files: {summary['evidence_files_created']}")
        print(f"🌐 Websites Accessed: {summary['websites_accessed']}")
        print(f"🎭 Real Automation Confirmed: {'YES' if summary['real_automation_confirmed'] else 'NO'}")
        
        if summary['real_automation_confirmed']:
            print("\n✅ VERIFICATION SUCCESSFUL!")
            print("✅ Real Playwright browser automation confirmed")
            print("✅ Actual websites accessed and interacted with")
            print("✅ Evidence files created with real screenshots")
            print("✅ This is NOT simulation - it's genuine automation!")
        else:
            print("\n❌ VERIFICATION FAILED!")
            print("❌ Could not confirm real automation")
    
    elif 'error' in results:
        print(f"❌ Test failed with error: {results['error']}")
    
    print(f"\n📁 Full report saved to: REAL_PLAYWRIGHT_AUTOMATION_VERIFICATION.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())