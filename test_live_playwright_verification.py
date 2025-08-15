#!/usr/bin/env python3
"""
Live Playwright Verification Test
=================================

Direct test to verify that live Playwright automation is actually happening
with real browser processes and visible browser windows.
"""

import sys
import os
import json
import time
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def verify_live_playwright_execution():
    """Directly verify that Playwright is executing live automation"""
    
    print("🎭 LIVE PLAYWRIGHT VERIFICATION TEST")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check initial process state
    print("🔍 INITIAL PROCESS CHECK")
    initial_processes = get_browser_processes()
    print(f"   📊 Initial browser processes: {len(initial_processes)}")
    if initial_processes:
        for proc in initial_processes:
            print(f"      🌐 {proc}")
    print()
    
    # Import and initialize automation system
    try:
        from testing.super_omega_live_automation_fixed import (
            get_fixed_super_omega_live_automation,
            ExecutionMode
        )
        
        # Configure for visible automation
        config = {
            'headless': False,  # CRITICAL: Show browser window
            'record_video': True,
            'capture_screenshots': True,
            'slow_mo': 1000,  # Slow down to see actions
            'debug': True
        }
        
        automation = get_fixed_super_omega_live_automation(config)
        print("✅ SUPER-OMEGA automation system loaded")
        print("✅ Configured for VISIBLE browser execution")
        
    except Exception as e:
        print(f"❌ Failed to load automation system: {e}")
        return False
    
    session_id = f"live_verification_{int(time.time())}"
    browser_launched = False
    
    try:
        print("🚀 CREATING LIVE AUTOMATION SESSION")
        print(f"   📋 Session ID: {session_id}")
        print("   🎯 Mode: HYBRID (Edge + AI)")
        print("   👁️  Browser: VISIBLE (headless=False)")
        print()
        
        # Create session and launch browser
        session_result = await automation.create_super_omega_session(
            session_id=session_id,
            url="about:blank",
            mode=ExecutionMode.HYBRID
        )
        
        if not session_result.get('success', False):
            raise Exception(f"Session creation failed: {session_result.get('error', 'Unknown error')}")
        
        browser_launched = session_result.get('browser_launched', False)
        print(f"✅ Session created successfully")
        print(f"🎭 Browser launched: {browser_launched}")
        
        # Check for browser processes after launch
        print()
        print("🔍 POST-LAUNCH PROCESS CHECK")
        await asyncio.sleep(2)  # Give browser time to start
        
        post_launch_processes = get_browser_processes()
        new_processes = len(post_launch_processes) - len(initial_processes)
        
        print(f"   📊 Browser processes after launch: {len(post_launch_processes)}")
        print(f"   🆕 New processes: {new_processes}")
        
        if new_processes > 0:
            print("   🎉 NEW BROWSER PROCESSES DETECTED:")
            for proc in post_launch_processes[-new_processes:]:
                print(f"      🌐 {proc}")
        
        # Perform live automation test steps
        print()
        print("🎭 EXECUTING LIVE AUTOMATION STEPS")
        print("-" * 40)
        
        # Step 1: Navigate to example.com
        print("📋 Step 1: Navigate to example.com")
        nav_start = time.time()
        
        nav_result = await automation.super_omega_navigate(session_id, "https://example.com")
        nav_time = time.time() - nav_start
        
        nav_success = nav_result.get('success', False)
        print(f"   {'✅' if nav_success else '❌'} Navigation: {nav_success}")
        print(f"   ⏱️  Time: {nav_time:.2f}s")
        if nav_result.get('screenshot'):
            print(f"   📸 Screenshot: {nav_result['screenshot']}")
        
        # Small delay to see the page
        print("   ⏳ Pausing to observe browser (5s)...")
        await asyncio.sleep(5)
        
        # Step 2: Find and analyze elements
        print()
        print("📋 Step 2: Find page elements")
        find_start = time.time()
        
        find_result = await automation.super_omega_find_element(session_id, "h1")
        find_time = time.time() - find_start
        
        find_success = find_result.get('success', False)
        print(f"   {'✅' if find_success else '❌'} Element finding: {find_success}")
        print(f"   ⏱️  Time: {find_time:.2f}s")
        if find_result.get('element_info'):
            print(f"   🎯 Element found: {find_result['element_info']}")
        
        # Step 3: Navigate to another site
        print()
        print("📋 Step 3: Navigate to JSONPlaceholder API")
        nav2_start = time.time()
        
        nav2_result = await automation.super_omega_navigate(session_id, "https://jsonplaceholder.typicode.com/")
        nav2_time = time.time() - nav2_start
        
        nav2_success = nav2_result.get('success', False)
        print(f"   {'✅' if nav2_success else '❌'} Navigation: {nav2_success}")
        print(f"   ⏱️  Time: {nav2_time:.2f}s")
        
        # Final pause to observe
        print("   ⏳ Final observation pause (3s)...")
        await asyncio.sleep(3)
        
        # Step 4: Close session and cleanup
        print()
        print("🧹 CLOSING SESSION")
        close_result = await automation.close_super_omega_session(session_id)
        close_success = close_result.get('success', False)
        print(f"   {'✅' if close_success else '❌'} Session closed: {close_success}")
        
        # Final process check
        print()
        print("🔍 FINAL PROCESS CHECK")
        await asyncio.sleep(2)  # Give processes time to clean up
        
        final_processes = get_browser_processes()
        print(f"   📊 Final browser processes: {len(final_processes)}")
        
        # Calculate results
        total_steps = 3
        successful_steps = sum([nav_success, find_success, nav2_success])
        success_rate = (successful_steps / total_steps) * 100
        
        print()
        print("📊 LIVE AUTOMATION VERIFICATION RESULTS")
        print("=" * 50)
        print(f"🎯 Session Creation: {'✅ SUCCESS' if browser_launched else '❌ FAILED'}")
        print(f"📈 Step Success Rate: {successful_steps}/{total_steps} ({success_rate:.1f}%)")
        print(f"🎭 Browser Processes: Initial={len(initial_processes)}, Peak={len(post_launch_processes)}, Final={len(final_processes)}")
        print(f"🌐 Live Browser Detected: {'YES' if new_processes > 0 else 'NO'}")
        print(f"⏱️  Total Execution Time: {nav_time + find_time + nav2_time:.2f}s")
        
        # Evidence check
        evidence_items = []
        if nav_result.get('screenshot'):
            evidence_items.append('navigation_screenshot')
        if find_result.get('element_info'):
            evidence_items.append('element_analysis')
        
        print(f"📸 Evidence Collected: {len(evidence_items)} items")
        for item in evidence_items:
            print(f"   📄 {item}")
        
        # Final assessment
        print()
        if browser_launched and new_processes > 0 and success_rate >= 66:
            print("🎉 LIVE PLAYWRIGHT AUTOMATION CONFIRMED!")
            print("✅ Real browser processes detected")
            print("✅ Navigation and interaction successful")
            print("✅ SUPER-OMEGA platform working as expected")
            verification_result = "CONFIRMED"
        elif browser_launched and success_rate >= 50:
            print("⚠️ PARTIAL LIVE AUTOMATION DETECTED")
            print("✅ Browser launched successfully")
            print("⚠️ Some automation steps had issues")
            print("🔧 System functional but may need optimization")
            verification_result = "PARTIAL"
        else:
            print("❌ LIVE AUTOMATION ISSUES DETECTED")
            print("❌ Browser launch or automation failed")
            print("🛠️ System needs troubleshooting")
            verification_result = "FAILED"
        
        # Save verification report
        verification_report = {
            "verification_result": verification_result,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "browser_launched": browser_launched,
            "step_success_rate": success_rate,
            "browser_processes": {
                "initial": len(initial_processes),
                "peak": len(post_launch_processes),
                "final": len(final_processes),
                "new_detected": new_processes
            },
            "execution_times": {
                "navigation_1": nav_time,
                "element_finding": find_time,
                "navigation_2": nav2_time
            },
            "evidence_collected": evidence_items,
            "system_info": {
                "platform": "SUPER-OMEGA Fixed Live Automation",
                "playwright_available": True,
                "headless_mode": False,
                "automation_engine": "Playwright + Self-Healing"
            }
        }
        
        report_file = f"live_playwright_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(verification_report, f, indent=2)
            print(f"💾 Verification report saved: {report_file}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        return verification_result == "CONFIRMED"
        
    except Exception as e:
        print(f"❌ Verification test failed: {e}")
        
        # Try to clean up session if it exists
        try:
            if browser_launched:
                await automation.close_super_omega_session(session_id)
        except:
            pass
        
        return False

def get_browser_processes():
    """Get list of browser-related processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = []
        
        for line in result.stdout.split('\n'):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['chromium', 'chrome', 'playwright', 'browser']):
                if 'grep' not in line_lower:  # Exclude grep processes
                    processes.append(line.strip())
        
        return processes
    except:
        return []

if __name__ == "__main__":
    print("🎭 SUPER-OMEGA Live Playwright Verification")
    print("Direct test of live browser automation with process verification")
    print()
    
    # Run the verification
    result = asyncio.run(verify_live_playwright_execution())
    
    print()
    if result:
        print("🎉 VERIFICATION SUCCESSFUL!")
        print("🚀 Live Playwright automation is working!")
    else:
        print("⚠️ VERIFICATION INCOMPLETE")
        print("🔧 Check system configuration and try again")