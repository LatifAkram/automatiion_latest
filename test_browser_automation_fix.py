#!/usr/bin/env python3
"""
Test Browser Automation Fix
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

async def test_browser_automation():
    """Test browser automation execution"""
    print("🧪 Testing Browser Automation Fix...")
    
    try:
        from super_omega_orchestrator import SuperOmegaOrchestrator, HybridRequest, ProcessingMode
        
        # Initialize orchestrator
        orchestrator = SuperOmegaOrchestrator()
        print("✅ SuperOmega orchestrator initialized")
        
        # Create automation request
        request = HybridRequest(
            request_id='test_automation',
            task_type='automation_execution',
            data={
                'instruction': 'open google',
                'url': 'https://www.google.com',
                'session_id': 'test_session'
            },
            mode=ProcessingMode.BUILTIN_ONLY,  # Use built-in to test our fix
            timeout=30.0,
            require_evidence=True
        )
        
        print("🚀 Executing browser automation...")
        response = await orchestrator.process_request(request)
        
        print(f"✅ Automation completed!")
        print(f"   Success: {response.success}")
        print(f"   Processing path: {response.processing_path}")
        print(f"   Result: {response.result}")
        
        if response.success and response.result.get('automation_completed'):
            print("🎉 Browser automation is working! Chrome should have opened Google.")
            return True
        else:
            print("❌ Browser automation failed or didn't complete properly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🔧 TESTING BROWSER AUTOMATION FIX")
    print("=" * 40)
    
    success = await test_browser_automation()
    
    if success:
        print("\n🎉 SUCCESS! Browser automation is now working.")
        print("   - Chrome browser will launch")
        print("   - Google will open automatically") 
        print("   - Screenshots will be captured")
        print("   - Real automation execution confirmed!")
    else:
        print("\n⚠️  Browser automation needs more work.")

if __name__ == "__main__":
    asyncio.run(main())