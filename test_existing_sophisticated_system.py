#!/usr/bin/env python3
"""
Test Existing Sophisticated System Integration
==============================================

Test the existing sophisticated automation systems to see what's working.
"""

import sys
from pathlib import Path

# Add paths for existing modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src' / 'core'))
sys.path.insert(0, str(current_dir / 'src' / 'platforms'))
sys.path.insert(0, str(current_dir / 'src' / 'testing'))
sys.path.insert(0, str(current_dir / 'src' / 'agents'))

def test_existing_systems():
    """Test what existing sophisticated systems are available"""
    
    print("🔍 TESTING EXISTING SOPHISTICATED SYSTEMS")
    print("=" * 60)
    
    # Test 1: Real-time Data Fabric AI
    try:
        from realtime_data_fabric_ai import RealTimeDataFabricAI
        data_fabric = RealTimeDataFabricAI()
        print("✅ RealTimeDataFabricAI - LOADED")
        print(f"   📊 Methods: {[m for m in dir(data_fabric) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ RealTimeDataFabricAI - FAILED: {e}")
    
    # Test 2: Comprehensive Automation Engine
    try:
        from comprehensive_automation_engine import ComprehensiveAutomationEngine
        automation_engine = ComprehensiveAutomationEngine()
        print("✅ ComprehensiveAutomationEngine - LOADED")
        print(f"   📊 Methods: {[m for m in dir(automation_engine) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ ComprehensiveAutomationEngine - FAILED: {e}")
    
    # Test 3: Commercial Platform Registry
    try:
        from commercial_platform_registry import CommercialPlatformRegistry
        platform_registry = CommercialPlatformRegistry()
        print("✅ CommercialPlatformRegistry - LOADED")
        print(f"   📊 Methods: {[m for m in dir(platform_registry) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ CommercialPlatformRegistry - FAILED: {e}")
    
    # Test 4: Super Omega Live Automation
    try:
        from super_omega_live_automation_fixed import SuperOmegaLiveAutomation
        live_automation = SuperOmegaLiveAutomation()
        print("✅ SuperOmegaLiveAutomation - LOADED")
        print(f"   📊 Methods: {[m for m in dir(live_automation) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ SuperOmegaLiveAutomation - FAILED: {e}")
    
    # Test 5: Advanced Automation Capabilities
    try:
        from advanced_automation_capabilities import AdvancedAutomationCapabilities
        advanced_automation = AdvancedAutomationCapabilities()
        print("✅ AdvancedAutomationCapabilities - LOADED")
        print(f"   📊 Methods: {[m for m in dir(advanced_automation) if not m.startswith('_')]}")
    except Exception as e:
        print(f"❌ AdvancedAutomationCapabilities - FAILED: {e}")
    
    print("\n🎯 TESTING AI PROVIDERS")
    print("=" * 40)
    
    # Test Gemini API
    try:
        import requests
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyBb-AFGtxM2biSnESY85nyk-fdR74O153c"
        payload = {
            "contents": [{"parts": [{"text": "Test: What is 2+2?"}]}]
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Gemini API - WORKING")
        else:
            print(f"❌ Gemini API - HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Gemini API - FAILED: {e}")
    
    # Test Local LLM
    try:
        import requests
        payload = {
            "model": "qwen2-vl-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test: What is 2+2?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }
        response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Local LLM (Vision) - WORKING")
        else:
            print(f"❌ Local LLM - HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Local LLM - FAILED: {e}")
    
    print("\n📋 SUMMARY:")
    print("   The test shows which existing sophisticated systems are available")
    print("   and which AI providers are working for real-time data processing.")

if __name__ == "__main__":
    test_existing_systems()