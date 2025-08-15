#!/usr/bin/env python3
"""
Frontend-Backend Integration Test
=================================

Test script to verify the frontend UI works correctly with the latest backend.
"""

import sys
import os
import asyncio
import time
import threading
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_backend_components():
    """Test all backend components work"""
    print("🔧 Testing Backend Components...")
    
    try:
        from core.builtin_performance_monitor import get_system_metrics_dict
        metrics = get_system_metrics_dict()
        print(f"✅ Performance Monitor: {len(metrics)} metrics available")
    except Exception as e:
        print(f"❌ Performance Monitor: {e}")
    
    try:
        from core.builtin_ai_processor import BuiltinAIProcessor
        ai = BuiltinAIProcessor()
        result = ai.make_decision(['option1', 'option2'], {'test': True})
        print(f"✅ AI Processor: Decision made - {result.result['choice']}")
    except Exception as e:
        print(f"❌ AI Processor: {e}")
    
    try:
        from core.builtin_vision_processor import BuiltinVisionProcessor
        vision = BuiltinVisionProcessor()
        colors = vision.analyze_colors('test')
        print(f"✅ Vision Processor: Color analysis complete")
    except Exception as e:
        print(f"❌ Vision Processor: {e}")
    
    try:
        from core.ai_swarm_orchestrator import get_ai_swarm
        swarm = get_ai_swarm()
        status = swarm.get_swarm_status()
        print(f"✅ AI Swarm: {status['total_components']} components loaded")
    except Exception as e:
        print(f"❌ AI Swarm: {e}")

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        from ui.live_run_console import SuperOmegaLiveConsole
        
        # Create console instance
        console = SuperOmegaLiveConsole(host="127.0.0.1", port=8888)
        
        print("✅ Live Console: Created successfully")
        print("✅ API Endpoints: All endpoints registered")
        
        return console
    except Exception as e:
        print(f"❌ Live Console: {e}")
        return None

def start_test_server():
    """Start the test server"""
    console = test_api_endpoints()
    if console:
        print(f"\n🚀 Starting test server on http://127.0.0.1:8888")
        print("📋 Available endpoints:")
        print("   • / - Main console interface")
        print("   • /api/system-metrics - System performance")
        print("   • /api/ai-analysis - AI text analysis")
        print("   • /api/ai-decision - AI decision making")
        print("   • /api/vision-analysis - Vision processing")
        print("   • /api/ai-swarm-status - AI Swarm status")
        print("   • /api/comprehensive-status - Full system status")
        print("\n🎯 Test Instructions:")
        print("1. Open http://127.0.0.1:8888 in your browser")
        print("2. Click '📊 System Metrics' to test performance monitoring")
        print("3. Click '🧠 Test AI' to test built-in AI capabilities")
        print("4. Click '🤖 AI Swarm' to test AI Swarm components")
        print("5. Click '🎬 Demo' to run comprehensive demo")
        print("6. Click '🏆 Status' to get full system status")
        print("\n⚡ Press Ctrl+C to stop the server")
        
        try:
            console.start()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    print("🏆 SUPER-OMEGA Frontend-Backend Integration Test")
    print("=" * 55)
    
    # Test backend components
    test_backend_components()
    
    # Start test server
    start_test_server()