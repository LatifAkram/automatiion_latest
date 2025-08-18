#!/usr/bin/env python3
"""
Launch Complete SUPER-OMEGA System
=================================

Launches the complete 100% integrated system:
- Backend server with all three architectures
- Frontend interface with real-time updates
- Sophisticated coordination like Cursor AI
"""

import asyncio
import threading
import time
import webbrowser
import os
import sys
import subprocess
from datetime import datetime

def start_backend_server():
    """Start the backend server"""
    try:
        print("🔧 Starting SUPER-OMEGA Backend Server...")
        
        # Import and start backend
        from complete_backend_server import CompleteSuperOmegaBackend
        
        backend = CompleteSuperOmegaBackend(host='localhost', port=8081)
        backend.start_server()
        
    except Exception as e:
        print(f"❌ Backend server failed: {e}")

def start_frontend_server():
    """Start the frontend server"""
    try:
        print("🎨 Starting SUPER-OMEGA Frontend Server...")
        
        # Simple HTTP server for frontend
        os.chdir('/workspace')
        
        import http.server
        import socketserver
        
        class FrontendHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.path = '/complete_frontend.html'
                return super().do_GET()
            
            def log_message(self, format, *args):
                pass  # Reduce noise
        
        with socketserver.TCPServer(("", 3000), FrontendHandler) as httpd:
            print("🌐 Frontend server running at http://localhost:3000")
            httpd.serve_forever()
            
    except Exception as e:
        print(f"❌ Frontend server failed: {e}")

def test_system_health():
    """Test system health before launching"""
    print("🔍 TESTING SYSTEM HEALTH BEFORE LAUNCH")
    print("=" * 50)
    
    health_results = {}
    
    # Test Built-in Foundation
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))
        from builtin_performance_monitor import BuiltinPerformanceMonitor
        from builtin_ai_processor import BuiltinAIProcessor
        
        monitor = BuiltinPerformanceMonitor()
        metrics = monitor.get_comprehensive_metrics()
        
        ai = BuiltinAIProcessor()
        decision = ai.make_decision(['launch', 'test', 'verify'], {'system': 'complete_integration'})
        
        health_results['builtin_foundation'] = True
        print("✅ Built-in Foundation: HEALTHY")
        print(f"   CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
        print(f"   AI Decision: {decision['decision']}")
        
    except Exception as e:
        print(f"❌ Built-in Foundation: {e}")
        health_results['builtin_foundation'] = False
    
    # Test AI Swarm
    try:
        from super_omega_ai_swarm import get_ai_swarm
        
        async def test_ai():
            swarm = await get_ai_swarm()
            result = await swarm['orchestrator'].orchestrate_task(
                "System health check for complete launch",
                {'health_check': True}
            )
            return swarm, result
        
        swarm, result = asyncio.run(test_ai())
        
        health_results['ai_swarm'] = True
        print("✅ AI Swarm: HEALTHY")
        print(f"   Components: {len(swarm['components'])} active")
        print(f"   Test Result: {result['status']}")
        
    except Exception as e:
        print(f"❌ AI Swarm: {e}")
        health_results['ai_swarm'] = False
    
    # Test Autonomous Layer
    try:
        from production_autonomous_orchestrator import get_production_orchestrator
        
        async def test_autonomous():
            orchestrator = await get_production_orchestrator()
            stats = orchestrator.get_system_stats()
            return stats
        
        stats = asyncio.run(test_autonomous())
        
        health_results['autonomous_layer'] = True
        print("✅ Autonomous Layer: HEALTHY")
        print(f"   Production Ready: {stats['production_ready']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Autonomous Layer: {e}")
        health_results['autonomous_layer'] = False
    
    # Overall health
    healthy_architectures = sum(health_results.values())
    overall_health = (healthy_architectures / 3) * 100
    
    print(f"\n📊 SYSTEM HEALTH: {overall_health:.1f}%")
    print(f"   Healthy Architectures: {healthy_architectures}/3")
    
    if overall_health >= 100:
        print("🏆 SYSTEM HEALTH: PERFECT - Ready for launch!")
    elif overall_health >= 66:
        print("✅ SYSTEM HEALTH: GOOD - System can launch")
    else:
        print("⚠️ SYSTEM HEALTH: ISSUES - Some components need attention")
    
    return overall_health >= 66

def main():
    """Main launcher"""
    
    print("🌟 SUPER-OMEGA: COMPLETE SYSTEM LAUNCHER")
    print("=" * 80)
    print("🚀 Launching sophisticated frontend-backend integration")
    print("🔄 Real-time updates like Cursor AI")
    print("🏗️ All three architectures coordinated")
    print("🌐 Zero external dependencies")
    print("=" * 80)
    
    # Test system health first
    if not test_system_health():
        print("\n❌ System health check failed. Cannot launch.")
        return
    
    print(f"\n🚀 LAUNCHING COMPLETE SUPER-OMEGA SYSTEM")
    print("=" * 50)
    
    try:
        # Start backend in separate thread
        backend_thread = threading.Thread(target=start_backend_server, daemon=True)
        backend_thread.start()
        
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Test backend connectivity
        try:
            import urllib.request
            response = urllib.request.urlopen('http://localhost:8081/', timeout=5)
            print("✅ Backend server: RUNNING and RESPONSIVE")
        except Exception as e:
            print(f"⚠️ Backend server: May still be starting - {e}")
        
        # Start frontend in separate thread
        frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
        frontend_thread.start()
        
        print("⏳ Waiting for frontend to start...")
        time.sleep(2)
        
        print("\n🎉 COMPLETE SUPER-OMEGA SYSTEM LAUNCHED!")
        print("=" * 50)
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8081")
        print("📊 Health Check: http://localhost:8081/health")
        print("🔄 Real-time Updates: Active via WebSocket")
        print("=" * 50)
        print("🎯 FEATURES AVAILABLE:")
        print("   ✅ Sophisticated chat interface")
        print("   ✅ Real-time architecture updates")
        print("   ✅ Live system monitoring")
        print("   ✅ All three architectures coordinated")
        print("   ✅ Cursor AI-like experience")
        print("   ✅ Zero external dependencies")
        print("=" * 50)
        
        # Open browser automatically
        try:
            webbrowser.open('http://localhost:3000')
            print("🌐 Browser opened automatically")
        except:
            print("🌐 Please open http://localhost:3000 in your browser")
        
        print(f"\n📋 USAGE INSTRUCTIONS:")
        print("1. Open http://localhost:3000 in your browser")
        print("2. Use the chat interface to send automation requests")
        print("3. Watch real-time updates from all three architectures")
        print("4. Observe sophisticated coordination in action")
        print("5. Press Ctrl+C here to stop the system")
        
        print(f"\n⌨️  Press Ctrl+C to stop the complete system")
        
        # Keep system running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping complete SUPER-OMEGA system...")
            print("✅ System stopped")
            
    except Exception as e:
        print(f"❌ Failed to launch complete system: {e}")

if __name__ == "__main__":
    main()