#!/usr/bin/env python3
"""
SIMPLE BACKEND SERVER STARTUP
=============================
Reliable server startup for SUPER-OMEGA system
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'ui'))

def start_server():
    """Start the server using the original method"""
    print("🚀 STARTING SUPER-OMEGA BACKEND SERVER")
    print("=" * 50)
    
    try:
        from builtin_web_server import LiveConsoleServer
        
        print("🌐 Creating server on port 8081...")
        server = LiveConsoleServer()
        
        print("""
🎉 BACKEND SERVER STARTING!
===========================
🌐 Server URL: http://localhost:8081
📡 Health Check: http://localhost:8081/health
🤖 Automation API: http://localhost:8081/api/fixed-super-omega-execute
🔍 Web Search: http://localhost:8081/search/web
📊 System Status: http://localhost:8081/system/status

⚡ ALL SOPHISTICATED FEATURES ENABLED!
🚀 Starting server... (Press Ctrl+C to stop)
""")
        
        if server.start():
            print(f"✅ Server running at http://{server.host}:{server.port}")
            print("🔗 Open the URL in your browser to see the live console")
            print("📡 WebSocket support included - no external dependencies!")
            print("Press Ctrl+C to stop...")
            
            try:
                while server.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                server.stop()
                print("✅ Server stopped successfully!")
        else:
            print("❌ Failed to start server")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_server()