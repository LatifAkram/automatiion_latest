#!/usr/bin/env python3
"""
Simple Server Test
==================

Test the builtin web server directly to debug issues.
"""

import sys
import os
from pathlib import Path
import time

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

def test_server():
    """Test server startup step by step"""
    print("🧪 Testing Server Startup")
    print("=" * 30)
    
    try:
        print("1. Importing modules...")
        from ui.builtin_web_server import LiveConsoleServer
        print("   ✅ Import successful")
        
        print("2. Creating server instance...")
        server = LiveConsoleServer(host="localhost", port=8080)
        print(f"   ✅ Server created: {server.host}:{server.port}")
        
        print("3. Starting server...")
        success = server.start()
        print(f"   {'✅' if success else '❌'} Start result: {success}")
        
        if success:
            print("4. Testing server status...")
            print(f"   Running: {server.running}")
            print(f"   Server object: {hasattr(server, 'server')}")
            
            if hasattr(server, 'server'):
                print(f"   Server address: {server.server.server_address}")
            
            print("5. Testing connection...")
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            try:
                result = sock.connect_ex(('localhost', 8080))
                print(f"   Connection result: {result} ({'success' if result == 0 else 'failed'})")
            except Exception as e:
                print(f"   Connection error: {e}")
            finally:
                sock.close()
            
            print("6. Keeping server alive for 5 seconds...")
            time.sleep(5)
            
            print("7. Stopping server...")
            server.stop()
            print("   ✅ Server stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_server()
    print(f"\n🏁 Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)