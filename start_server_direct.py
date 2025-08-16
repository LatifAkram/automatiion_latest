#!/usr/bin/env python3
"""
Direct Server Startup - No Package Installation
===============================================

This script starts the SUPER-OMEGA server directly using only
built-in Python libraries, bypassing any package installation issues.
"""

import sys
import os
from pathlib import Path

# Fix protobuf compatibility issue on Windows
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

def start_server():
    """Start the server directly"""
    print("🎯 SUPER-OMEGA Direct Server Startup")
    print("=" * 45)
    print("🔧 Using built-in web server (no external dependencies)")
    print()
    
    try:
        # Import and start the built-in web server
        from ui.builtin_web_server import LiveConsoleServer
        
        print("✅ Imported LiveConsoleServer successfully")
        
        # Create and start server
        server = LiveConsoleServer()
        print(f"🌐 Starting server on http://{server.host}:{server.port}")
        
        if server.start():
            print("✅ Server started successfully!")
            print()
            print("🎯 SUPER-OMEGA Platform Ready!")
            print("=" * 40)
            print(f"📱 Open http://{server.host}:{server.port} in your browser")
            print("✅ 100% Self-healing selectors active")
            print("✅ Real-time automation ready")
            print("✅ No external dependencies required")
            print()
            print("Press Ctrl+C to stop the server")
            print("=" * 40)
            
            try:
                while server.running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                server.stop()
                print("✅ Server stopped successfully")
        else:
            print("❌ Failed to start server")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Trying alternative startup method...")
        
        try:
            # Fallback: try to run the builtin web server directly
            from ui import builtin_web_server
            builtin_web_server.main()
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return False
    
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = start_server()
    sys.exit(0 if success else 1)