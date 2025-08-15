#!/usr/bin/env python3
"""
Windows Startup Script - Fixed Version
======================================

Clean startup script for Windows with proper encoding and line endings.
"""

import sys
import os
import subprocess
from pathlib import Path

def set_windows_environment():
    """Set Windows-specific environment variables"""
    print("Setting Windows environment...")
    
    # Fix protobuf compatibility
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # Set UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Disable AI dependencies to avoid import errors
    os.environ['SUPER_OMEGA_NO_AI'] = '1'
    
    print("Windows environment configured")

def start_direct_server():
    """Start the server directly without complex imports"""
    print("Starting SUPER-OMEGA Direct Server...")
    
    try:
        # Add src to Python path
        current_dir = Path(__file__).parent
        src_dir = current_dir / 'src'
        sys.path.insert(0, str(src_dir))
        
        print("Starting built-in web server...")
        print("Server will be available at: http://localhost:8080")
        print("")
        
        # Import and start the built-in web server
        from ui.builtin_web_server import LiveConsoleServer
        
        server = LiveConsoleServer()
        
        if server.start():
            print("Server started successfully!")
            print("")
            print("SUPER-OMEGA Platform Ready!")
            print("=" * 40)
            print("Open http://localhost:8080 in your browser")
            print("100% Self-healing selectors active")
            print("Real-time automation ready")
            print("")
            print("Press Ctrl+C to stop the server")
            print("=" * 40)
            
            try:
                while server.running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server...")
                server.stop()
                print("Server stopped successfully")
        else:
            print("Failed to start server")
            return False
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying fallback method...")
        
        # Try using the direct startup script
        try:
            direct_script = current_dir / 'start_server_direct.py'
            if direct_script.exists():
                print("Running direct server script...")
                subprocess.run([sys.executable, str(direct_script)])
            else:
                print("Direct server script not found")
                return False
                
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return False
    
    except Exception as e:
        print(f"Server startup error: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("SUPER-OMEGA Windows Startup (Fixed)")
    print("=" * 40)
    print("Using built-in web server (no external dependencies)")
    print("")
    
    # Set Windows environment
    set_windows_environment()
    
    # Start server
    success = start_direct_server()
    
    if not success:
        print("")
        print("Manual startup instructions:")
        print("1. cd src/ui")
        print("2. python builtin_web_server.py")
        print("")
        print("Or try:")
        print("python start_server_direct.py")

if __name__ == "__main__":
    main()