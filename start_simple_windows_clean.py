#!/usr/bin/env python3
"""
Simple Windows Startup - Clean Version
======================================

Clean startup script for Windows with no syntax errors.
"""

import sys
import os
import subprocess
from pathlib import Path

def install_core_dependencies():
    """Install only core dependencies needed for automation"""
    print("Installing core automation dependencies...")
    
    core_packages = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "aiohttp>=3.8.0", 
        "websockets>=11.0.0",
        "psutil>=5.9.0",
        "playwright>=1.40.0"
    ]
    
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            continue
    
    # Install Playwright browsers
    try:
        print("Installing Playwright browsers...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Playwright browser installation failed, but continuing...")

def set_windows_environment():
    """Set Windows-specific environment variables"""
    print("Configuring Windows environment...")
    
    # Fix protobuf compatibility
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # Set UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Disable AI dependencies to avoid import errors
    os.environ['SUPER_OMEGA_NO_AI'] = '1'
    
    print("Windows environment configured")
    print("AI dependencies disabled (fallback mode)")

def start_simple_server():
    """Start the server with minimal dependencies"""
    print("Starting SUPER-OMEGA Simple Server...")
    
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        print("Starting core automation server...")
        print("Server will be available at: http://localhost:8888")
        print("Playwright automation ready!")
        print("")
        print("CORE FEATURES AVAILABLE:")
        print("  Live Playwright automation")
        print("  Basic workflow support")
        print("  Evidence collection")
        print("  Real-time monitoring")
        print("  AI features in fallback mode")
        print("")
        print("Open http://localhost:8888 in your browser to start!")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Try to start the builtin web server directly
        from src.ui.builtin_web_server import LiveConsoleServer
        
        server = LiveConsoleServer()
        
        # Start the server
        if server.start():
            print(f"Server running at http://{server.host}:{server.port}")
            print("Open the URL in your browser to start automation")
            print("WebSocket support included - no external dependencies!")
            print("Press Ctrl+C to stop...")
            
            try:
                while server.running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server...")
                server.stop()
        else:
            print("Failed to start server")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative method...")
        
        # Try direct execution of the live console
        try:
            script_path = Path("src/ui/super_omega_live_console_fixed.py")
            if script_path.exists():
                print("Starting live console directly...")
                subprocess.run([sys.executable, str(script_path)])
            else:
                print("Live console script not found")
                
        except Exception as fallback_error:
            print(f"All startup methods failed: {fallback_error}")
            print("Manual startup required:")
            print("   cd src/ui")
            print("   python builtin_web_server.py")
            
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        
    except Exception as e:
        print(f"Server startup failed: {e}")
        print("Troubleshooting tips:")
        print("  1. Make sure core dependencies are installed")
        print("  2. Check if port 8888 is available")
        print("  3. Try running as administrator")

def main():
    """Main simple startup function"""
    print("SUPER-OMEGA Simple Windows Startup")
    print("=" * 45)
    print("This version avoids AI dependencies for Windows compatibility")
    print("")
    
    # Step 1: Install core dependencies only
    install_core_dependencies()
    
    # Step 2: Configure Windows environment
    set_windows_environment()
    
    # Step 3: Start the simple server
    start_simple_server()

if __name__ == "__main__":
    main()