#!/usr/bin/env python3
"""
Minimal Windows Startup - Web Server Only
=========================================

This script starts just the basic web server with Playwright automation,
avoiding all complex imports that cause Windows compatibility issues.
"""

import sys
import os
import subprocess
from pathlib import Path

def set_environment():
    """Set minimal environment variables"""
    print("🔧 Setting minimal environment...")
    
    # Fix protobuf compatibility
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # Set UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    print("✅ Environment configured")

def install_playwright():
    """Install only Playwright"""
    print("📦 Installing Playwright...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], 
                      check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
        print("✅ Playwright installed")
    except subprocess.CalledProcessError:
        print("⚠️ Playwright installation failed, but continuing...")

def start_minimal_server():
    """Start minimal web server"""
    print("🚀 Starting Minimal Web Server...")
    print("🌐 Server will be available at: http://localhost:8888")
    print("✅ Playwright automation ready!")
    print("")
    print("📱 Open http://localhost:8888 in your browser!")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Start the builtin web server directly
        exec("""
import sys
sys.path.insert(0, 'src')

# Import only the basic web server
from ui.builtin_web_server import LiveConsoleServer
import time

print("✅ Starting LiveConsoleServer...")

server = LiveConsoleServer()

if server.start():
    print(f"✅ Server running at http://{server.host}:{server.port}")
    print("🔗 Open the URL in your browser to start automation")
    print("Press Ctrl+C to stop...")
    
    try:
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n🛑 Stopping server...")
        server.stop()
else:
    print("❌ Failed to start server")
""")
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("🔧 Trying direct execution...")
        
        # Try running the builtin web server directly
        try:
            script_path = Path("src/ui/builtin_web_server.py")
            if script_path.exists():
                print("✅ Running builtin web server directly...")
                subprocess.run([sys.executable, str(script_path)])
            else:
                print("❌ Builtin web server not found")
                print("🆘 Manual startup required:")
                print("   cd src/ui")
                print("   python builtin_web_server.py")
                
        except Exception as fallback_error:
            print(f"❌ All startup methods failed: {fallback_error}")

def main():
    """Main minimal startup function"""
    print("🎯 SUPER-OMEGA Minimal Windows Startup")
    print("=" * 45)
    print("ℹ️  This version uses only basic web server functionality")
    print("")
    
    # Step 1: Set environment
    set_environment()
    
    # Step 2: Install Playwright
    install_playwright()
    
    # Step 3: Start minimal server
    start_minimal_server()

if __name__ == "__main__":
    main()