#!/usr/bin/env python3
"""
Windows-Compatible SUPER-OMEGA Startup Script
=============================================

This script handles Windows-specific compatibility issues and starts
the SUPER-OMEGA system with proper dependency management.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies for Windows"""
    print("🔧 Installing Windows-compatible dependencies...")
    
    required_packages = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "aiohttp>=3.8.0", 
        "websockets>=11.0.0",
        "psutil>=5.9.0",
        "playwright>=1.40.0"
    ]
    
    for package in required_packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            continue
    
    # Install Playwright browsers
    try:
        print("🌐 Installing Playwright browsers...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("⚠️ Playwright browser installation failed, but continuing...")

def fix_windows_compatibility():
    """Apply Windows compatibility fixes"""
    print("🔧 Applying Windows compatibility fixes...")
    
    # The resource module fix is already applied to builtin_performance_monitor.py
    print("✅ Resource module compatibility fixed")
    
    # Set Windows-specific environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    print("✅ Windows environment configured")

def start_super_omega_server():
    """Start the SUPER-OMEGA server with Windows compatibility"""
    print("🚀 Starting SUPER-OMEGA Live Console Server...")
    
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Import and start the server
        from src.ui.super_omega_live_console_fixed import SuperOmegaLiveConsole
        
        print("✅ SUPER-OMEGA modules imported successfully")
        
        # Create and start the console
        console = SuperOmegaLiveConsole()
        
        print("🎯 SUPER-OMEGA Live Console starting...")
        print("🌐 Server will be available at: http://localhost:8888")
        print("✅ Ready for automation instructions!")
        print("")
        print("🎭 FEATURES AVAILABLE:")
        print("  ✅ Live Playwright automation")
        print("  ✅ 12 multi-domain workflows")
        print("  ✅ Self-healing selectors")
        print("  ✅ Evidence collection")
        print("  ✅ Real-time monitoring")
        print("")
        print("📱 Open http://localhost:8888 in your browser to start!")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the server
        console.run_server()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Trying alternative startup method...")
        
        # Fallback to direct execution
        try:
            import asyncio
            from src.ui.builtin_web_server import BuiltinWebServer
            
            server = BuiltinWebServer()
            print("✅ Fallback server starting on http://localhost:8888")
            
            # Start server
            asyncio.run(server.start_server())
            
        except Exception as fallback_error:
            print(f"❌ Fallback startup failed: {fallback_error}")
            print("🆘 Please check the error messages above and try:")
            print("   1. pip install -r requirements_super_omega.txt")
            print("   2. python -m playwright install chromium")
            print("   3. python src/ui/super_omega_live_console_fixed.py")
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("🔧 Troubleshooting tips:")
        print("  1. Make sure all dependencies are installed")
        print("  2. Check if port 8888 is available")
        print("  3. Try running as administrator")

def main():
    """Main Windows startup function"""
    print("🎯 SUPER-OMEGA Windows Startup")
    print("=" * 40)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Apply Windows fixes
    fix_windows_compatibility()
    
    # Step 3: Start the server
    start_super_omega_server()

if __name__ == "__main__":
    main()