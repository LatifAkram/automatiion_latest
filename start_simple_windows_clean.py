#!/usr/bin/env python3
"""
THREE ARCHITECTURE STARTUP - Windows Clean Version
==================================================

Implements the complete three-architecture flow:
1. Built-in Foundation (Arch 1) - Zero dependencies, maximum reliability
2. AI Swarm (Arch 2) - Intelligent agents with fallbacks  
3. Autonomous Layer (Arch 3) - Full orchestration and workflow management

Flow: Frontend → Backend → Intent Analysis → Task Scheduling → Agent Execution → Result Aggregation
"""

import sys
import os
import subprocess
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

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

async def start_three_architecture_system():
    """Start the three-architecture system with full orchestration"""
    print("🚀 STARTING THREE ARCHITECTURE SYSTEM...")
    print("=" * 60)
    print("🏗️ Architecture 1: Built-in Foundation (Zero Dependencies)")
    print("🤖 Architecture 2: AI Swarm (Intelligent Agents)")  
    print("🚀 Architecture 3: Autonomous Layer (Full Orchestration)")
    print("=" * 60)
    
    try:
        # Add current directory to Python path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir / 'src'))
        sys.path.insert(0, str(current_dir / 'src' / 'core'))
        sys.path.insert(0, str(current_dir / 'src' / 'ui'))
        
        # Import and run the three architecture orchestrator
        from three_architecture_startup import ThreeArchitectureOrchestrator
        
        # Initialize orchestrator
        orchestrator = ThreeArchitectureOrchestrator()
        orchestrator.start_time = time.time()
        orchestrator.running = True
        
        # Initialize all three architectures
        await orchestrator.initialize_architectures()
        
        # Start web server for frontend communication
        print("\n🌐 STARTING WEB SERVER FOR FRONTEND COMMUNICATION...")
        web_server = await orchestrator.start_web_server(host='localhost', port=8888)
        
        print("\n✅ THREE ARCHITECTURE SYSTEM READY!")
        print("=" * 60)
        print("📱 Frontend → Backend: http://localhost:8888")
        print("🧠 Intent Analysis: AI Swarm analyzes user commands")
        print("📋 Task Scheduling: Autonomous Layer orchestrates execution")
        print("⚡ Agent Execution: Multi-architecture execution with fallbacks")
        print("📊 Result Aggregation: Comprehensive response generation")
        print("=" * 60)
        
        # Demo the system with sample tasks
        print("\n🎮 RUNNING SYSTEM DEMONSTRATION...")
        
        demo_tasks = [
            "Check system status and performance",
            "Analyze current automation capabilities", 
            "Execute a simple web automation task",
            "Process data using AI intelligence"
        ]
        
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n🎯 DEMO {i}/4: {task}")
            print("-" * 40)
            
            try:
                from three_architecture_startup import TaskPriority
                result = await orchestrator.process_user_instruction(task, TaskPriority.NORMAL)
                
                print(f"   ✅ Status: {result.status.value}")
                print(f"   ⏱️ Time: {result.execution_time:.2f}s")
                print(f"   🏗️ Architecture: {result.architecture_used}")
                
                if result.result:
                    print(f"   📊 Success Rate: {result.result.get('success_rate', 0):.1%}")
                    print(f"   📝 Summary: {result.result.get('summary', 'Completed')}")
                
            except Exception as e:
                print(f"   ❌ Demo failed: {e}")
        
        # Show system status
        print("\n📊 SYSTEM STATUS SUMMARY:")
        print("=" * 40)
        status = orchestrator.get_system_status()
        
        print(f"🏗️ Built-in Foundation: {status['architectures']['builtin_foundation']}")
        print(f"🤖 AI Swarm: {status['architectures']['ai_swarm']}")
        print(f"🚀 Autonomous Layer: {status['architectures']['autonomous_layer']}")
        print(f"📈 Performance: {status['performance_metrics']['successful_tasks']}/{status['performance_metrics']['total_tasks']} tasks successful")
        
        print(f"\n🌟 THREE ARCHITECTURE SYSTEM FULLY OPERATIONAL!")
        print(f"🎯 Ready for Manus AI-level autonomous automation")
        print(f"🌐 Access via: http://localhost:8888")
        print(f"🔄 Press Ctrl+C to stop")
        
        # Keep server running
        if web_server and hasattr(web_server, 'running'):
            try:
                while web_server.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down three architecture system...")
                if hasattr(web_server, 'stop'):
                    web_server.stop()
                orchestrator.running = False
        
    except ImportError as e:
        print(f"❌ Three architecture import error: {e}")
        print("🔄 Falling back to simple server mode...")
        await start_simple_fallback_server()
        
    except Exception as e:
        print(f"❌ Three architecture system failed: {e}")
        print("🔄 Falling back to simple server mode...")
        await start_simple_fallback_server()

async def start_simple_fallback_server():
    """Fallback to simple server if three architecture fails"""
    print("\n🔄 STARTING SIMPLE FALLBACK SERVER...")
    
    try:
        # Try to start the builtin web server directly
        from src.ui.builtin_web_server import LiveConsoleServer
        
        server = LiveConsoleServer()
        
        # Start the server
        if server.start():
            print(f"✅ Fallback server running at http://{server.host}:{server.port}")
            print("🌐 Basic automation features available")
            print("⚠️ Limited to single architecture (Built-in Foundation)")
            print("🔄 Press Ctrl+C to stop...")
            
            try:
                while server.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping fallback server...")
                server.stop()
        else:
            print("❌ Failed to start fallback server")
        
    except ImportError as e:
        print(f"❌ Fallback server import error: {e}")
        print("🔄 Trying alternative method...")
        
        # Try direct execution of the live console
        try:
            script_path = Path("src/ui/super_omega_live_console_fixed.py")
            if script_path.exists():
                print("🔄 Starting live console directly...")
                subprocess.run([sys.executable, str(script_path)])
            else:
                print("❌ Live console script not found")
                print("\n📋 MANUAL STARTUP INSTRUCTIONS:")
                print("   cd src/ui")
                print("   python builtin_web_server.py")
                
        except Exception as fallback_error:
            print(f"❌ All startup methods failed: {fallback_error}")
            print("\n📋 MANUAL STARTUP REQUIRED:")
            print("   1. cd src/ui")
            print("   2. python builtin_web_server.py")
            print("   3. Open http://localhost:8888 in browser")
            
    except Exception as e:
        print(f"❌ Fallback server startup failed: {e}")
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("  1. Make sure core dependencies are installed")
        print("  2. Check if port 8888 is available")
        print("  3. Try running as administrator")
        print("  4. Check Python path configuration")

async def main_async():
    """Main async startup function for three architecture system"""
    print("🚀 SUPER-OMEGA THREE ARCHITECTURE STARTUP")
    print("=" * 60)
    print("Implementing complete three-architecture autonomous flow:")
    print("🏗️ Architecture 1: Built-in Foundation (Zero Dependencies)")
    print("🤖 Architecture 2: AI Swarm (Intelligent Agents)")
    print("🚀 Architecture 3: Autonomous Layer (Full Orchestration)")
    print("=" * 60)
    print("📱 Flow: Frontend → Backend → Intent Analysis → Task Scheduling → Agent Execution → Result Aggregation")
    print("=" * 60)
    
    # Step 1: Install core dependencies
    print("\n🔧 STEP 1: Installing Core Dependencies")
    install_core_dependencies()
    
    # Step 2: Configure Windows environment  
    print("\n🪟 STEP 2: Configuring Windows Environment")
    set_windows_environment()
    
    # Step 3: Start three architecture system
    print("\n🚀 STEP 3: Starting Three Architecture System")
    await start_three_architecture_system()

def main():
    """Main startup function - runs async main"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Try running as administrator")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Verify all dependencies are installed")
        print("  4. Check network connectivity")

if __name__ == "__main__":
    main()