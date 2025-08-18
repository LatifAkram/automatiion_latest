#!/usr/bin/env python3
"""
STOP SERVER SCRIPT - Windows Compatible
======================================

Alternative way to stop the three architecture server
if Ctrl+C is not working properly on Windows.
"""

import sys
import os
import subprocess
import time

def stop_three_architecture_server():
    """Stop the three architecture server"""
    
    print("🛑 STOPPING THREE ARCHITECTURE SERVER")
    print("=" * 50)
    
    try:
        # Method 1: Find Python processes on port 8888
        print("🔍 Looking for server processes...")
        
        if os.name == 'nt':  # Windows
            # Windows command to find processes using port 8888
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            pids_to_kill = []
            for line in lines:
                if ':8888' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        pids_to_kill.append(pid)
                        print(f"   Found process PID: {pid}")
            
            # Kill the processes
            for pid in pids_to_kill:
                try:
                    subprocess.run(['taskkill', '/F', '/PID', pid], check=True)
                    print(f"   ✅ Stopped process PID: {pid}")
                except subprocess.CalledProcessError:
                    print(f"   ⚠️ Could not stop PID: {pid}")
            
            if pids_to_kill:
                print(f"✅ Stopped {len(pids_to_kill)} server process(es)")
            else:
                print("⚠️ No server processes found on port 8888")
                
        else:  # Linux/Mac
            # Unix command to find and kill processes
            try:
                result = subprocess.run(['lsof', '-ti:8888'], capture_output=True, text=True)
                pids = result.stdout.strip().split('\n')
                
                for pid in pids:
                    if pid:
                        subprocess.run(['kill', '-9', pid])
                        print(f"   ✅ Stopped process PID: {pid}")
                        
                print(f"✅ Stopped {len([p for p in pids if p])} server process(es)")
                
            except subprocess.CalledProcessError:
                print("⚠️ No server processes found on port 8888")
    
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        
        # Method 2: Kill all Python processes (more aggressive)
        print("\n🔄 Trying alternative method...")
        
        try:
            if os.name == 'nt':  # Windows
                # Kill all python processes running the server scripts
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq *three_architecture*'], 
                             capture_output=True)
                subprocess.run(['taskkill', '/F', '/IM', 'python3.exe', '/FI', 'WINDOWTITLE eq *three_architecture*'], 
                             capture_output=True)
            else:
                subprocess.run(['pkill', '-f', 'python.*three_architecture'], capture_output=True)
                subprocess.run(['pkill', '-f', 'python.*production_three_architecture'], capture_output=True)
            
            print("✅ Alternative stop method completed")
            
        except Exception as alt_error:
            print(f"❌ Alternative method failed: {alt_error}")
    
    # Verify server is stopped
    print("\n🔍 Verifying server is stopped...")
    time.sleep(2)
    
    try:
        import urllib.request
        
        req = urllib.request.Request("http://localhost:8888/")
        with urllib.request.urlopen(req, timeout=3) as response:
            print("⚠️ Server still running on port 8888")
            return False
            
    except:
        print("✅ Server successfully stopped - port 8888 is free")
        return True

def show_stop_instructions():
    """Show instructions for stopping the server"""
    
    print("\n💡 SERVER STOP INSTRUCTIONS:")
    print("=" * 40)
    print("Method 1: Ctrl+C in server terminal")
    print("   - Press Ctrl+C in the terminal running the server")
    print("   - May need to press multiple times on Windows")
    print("   - Try Ctrl+Break if Ctrl+C doesn't work")
    print("")
    print("Method 2: Run this stop script")
    print("   - python stop_server.py")
    print("   - Automatically finds and stops server processes")
    print("")
    print("Method 3: Close terminal window")
    print("   - Simply close the terminal window running the server")
    print("   - Windows will terminate the process")
    print("")
    print("Method 4: Task Manager (Windows)")
    print("   - Open Task Manager (Ctrl+Shift+Esc)")
    print("   - Find python.exe or python3.exe processes")
    print("   - End the process running the server")
    print("=" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        show_stop_instructions()
    else:
        success = stop_three_architecture_server()
        
        if not success:
            print("\n🔧 If server is still running, try:")
            print("   1. Close the terminal window")
            print("   2. Use Task Manager to end Python processes")
            print("   3. Restart your computer (last resort)")
        else:
            print("\n🎯 Server successfully stopped!")
            print("   You can now restart with: python start_simple_windows_clean.py")