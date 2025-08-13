#!/usr/bin/env python3
"""
Local LLM Setup Script
======================

This script helps set up and start a local LLM server for the automation platform.
"""

import subprocess
import sys
import os
import time
import requests
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_ollama():
    """Install Ollama if not already installed."""
    try:
        # Check if ollama is already installed
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("📦 Installing Ollama...")
    
    # Install Ollama using curl
    try:
        subprocess.run([
            'curl', '-fsSL', 'https://ollama.ai/install.sh', '|', 'sh'
        ], shell=True, check=True)
        print("✅ Ollama installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Ollama: {e}")
        return False

def pull_model(model_name="deepseek-coder:6.7b"):
    """Pull the specified model."""
    print(f"📥 Pulling model: {model_name}")
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print(f"✅ Model {model_name} pulled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull model: {e}")
        return False

def start_ollama_server():
    """Start Ollama server."""
    print("🚀 Starting Ollama server...")
    try:
        # Start ollama serve in background
        process = subprocess.Popen(['ollama', 'serve'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running on http://localhost:11434")
                return process
        except requests.exceptions.RequestException:
            pass
        
        print("❌ Failed to start Ollama server")
        return None
        
    except Exception as e:
        print(f"❌ Error starting Ollama server: {e}")
        return None

def test_local_llm():
    """Test the local LLM connection."""
    print("🧪 Testing local LLM connection...")
    
    test_payload = {
        "model": "deepseek-coder:6.7b",
        "messages": [
            {"role": "system", "content": "Always answer in rhymes. Today is Thursday"},
            {"role": "user", "content": "What day is it today?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Local LLM test successful!")
            print(f"Response: {data.get('message', {}).get('content', 'No content')}")
            return True
        else:
            print(f"❌ Local LLM test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Local LLM test failed: {e}")
        return False

def create_config_file():
    """Create a configuration file for the local LLM."""
    config = {
        "local_llm_url": "http://localhost:11434",
        "local_llm_model": "deepseek-coder:6.7b",
        "local_llm_max_tokens": 2000,
        "local_llm_temperature": 0.7
    }
    
    config_file = Path(".env")
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write("# Local LLM Configuration\n")
            f.write(f"LOCAL_LLM_URL={config['local_llm_url']}\n")
            f.write(f"LOCAL_LLM_MODEL={config['local_llm_model']}\n")
            f.write(f"LOCAL_LLM_MAX_TOKENS={config['local_llm_max_tokens']}\n")
            f.write(f"LOCAL_LLM_TEMPERATURE={config['local_llm_temperature']}\n")
        print("✅ Created .env configuration file")
    else:
        print("✅ .env configuration file already exists")

def main():
    """Main setup function."""
    print("🤖 Local LLM Setup for Automation Platform")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install Ollama
    if not install_ollama():
        print("❌ Setup failed: Could not install Ollama")
        return False
    
    # Pull model
    if not pull_model():
        print("❌ Setup failed: Could not pull model")
        return False
    
    # Start server
    server_process = start_ollama_server()
    if not server_process:
        print("❌ Setup failed: Could not start server")
        return False
    
    # Test connection
    if not test_local_llm():
        print("❌ Setup failed: Local LLM test failed")
        return False
    
    # Create config
    create_config_file()
    
    print("\n🎉 Local LLM setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. The Ollama server is running on http://localhost:11434")
    print("2. Your automation platform will now use the local LLM")
    print("3. To stop the server, run: pkill -f ollama")
    print("4. To restart, run: ollama serve")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)