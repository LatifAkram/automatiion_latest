#!/usr/bin/env python3
"""
Setup Script for Autonomous Multi-Agent Automation Platform
==========================================================

This script sets up the complete platform with all dependencies,
configuration, and initial setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def install_playwright_browsers():
    """Install Playwright browsers."""
    print("\n🌐 Installing Playwright browsers...")
    if not run_command("playwright install", "Installing Playwright browsers"):
        return False
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "data/logs",
        "data/media",
        "data/media/screenshots",
        "data/media/videos",
        "data/media/recordings",
        "data/vector_db",
        "data/audit",
        "config",
        "workflows",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def create_config_files():
    """Create configuration files."""
    print("\n⚙️ Creating configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# AI Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
LOCAL_LLM_URL=http://localhost:8000
LOCAL_LLM_MODEL=deepseek-coder

# Database Configuration
DATABASE_PATH=./data/automation.db
VECTOR_DB_PATH=./data/vector_db

# Search Configuration
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_CX=your_google_search_cx_here
BING_SEARCH_API_KEY=your_bing_search_api_key_here
GITHUB_TOKEN=your_github_token_here
STACK_OVERFLOW_KEY=your_stack_overflow_key_here
NEWS_API_KEY=your_news_api_key_here

# Automation Configuration
BROWSER_TYPE=chromium
HEADLESS=true
BROWSER_TIMEOUT=30
VIEWPORT_WIDTH=1920
VIEWPORT_HEIGHT=1080
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
LOCALE=en-US
TIMEZONE=UTC

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./data/logs/automation.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# Media Configuration
MEDIA_PATH=./data/media
CAPTURE_SCREENSHOTS=true
CAPTURE_VIDEOS=false
MEDIA_QUALITY=medium

# Performance Configuration
MAX_CONCURRENT_WORKFLOWS=10
MAX_SEARCH_RESULTS=20
SEARCH_TIMEOUT=30
"""
        
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️ .env file already exists")
    
    # Create config.json
    config_file = Path("config/config.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    config_content = {
        "platform": {
            "name": "Autonomous Multi-Agent Automation Platform",
            "version": "1.0.0",
            "description": "Ultra-complex automation platform with multi-agent orchestration"
        },
        "features": {
            "multi_agent": True,
            "ai_planning": True,
            "web_automation": True,
            "search_integration": True,
            "data_extraction": True,
            "conversational_ai": True,
            "self_healing": True,
            "compliance": True
        },
        "agents": {
            "planner": {
                "enabled": True,
                "max_concurrent_plans": 5
            },
            "executor": {
                "enabled": True,
                "max_concurrent_tasks": 10
            },
            "conversational": {
                "enabled": True,
                "max_conversations": 100
            },
            "search": {
                "enabled": True,
                "max_concurrent_searches": 5
            },
            "dom_extractor": {
                "enabled": True,
                "max_concurrent_extractions": 5
            }
        },
        "ai_providers": {
            "openai": {
                "enabled": True,
                "model": "gpt-4",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            "anthropic": {
                "enabled": True,
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            "google": {
                "enabled": True,
                "model": "gemini-pro",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            "local": {
                "enabled": True,
                "model": "deepseek-coder",
                "max_tokens": 4000,
                "temperature": 0.7
            }
        },
        "automation": {
            "browser_type": "chromium",
            "headless": True,
            "timeout": 30,
            "viewport": {
                "width": 1920,
                "height": 1080
            },
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "locale": "en-US",
            "timezone": "UTC"
        },
        "search": {
            "providers": ["google", "bing", "duckduckgo", "github", "stackoverflow"],
            "max_results": 20,
            "timeout": 30
        },
        "storage": {
            "database": "./data/automation.db",
            "vector_db": "./data/vector_db",
            "media": "./data/media",
            "logs": "./data/logs",
            "audit": "./data/audit"
        },
        "security": {
            "encryption_enabled": True,
            "pii_detection": True,
            "audit_logging": True,
            "compliance_reporting": True
        },
        "monitoring": {
            "performance_tracking": True,
            "error_reporting": True,
            "health_checks": True,
            "metrics_collection": True
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(config_content, f, indent=2)
    print("✅ Created config.json file")
    
    return True


def create_sample_workflows():
    """Create sample workflow files."""
    print("\n📋 Creating sample workflows...")
    
    workflows_dir = Path("workflows")
    workflows_dir.mkdir(exist_ok=True)
    
    # Sample e-commerce workflow
    ecommerce_workflow = {
        "name": "E-commerce Product Research",
        "description": "Research product information from multiple e-commerce sites",
        "domain": "ecommerce",
        "version": "1.0.0",
        "author": "Platform User",
        "created_at": "2024-01-01T00:00:00Z",
        "tasks": [
            {
                "id": "search_products",
                "name": "Search for products",
                "type": "search",
                "description": "Search for product information",
                "parameters": {
                    "query": "wireless headphones best 2024",
                    "sources": ["google", "bing", "duckduckgo"],
                    "max_results": 10
                },
                "dependencies": [],
                "timeout": 30
            },
            {
                "id": "extract_data",
                "name": "Extract product data",
                "type": "dom_extraction",
                "description": "Extract product information from websites",
                "parameters": {
                    "urls": ["https://www.amazon.com", "https://www.bestbuy.com"],
                    "selectors": {
                        "product_title": "h1.product-title, .product-name h1",
                        "product_price": ".price, [class*='price']",
                        "product_rating": ".rating, [class*='rating']",
                        "product_description": ".description, .product-details"
                    },
                    "content_type": "ecommerce"
                },
                "dependencies": ["search_products"],
                "timeout": 60
            },
            {
                "id": "process_data",
                "name": "Process and analyze data",
                "type": "data_processing",
                "description": "Process extracted data and generate insights",
                "parameters": {
                    "operations": [
                        {
                            "type": "filter",
                            "field": "price",
                            "condition": "less_than",
                            "value": 200
                        },
                        {
                            "type": "sort",
                            "field": "rating",
                            "direction": "desc"
                        },
                        {
                            "type": "aggregate",
                            "field": "price",
                            "aggregate_type": "average"
                        }
                    ]
                },
                "dependencies": ["extract_data"],
                "timeout": 30
            }
        ],
        "outputs": {
            "product_data": "Processed product information",
            "insights": "Analysis and recommendations"
        }
    }
    
    with open(workflows_dir / "ecommerce_research.json", "w") as f:
        json.dump(ecommerce_workflow, f, indent=2)
    print("✅ Created ecommerce_research.json workflow")
    
    # Sample news automation workflow
    news_workflow = {
        "name": "News Article Automation",
        "description": "Automate news article reading and analysis",
        "domain": "news",
        "version": "1.0.0",
        "author": "Platform User",
        "created_at": "2024-01-01T00:00:00Z",
        "tasks": [
            {
                "id": "navigate_news",
                "name": "Navigate to news site",
                "type": "web_automation",
                "description": "Navigate to news website",
                "parameters": {
                    "url": "https://news.ycombinator.com",
                    "actions": [
                        {"type": "navigate", "url": "https://news.ycombinator.com"},
                        {"type": "wait", "selector": ".athing", "timeout": 5000},
                        {"type": "screenshot", "name": "homepage"}
                    ],
                    "record_video": False,
                    "capture_screenshot": True
                },
                "dependencies": [],
                "timeout": 30
            },
            {
                "id": "extract_articles",
                "name": "Extract article information",
                "type": "dom_extraction",
                "description": "Extract article links and metadata",
                "parameters": {
                    "url": "https://news.ycombinator.com",
                    "selectors": {
                        "article_links": "a.storylink",
                        "article_titles": ".storylink",
                        "article_scores": ".score",
                        "article_comments": ".subtext a:last-child"
                    },
                    "content_type": "news"
                },
                "dependencies": ["navigate_news"],
                "timeout": 30
            }
        ],
        "outputs": {
            "articles": "List of articles with metadata",
            "screenshots": "Page screenshots"
        }
    }
    
    with open(workflows_dir / "news_automation.json", "w") as f:
        json.dump(news_workflow, f, indent=2)
    print("✅ Created news_automation.json workflow")
    
    return True


def create_startup_scripts():
    """Create startup scripts."""
    print("\n🚀 Creating startup scripts...")
    
    # Create start.py
    start_script = """#!/usr/bin/env python3
\"\"\"
Startup Script for Autonomous Multi-Agent Automation Platform
============================================================

This script starts the platform with proper configuration and error handling.
\"\"\"

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.orchestrator import MultiAgentOrchestrator
from src.core.config import Config
from src.utils.logger import setup_logging
from src.api.server import start_api_server

async def main():
    \"\"\"Main startup function.\"\"\"
    print("🚀 Starting Autonomous Multi-Agent Automation Platform...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")
        
        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator(config)
        await orchestrator.initialize()
        logger.info("Multi-agent orchestrator initialized")
        
        # Start API server
        print("🌐 Starting API server...")
        api_task = asyncio.create_task(start_api_server(orchestrator))
        logger.info("API server started")
        
        # Wait for API server
        await api_task
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("start.py", "w") as f:
        f.write(start_script)
    
    # Make executable
    os.chmod("start.py", 0o755)
    print("✅ Created start.py script")
    
    # Create start.sh for Unix systems
    start_sh = """#!/bin/bash

# Startup Script for Autonomous Multi-Agent Automation Platform
# =============================================================

echo "🚀 Starting Autonomous Multi-Agent Automation Platform..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the platform
python3 start.py
"""
    
    with open("start.sh", "w") as f:
        f.write(start_sh)
    
    # Make executable
    os.chmod("start.sh", 0o755)
    print("✅ Created start.sh script")
    
    # Create start.bat for Windows
    start_bat = """@echo off

REM Startup Script for Autonomous Multi-Agent Automation Platform
REM =============================================================

echo 🚀 Starting Autonomous Multi-Agent Automation Platform...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist venv\\Scripts\\activate.bat (
    echo 📦 Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Start the platform
python start.py
pause
"""
    
    with open("start.bat", "w") as f:
        f.write(start_bat)
    print("✅ Created start.bat script")
    
    return True


def create_documentation():
    """Create basic documentation."""
    print("\n📚 Creating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Quick start guide
    quick_start = """# Quick Start Guide

## Autonomous Multi-Agent Automation Platform

### Prerequisites
- Python 3.8 or higher
- Internet connection for API access
- At least 4GB RAM
- 2GB free disk space

### Installation

1. **Clone or download the platform**
   ```bash
   git clone <repository-url>
   cd autonomous-automation-platform
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Configure your API keys**
   Edit the `.env` file and add your API keys:
   - OpenAI API key
   - Anthropic API key
   - Google API key
   - Other service API keys as needed

4. **Start the platform**
   ```bash
   python start.py
   ```

### Usage

1. **Access the API**
   - Open your browser to `http://localhost:8000`
   - View the interactive API documentation

2. **Create a workflow**
   - Use the API to create workflows
   - Or use the sample workflows in the `workflows/` directory

3. **Execute workflows**
   - Send workflow requests to the API
   - Monitor progress and results

### Sample Workflows

The platform includes sample workflows:
- `workflows/ecommerce_research.json` - E-commerce product research
- `workflows/news_automation.json` - News article automation

### API Endpoints

- `GET /health` - Check platform health
- `POST /workflows` - Create and execute workflows
- `GET /workflows/{id}` - Get workflow status
- `POST /chat` - Chat with the AI agent
- `GET /analytics` - Get platform analytics

### Configuration

Edit `config/config.json` to customize:
- AI provider settings
- Automation parameters
- Search configurations
- Security settings

### Troubleshooting

1. **Check logs**
   - View logs in `data/logs/automation.log`

2. **Verify API keys**
   - Ensure all required API keys are set in `.env`

3. **Check dependencies**
   - Run `pip install -r requirements.txt`

4. **Browser issues**
   - Run `playwright install` to install browsers

### Support

For issues and questions:
- Check the logs for error messages
- Review the configuration files
- Ensure all dependencies are installed
"""
    
    with open(docs_dir / "quick_start.md", "w") as f:
        f.write(quick_start)
    print("✅ Created quick_start.md")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Autonomous Multi-Agent Automation Platform Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Install Playwright browsers
    if not install_playwright_browsers():
        sys.exit(1)
    
    # Create configuration files
    if not create_config_files():
        sys.exit(1)
    
    # Create sample workflows
    if not create_sample_workflows():
        sys.exit(1)
    
    # Create startup scripts
    if not create_startup_scripts():
        sys.exit(1)
    
    # Create documentation
    if not create_documentation():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your API keys")
    print("2. Review the configuration in config/config.json")
    print("3. Run 'python start.py' to start the platform")
    print("4. Open http://localhost:8000 to access the API")
    print("\n📚 Documentation:")
    print("- Quick start guide: docs/quick_start.md")
    print("- Sample workflows: workflows/")
    print("- Configuration: config/")
    print("\n🔧 Troubleshooting:")
    print("- Check logs in data/logs/")
    print("- Verify API keys in .env")
    print("- Run 'python test_platform.py' to test the platform")


if __name__ == "__main__":
    main()