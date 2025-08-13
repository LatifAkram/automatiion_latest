# 🚀 Quick Start Guide - Autonomous Multi-Agent Automation Platform

## ⚡ Immediate Setup (5 Minutes)

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python test_platform.py
```

### 2. Configuration
Create a `.env` file in the project root:
```env
# AI Providers (Optional - platform works without them)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Search APIs (Optional)
GOOGLE_SEARCH_API_KEY=your_google_search_key
GOOGLE_SEARCH_CX=your_search_engine_id
BING_SEARCH_API_KEY=your_bing_key
GITHUB_TOKEN=your_github_token

# Local LLM (Optional - for offline operation)
LOCAL_LLM_URL=http://127.0.0.1:1234
LOCAL_LLM_MODEL=deepseek-coder-v2-lite-instruct

# Platform Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start the Platform
```bash
# Start the platform
python main.py
```

### 4. Access the Platform
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Info**: http://localhost:8000/system/info

## 🎯 Quick Examples

### Example 1: E-commerce Price Comparison
```python
import requests

# Create workflow
workflow = {
    "name": "Laptop Price Comparison",
    "domain": "ecommerce",
    "parameters": {
        "product": "gaming laptop",
        "budget": 1500,
        "sites": ["amazon", "bestbuy"]
    }
}

response = requests.post("http://localhost:8000/workflows", json=workflow)
workflow_id = response.json()["workflow_id"]

# Check status
status = requests.get(f"http://localhost:8000/workflows/{workflow_id}")
print(status.json())
```

### Example 2: Financial Market Analysis
```python
workflow = {
    "name": "Tech Stock Analysis",
    "domain": "finance",
    "parameters": {
        "sectors": ["technology"],
        "analysis_type": "comprehensive",
        "include_sentiment": True
    }
}

response = requests.post("http://localhost:8000/workflows", json=workflow)
```

### Example 3: Research Data Gathering
```python
workflow = {
    "name": "AI Healthcare Research",
    "domain": "research",
    "parameters": {
        "topic": "artificial intelligence in healthcare",
        "sources": ["academic_papers", "news", "patents"],
        "time_period": "last_2_years"
    }
}

response = requests.post("http://localhost:8000/workflows", json=workflow)
```

## 💬 Chat with AI Agent
```python
# Start a conversation
chat_request = {
    "message": "What's the status of my workflow?",
    "session_id": "my_session",
    "context": {"workflow_id": "workflow_123"}
}

response = requests.post("http://localhost:8000/chat", json=chat_request)
print(response.json()["response"])
```

## 🔧 Advanced Usage

### Custom Workflow Definition
```python
workflow = {
    "name": "Custom Automation",
    "domain": "custom",
    "parameters": {
        "tasks": [
            {
                "type": "web_automation",
                "url": "https://example.com",
                "actions": [
                    {"type": "click", "selector": ".button"},
                    {"type": "type", "selector": "input", "text": "search term"}
                ]
            },
            {
                "type": "api_call",
                "method": "GET",
                "url": "https://api.example.com/data"
            }
        ]
    }
}
```

### Real-Time Data Processing
```python
# The platform automatically handles:
# - Live web scraping
# - Real-time API calls
# - Dynamic content extraction
# - Self-healing mechanisms
# - Performance optimization
```

## 📊 Monitoring & Analytics

### View Performance Metrics
```bash
curl http://localhost:8000/metrics/performance
```

### Check Agent Status
```bash
curl http://localhost:8000/system/agents
```

### View Audit Logs
```bash
curl http://localhost:8000/audit/logs
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in .env file
   API_PORT=8001
   ```

2. **Browser Issues**
   ```bash
   # Install Playwright browsers
   playwright install
   ```

3. **Database Issues**
   ```bash
   # Reset database
   rm -rf data/
   python main.py
   ```

### Logs
```bash
# View logs
tail -f data/logs/automation.log
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN playwright install

EXPOSE 8000
CMD ["python", "main.py"]
```

### Environment Variables
```env
ENVIRONMENT=production
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000
```

## 📚 Next Steps

1. **Explore API Documentation**: http://localhost:8000/docs
2. **Run Demo**: `python demo_complex_workflow.py`
3. **Customize Workflows**: Modify workflow parameters
4. **Add AI Providers**: Configure API keys for enhanced capabilities
5. **Scale Deployment**: Deploy to production environment

## 🎉 Success!

Your Autonomous Multi-Agent Automation Platform is now running and ready to handle ultra-complex workflows with 100% real-time data processing!

**Key Features Active:**
- ✅ Multi-agent orchestration
- ✅ Real-time data processing
- ✅ Self-healing capabilities
- ✅ Enterprise compliance
- ✅ Conversational AI
- ✅ Cross-domain automation