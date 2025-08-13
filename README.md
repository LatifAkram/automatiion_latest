# Autonomous Multi-Agent Automation Platform

A comprehensive, autonomous, adaptive, multi-agent automation platform capable of executing ultra-complex workflows across diverse domains including e-commerce, advisory, entertainment, banking, finance, insurance, stock market analysis, ticket booking, and beyond.

## 🚀 Core Vision

This platform implements a sophisticated multi-agent system with:

- **AI-1: Planner Agent** - Intelligent planning and task breakdown
- **AI-2: Execution Agents** - Parallel workflow execution with self-healing
- **AI-3: Conversational Agent** - Reasoning, context management, and human interaction
- **Vector Database** - Learning and pattern recognition
- **Enterprise Compliance** - Audit logging, PII detection, and governance

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI-1: Planner │    │ AI-2: Execution │    │ AI-3: Conversational │
│     Agent       │    │     Agents      │    │      Agent      │
│                 │    │                 │    │                 │
│ • Task Analysis │    │ • Web Automation│    │ • Context Mgmt  │
│ • Plan Creation │    │ • API Calls     │    │ • Reasoning     │
│ • Optimization  │    │ • Self-Healing  │    │ • Human Takeover│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │                 │
                    │ • Coordination  │
                    │ • Learning      │
                    │ • Optimization  │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Vector Store  │
                    │                 │
                    │ • Pattern Storage│
                    │ • Similarity Search│
                    │ • Learning      │
                    └─────────────────┘
```

### Core Components

1. **Planner Agent (AI-1)**
   - Breaks down complex workflows into executable tasks
   - Detects live data needs and spins up Search Agents
   - Identifies URLs for DOM Extraction Agents
   - Optimizes plans based on historical performance

2. **Execution Agents (AI-2)**
   - Uses Playwright/Selenium for web automation
   - Handles API calls, data processing, file operations
   - Implements selector drift detection and self-healing
   - Captures screenshots and videos for every step

3. **Conversational Agent (AI-3)**
   - Maintains context across sessions
   - Provides reasoning for decisions
   - Supports human takeover of workflow steps
   - Better than ChatGPT/Cursor AI for automation context

4. **Search Agents**
   - Google, Bing, DuckDuckGo search
   - GitHub, StackOverflow, API docs
   - News, Reddit, YouTube integration
   - Results better than Perplexity AI

5. **DOM Extraction Agents**
   - Intelligent web scraping
   - Structured data extraction
   - Form detection and handling
   - Visual similarity matching

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd autonomous-automation-platform
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**
   ```bash
   playwright install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Test the platform**
   ```bash
   python test_platform.py
   ```

6. **Start the platform**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

Create a `.env` file with the following configuration:

```env
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_PATH=./data

# AI Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
LOCAL_LLM_URL=http://127.0.0.1:1234

# Search APIs
GOOGLE_SEARCH_API_KEY=your_google_search_key
GOOGLE_SEARCH_CX=your_search_engine_id
BING_SEARCH_API_KEY=your_bing_key
GITHUB_TOKEN=your_github_token
STACK_OVERFLOW_KEY=your_stackoverflow_key

# Automation
BROWSER_TYPE=chromium
HEADLESS=true
MAX_PARALLEL_AGENTS=5
MAX_PARALLEL_WORKFLOWS=3

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

## 🚀 Usage

### Starting the Platform

```bash
python main.py
```

The platform will:
1. Initialize all AI agents
2. Set up the vector database
3. Start the FastAPI server
4. Begin listening for workflow requests

### API Endpoints

Once running, access the API at `http://localhost:8000`:

- **Health Check**: `GET /health`
- **Create Workflow**: `POST /workflows`
- **Get Workflow Status**: `GET /workflows/{workflow_id}`
- **List Workflows**: `GET /workflows`
- **Chat with Agent**: `POST /chat`
- **Performance Metrics**: `GET /analytics/performance`
- **Agent Status**: `GET /analytics/agents`
- **System Info**: `GET /system/info`

### Example Workflow Request

```python
import requests

# Create a workflow
workflow_request = {
    "name": "E-commerce Product Research",
    "description": "Research products on multiple e-commerce sites",
    "domain": "ecommerce",
    "parameters": {
        "product": "laptop",
        "budget": 1000,
        "sites": ["amazon", "bestbuy", "newegg"]
    },
    "tags": ["research", "ecommerce", "automation"]
}

response = requests.post("http://localhost:8000/workflows", json=workflow_request)
workflow_id = response.json()["workflow_id"]

# Check status
status = requests.get(f"http://localhost:8000/workflows/{workflow_id}")
print(status.json())
```

### Chat with the Agent

```python
# Chat with the conversational agent
chat_request = {
    "message": "What's the status of my workflow?",
    "session_id": "user123",
    "context": {"workflow_id": workflow_id}
}

response = requests.post("http://localhost:8000/chat", json=chat_request)
print(response.json()["response"])
```

## 🔧 Advanced Features

### Self-Healing Capabilities

The platform automatically detects and fixes common issues:

- **Selector Drift**: When web elements change, the system finds alternative selectors
- **API Failures**: Automatic retry with exponential backoff
- **Data Validation**: Ensures extracted data meets expected formats
- **Performance Optimization**: Learns from past executions to improve future runs

### Learning and Optimization

- **Vector Database**: Stores execution patterns for similarity search
- **Performance Metrics**: Tracks success rates, execution times, and failure patterns
- **Template Reuse**: Identifies and reuses successful workflow patterns
- **Continuous Improvement**: Automatically optimizes based on historical data

### Enterprise Compliance

- **Audit Logging**: Comprehensive logging of all activities
- **PII Detection**: Automatic detection and masking of sensitive data
- **RBAC**: Role-based access control for enterprise deployments
- **Compliance Reports**: SOC2, GDPR, HIPAA compliance reporting

## 📊 Monitoring and Analytics

### Performance Dashboard

Access real-time metrics at `http://localhost:8000/analytics/performance`:

```json
{
  "total_workflows": 150,
  "successful_workflows": 142,
  "failed_workflows": 8,
  "success_rate": 0.947,
  "avg_execution_time": 45.2,
  "active_agents": 3,
  "queue_size": 2
}
```

### Agent Status

Monitor individual agent health at `http://localhost:8000/analytics/agents`:

```json
{
  "agents": [
    {
      "type": "execution",
      "agent_id": "exec_001",
      "is_busy": false,
      "current_task": null,
      "uptime": 3600,
      "tasks_completed": 45
    }
  ]
}
```

## 🧪 Testing

### Run Platform Tests

```bash
python test_platform.py
```

### Run Unit Tests

```bash
pytest tests/
```

### Run Integration Tests

```bash
pytest tests/integration/
```

## 🔒 Security

### PII Detection

The platform automatically detects and masks:

- Email addresses
- Phone numbers
- Social Security numbers
- Credit card numbers
- IP addresses

### Audit Trail

All activities are logged with:

- Timestamp and user identification
- Action details and context
- Before/after state changes
- Compliance metadata

### Encryption

- Data at rest encryption
- Secure API communication
- Encrypted audit logs
- Secure credential storage

## 🚀 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t automation-platform .

# Run the container
docker run -p 8000:8000 -v ./data:/app/data automation-platform
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: automation-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: automation-platform
  template:
    metadata:
      labels:
        app: automation-platform
    spec:
      containers:
      - name: automation-platform
        image: automation-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🎯 Roadmap

### Phase 1: Core Platform ✅
- [x] Multi-agent orchestration
- [x] SQLite database integration
- [x] Vector database for learning
- [x] Live data agents (Search, DOM Extraction)
- [x] Basic self-healing capabilities

### Phase 2: Advanced Features 🚧
- [ ] AI Connector Builder
- [ ] Continuous Test Farm
- [ ] Advanced selector drift detection
- [ ] Performance optimization engine

### Phase 3: Enterprise Features 📋
- [ ] Advanced compliance layer
- [ ] Public connector ecosystem
- [ ] Enterprise orchestration scaling
- [ ] Multi-tenant support

---

**Built with ❤️ for the future of automation**