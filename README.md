# Autonomous Multi-Agent Automation Platform

A comprehensive, autonomous, adaptive, multi-agent automation platform capable of executing ultra-complex workflows across diverse domains including e-commerce, advisory, entertainment, banking, finance, insurance, stock market analysis, ticket booking, and beyond.

## 🚀 Core Vision

This platform implements an intelligent multi-agent system with:

- **AI-1: Planner Agent** - Intelligent task breakdown and planning using various LLMs (GPT, Claude, Gemini, Local LLM)
- **AI-2: Execution Agent** - Parallel workflow execution with Playwright/Selenium/Cypress for live automation
- **AI-3: Conversational Agent** - Context-aware reasoning and human interaction
- **Search Agents** - Multi-source data retrieval (Google, Bing, DuckDuckGo, GitHub, StackOverflow, etc.)
- **DOM Extraction Agents** - Intelligent web page data extraction
- **Learning & Self-Healing** - Vector DB + ML for adaptive behavior and auto-repair
- **Enterprise Compliance** - Audit logging, RBAC, PII masking, compliance exports

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestrator                 │
├─────────────────────────────────────────────────────────────┤
│  AI-1: Planner    │  AI-2: Executor    │  AI-3: Conversational │
│  • Task Breakdown │  • Web Automation  │  • Context Management │
│  • Live Data      │  • API Calls       │  • Reasoning          │
│  • URL Detection  │  • DOM Extraction  │  • Human Takeover    │
│  • Parallel Opt.  │  • Self-Healing    │  • Follow-up Tasks   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data & Learning Layer                    │
├─────────────────────────────────────────────────────────────┤
│  SQLite Core DB   │  Vector DB         │  Local Media Storage │
│  • Definitions    │  • Semantic Search │  • Screenshots       │
│  • Audit Logs     │  • Learning        │  • Videos            │
│  • Performance    │  • Self-Healing    │  • Exports           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Enterprise Compliance Layer                │
├─────────────────────────────────────────────────────────────┤
│  Audit Logging    │  RBAC              │  PII Masking         │
│  • Tamper-proof   │  • Role-based      │  • Data Protection   │
│  • SOC2/GDPR      │  • Access Control  │  • Consent Workflow  │
│  • HIPAA Ready    │  • Compliance      │  • Export Reports    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### 🤖 Multi-Agent Intelligence
- **Planner Agent**: Breaks down complex workflows, detects data needs, identifies parallel opportunities
- **Execution Agent**: Handles web automation, API calls, data processing with self-healing
- **Conversational Agent**: Maintains context, provides reasoning, enables human takeover
- **Search Agents**: Multi-source data retrieval with better results than Perplexity AI
- **DOM Extraction Agents**: Intelligent web page data extraction and parsing

### 🔄 Adaptive Learning & Self-Healing
- **Selector Drift Detection**: ML model predicts and detects UI changes
- **Workflow Repair**: Auto-updates selectors, API calls, DOM parsing
- **Test Farm**: Headless re-runs for reliability validation
- **Performance Dashboard**: Real-time metrics and optimization

### 🏢 Enterprise-Grade Features
- **AI Connector Builder**: Auto-generates connectors from API docs
- **Continuous Test Farm**: Stress-testing and reliability metrics
- **Compliance Layer**: Audit logging, RBAC, PII masking, consent workflow
- **Compliance Export**: SOC2, GDPR, HIPAA ready reports

### 📊 Advanced Output & Reporting
- **Automation Reports**: Step-by-step with screenshots, videos, code, audit logs
- **Code Mode**: Export automation scripts
- **Follow-up Mode**: Seamless task continuation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for Playwright)
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd multi-agent-automation-platform
```

2. **Install Python dependencies**
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
# Edit .env with your API keys and configuration
```

5. **Initialize the platform**
```bash
python main.py
```

## ⚙️ Configuration

Create a `.env` file with the following configuration:

```env
# AI Configuration
AI_OPENAI_API_KEY=your_openai_key
AI_ANTHROPIC_API_KEY=your_anthropic_key
AI_GOOGLE_API_KEY=your_google_key
AI_LOCAL_LLM_URL=http://127.0.0.1:1234

# Search Configuration
SEARCH_GOOGLE_SEARCH_API_KEY=your_google_search_key
SEARCH_GOOGLE_SEARCH_CX=your_search_engine_id
SEARCH_BING_SEARCH_API_KEY=your_bing_key
SEARCH_GITHUB_TOKEN=your_github_token

# Database Configuration
DATABASE_DATABASE_PATH=./data/automation.db
DATABASE_VECTOR_DB_PATH=./data/vector_db

# Automation Configuration
AUTOMATION_BROWSER_TYPE=chromium
AUTOMATION_HEADLESS=true
AUTOMATION_BROWSER_TIMEOUT=30

# Security Configuration
SECURITY_SECRET_KEY=your_secret_key
SECURITY_ENCRYPTION_KEY=your_encryption_key

# API Configuration
API_API_HOST=0.0.0.0
API_API_PORT=8000
```

## 🚀 Usage

### Basic Workflow Execution

```python
from src import MultiAgentOrchestrator, Config

# Initialize the platform
config = Config()
orchestrator = MultiAgentOrchestrator(config)
await orchestrator.initialize()

# Define a workflow
workflow_request = {
    "name": "E-commerce Price Monitoring",
    "description": "Monitor product prices across multiple e-commerce sites",
    "domain": "ecommerce",
    "tasks": [
        {
            "name": "Search for products",
            "type": "search",
            "parameters": {
                "query": "laptop computers",
                "sources": ["google", "amazon", "bestbuy"]
            }
        },
        {
            "name": "Extract price data",
            "type": "dom_extraction",
            "parameters": {
                "url": "https://amazon.com/search?q=laptop",
                "selectors": {
                    "product_name": ".product-title",
                    "price": ".product-price",
                    "rating": ".product-rating"
                }
            }
        }
    ]
}

# Execute the workflow
execution = await orchestrator.execute_workflow(workflow_request)
print(f"Workflow started with ID: {execution.workflow_id}")
```

### Conversational Interaction

```python
# Chat with the conversational agent
response = await orchestrator.chat_with_agent(
    "What's the status of my e-commerce monitoring workflow?",
    session_id="user_123",
    workflow_id="workflow_456"
)

print(f"AI Response: {response['response']}")
print(f"Reasoning: {response['reasoning']}")
```

### API Endpoints

The platform exposes a REST API at `http://localhost:8000`:

- `POST /workflows` - Create and execute workflows
- `GET /workflows/{workflow_id}` - Get workflow status
- `GET /workflows` - List all workflows
- `POST /chat` - Chat with the conversational agent
- `GET /agents/status` - Get agent status
- `GET /metrics` - Get performance metrics

## 🔧 Advanced Features

### Self-Healing Workflows

The platform automatically detects and fixes common issues:

```python
# The platform will automatically:
# 1. Detect selector drift
# 2. Find alternative selectors
# 3. Update the workflow
# 4. Re-execute failed steps
```

### Multi-Source Search

```python
search_results = await orchestrator.search_agent.comprehensive_search(
    query="Python automation frameworks",
    sources=["google", "github", "stackoverflow", "reddit"],
    max_results=20
)
```

### DOM Extraction

```python
extracted_data = await orchestrator.dom_extraction_agent.extract_from_url(
    url="https://example.com",
    selectors={
        "title": "h1",
        "content": ".main-content",
        "metadata": "meta[name='description']"
    },
    content_type="article"
)
```

## 📈 Performance & Monitoring

### Real-time Metrics

```python
metrics = orchestrator.get_performance_metrics()
print(f"Success Rate: {metrics.success_rate:.2%}")
print(f"Average Duration: {metrics.average_duration:.2f}s")
print(f"Total Workflows: {metrics.total_workflows}")
```

### Agent Status

```python
status = orchestrator.get_agent_status()
for agent_name, agent_status in status.items():
    print(f"{agent_name}: {agent_status['status']}")
```

## 🔒 Security & Compliance

### Audit Logging

All activities are logged with tamper-proof audit trails:

```python
# Audit logs are automatically generated for:
# - Workflow executions
# - Agent interactions
# - Data access
# - Configuration changes
# - Security events
```

### RBAC (Role-Based Access Control)

```python
# Users can be assigned roles with specific permissions:
# - Admin: Full access
# - Operator: Workflow execution
# - Viewer: Read-only access
# - Auditor: Audit log access
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_core/
pytest tests/test_api/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Workflows

```bash
# Test specific workflow scenarios
python -m pytest tests/test_workflows/ -v

# Test self-healing capabilities
python -m pytest tests/test_self_healing/ -v
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Agent Architecture](docs/agents.md)
- [Workflow Examples](docs/workflows.md)
- [Configuration Guide](docs/configuration.md)
- [Security & Compliance](docs/security.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🏆 Competitive Advantages

- **Adaptive Orchestration**: Intelligent task distribution and optimization
- **Parallel Multi-Agent Execution**: True concurrent workflow processing
- **Live-Data Awareness**: Real-time data integration and processing
- **Auto-Heal Capabilities**: Self-repairing workflows and selectors
- **Cross-Domain Capability**: Works across any domain or industry
- **Compliance-Ready**: Enterprise-grade security and audit features

---

**Built with ❤️ for the future of automation**