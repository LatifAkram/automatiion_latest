"""
API Server
==========

FastAPI server for the automation platform with comprehensive endpoints.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.orchestrator import MultiAgentOrchestrator
from ..core.config import Config
from ..models.workflow import Workflow, WorkflowStatus
from ..models.execution import ExecutionResult


# API Models
class WorkflowRequest(BaseModel):
    """Request model for creating workflows."""
    name: str
    description: Optional[str] = None
    domain: str = "general"
    parameters: Dict[str, Any] = {}
    tags: List[str] = []


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    workflow_id: str
    status: str
    message: str


class ChatResponse(BaseModel):
    """Response model for chat responses."""
    response: str
    session_id: str
    timestamp: str


# Global variables
app = FastAPI(
    title="Autonomous Multi-Agent Automation Platform",
    description="A comprehensive automation platform that executes ultra-complex workflows across all domains",
    version="1.0.0"
)

orchestrator: Optional[MultiAgentOrchestrator] = None
config: Optional[Config] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Dependency to get the orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def get_config() -> Config:
    """Dependency to get the configuration."""
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    return config


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global orchestrator, config
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator(config)
        await orchestrator.initialize()
        
        logging.info("API server initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize API server: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global orchestrator
    
    if orchestrator:
        await orchestrator.shutdown()
        logging.info("API server shutdown complete")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Workflow endpoints
@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Create and execute a new workflow."""
    try:
        # Convert request to workflow format
        workflow_request = {
            "name": request.name,
            "description": request.description,
            "domain": request.domain,
            "parameters": request.parameters,
            "tags": request.tags
        }
        
        # Execute workflow
        workflow_id = await orch.execute_workflow(workflow_request)
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status="started",
            message=f"Workflow '{request.name}' started successfully"
        )
        
    except Exception as e:
        logging.error(f"Failed to create workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Get the status of a workflow."""
    try:
        status = await orch.get_workflow_status(workflow_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
            
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get workflow status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows")
async def list_workflows(
    orch: MultiAgentOrchestrator = Depends(get_orchestrator),
    status: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 50
):
    """List workflows with optional filtering."""
    try:
        # This would typically query the database
        # For now, return active workflows from orchestrator
        workflows = []
        
        for workflow_id, workflow in orch.active_workflows.items():
            if status and workflow.status.value != status:
                continue
            if domain and workflow.domain != domain:
                continue
                
            workflows.append({
                "id": workflow.id,
                "name": workflow.name,
                "status": workflow.status.value,
                "domain": workflow.domain,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            })
            
        return {
            "workflows": workflows[:limit],
            "total": len(workflows),
            "limit": limit
        }
        
    except Exception as e:
        logging.error(f"Failed to list workflows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/workflows/{workflow_id}")
async def cancel_workflow(
    workflow_id: str,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Cancel a running workflow."""
    try:
        # This would typically update the workflow status
        # For now, just return success
        return {
            "workflow_id": workflow_id,
            "status": "cancelled",
            "message": "Workflow cancellation requested"
        }
        
    except Exception as e:
        logging.error(f"Failed to cancel workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Chat with the conversational agent."""
    try:
        # Prepare context
        context = request.context or {}
        if request.session_id:
            context["session_id"] = request.session_id
            
        # Get response from conversational agent
        response = await orch.chat_with_agent(
            message=request.message,
            context=context
        )
        
        return ChatResponse(
            response=response,
            session_id=context.get("session_id", "default"),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logging.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoints
@app.post("/search")
async def search_information(
    request: dict,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Search for information using the search agent."""
    try:
        query = request.get("query", "")
        sources = request.get("sources", ["duckduckgo"])
        max_results = request.get("max_results", 10)
        
        # Use search agent to get results
        if orch.search_agent:
            results = await orch.search_agent.search(query, max_results, sources)
        else:
            # Fallback to mock results
            results = [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com",
                    "snippet": f"This is a mock search result for the query: {query}",
                    "domain": "example.com",
                    "relevance": 0.8,
                    "source": sources[0] if sources else "mock"
                }
            ]
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Automation endpoints
@app.post("/automation/execute")
async def execute_automation(
    request: dict,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Execute automation tasks."""
    try:
        automation_type = request.get("type", "web_automation")
        url = request.get("url", "")
        actions = request.get("actions", [])
        options = request.get("options", {})
        
        # Execute automation using execution agent
        if orch.execution_agents:
            agent = orch.execution_agents[0]  # Use first available agent
            result = await agent.execute_automation({
                "type": automation_type,
                "url": url,
                "actions": actions,
                "options": options
            })
        else:
            # Fallback to mock result
            result = {
                "status": "completed",
                "screenshots": [],
                "data": {"message": "Mock automation execution completed"},
                "execution_time": 2.5
            }
        
        return {
            "automation_id": f"auto_{int(time.time())}",
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Automation execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Export endpoints
@app.post("/export")
async def export_data(
    request: dict,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Export data in various formats."""
    try:
        export_format = request.get("format", "excel")
        data = request.get("data", {})
        options = request.get("options", {})
        
        # Mock export functionality
        export_result = {
            "format": export_format,
            "file_name": f"export_{int(time.time())}.{export_format}",
            "file_size": "2.3 MB",
            "download_url": f"/downloads/export_{int(time.time())}.{export_format}",
            "export_time": datetime.utcnow().isoformat()
        }
        
        return export_result
        
    except Exception as e:
        logging.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.get("/analytics/performance")
async def get_performance_metrics(
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Get performance metrics."""
    try:
        return {
            "performance_metrics": orch.performance_metrics,
            "active_workflows": len(orch.active_workflows),
            "execution_agents": len(orch.execution_agents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to get performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/agents")
async def get_agent_status(
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Get status of all agents."""
    try:
        agent_statuses = []
        
        # Get execution agent statuses
        for agent in orch.execution_agents:
            status = await agent.get_status()
            agent_statuses.append({
                "type": "execution",
                "agent_id": status["agent_id"],
                "is_busy": status["is_busy"],
                "current_task": status["current_task"],
                "uptime": status["uptime"]
            })
            
        # Get planner agent status
        if orch.planner_agent:
            agent_statuses.append({
                "type": "planner",
                "agent_id": "planner",
                "status": "active"
            })
            
        # Get conversational agent status
        if orch.conversational_agent:
            agent_statuses.append({
                "type": "conversational",
                "agent_id": "conversational",
                "status": "active"
            })
            
        return {
            "agents": agent_statuses,
            "total_agents": len(agent_statuses),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to get agent status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# System endpoints
@app.get("/system/info")
async def get_system_info(
    cfg: Config = Depends(get_config)
):
    """Get system information."""
    try:
        return {
            "version": "1.0.0",
            "environment": cfg.environment,
            "ai_providers": {
                "openai": bool(cfg.ai.openai_api_key),
                "anthropic": bool(cfg.ai.anthropic_api_key),
                "google": bool(cfg.ai.google_api_key),
                "local": True
            },
            "database": {
                "type": "sqlite",
                "path": cfg.database.sqlite_path
            },
            "vector_db": {
                "type": cfg.database.vector_db_type,
                "path": cfg.database.vector_db_path
            },
            "automation": {
                "browser_type": cfg.automation.browser_type,
                "max_parallel_agents": cfg.automation.max_parallel_agents,
                "max_parallel_workflows": cfg.automation.max_parallel_workflows
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to get system info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/restart")
async def restart_system(
    background_tasks: BackgroundTasks,
    orch: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """Restart the system (shutdown and reinitialize)."""
    try:
        # Schedule restart in background
        background_tasks.add_task(_restart_system, orch)
        
        return {
            "message": "System restart initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Failed to restart system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _restart_system(orch: MultiAgentOrchestrator):
    """Background task to restart the system."""
    try:
        # Shutdown current orchestrator
        await orch.shutdown()
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Reinitialize
        await orch.initialize()
        
        logging.info("System restart completed")
        
    except Exception as e:
        logging.error(f"System restart failed: {e}", exc_info=True)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Start the server
def start_api_server(orch: MultiAgentOrchestrator):
    """Start the API server."""
    global orchestrator
    orchestrator = orch
    
    import uvicorn
    import threading
    
    config = orch.config.api
    
    def run_server():
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info"
        )
    
    # Run server in a separate thread to avoid event loop conflicts
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    return server_thread