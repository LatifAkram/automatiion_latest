"""
Multi-Agent Orchestrator
========================

The central brain that coordinates all AI agents:
- AI-1: Planner Agent (Brain)
- AI-2: Execution Agents (Automation)
- AI-3: Conversational Agent (Reasoning & Context)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

from .config import Config
from .database import DatabaseManager
from .vector_store import VectorStore
from .audit import AuditLogger

from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutionAgent
from ..agents.conversational import ConversationalAgent
from ..agents.search import SearchAgent
from ..agents.dom_extractor import DOMExtractionAgent
from ..agents.advanced_automation_capabilities import AdvancedAutomationCapabilities
from ..agents.intelligent_automation import IntelligentAutomationAgent
from ..agents.parallel_executor import ParallelExecutor
from ..agents.sector_specialists import SectorManager
from ..agents.conversational_ai import ConversationalAI
from ..utils.media_capture import MediaCapture
from ..utils.selector_drift import SelectorDriftDetector
from ..utils.code_generator import CodeGenerator

from ..models.workflow import Workflow, WorkflowStep, WorkflowStatus
from ..models.task import Task, TaskStatus, TaskType
from ..models.execution import ExecutionResult, ExecutionLog


class MultiAgentOrchestrator:
    """
    Main orchestrator that coordinates all AI agents and manages workflow execution.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.database = DatabaseManager(config.database)
        self.vector_store = VectorStore(config.database)
        self.audit_logger = AuditLogger(config)
        
        # AI Provider
        from ..core.ai_provider import AIProvider
        self.ai_provider = AIProvider(config.ai)
        
        # Initialize advanced automation capabilities
        self.advanced_capabilities = AdvancedAutomationCapabilities()
        
        # Initialize agents
        self.planner_agent = PlannerAgent(config.ai, self.vector_store, self.audit_logger)
        self.execution_agent = ExecutionAgent(config.automation, self.media_capture, SelectorDriftDetector(config.automation), self.audit_logger)
        self.conversational_ai = ConversationalAI(config, self.ai_provider)
        self.parallel_executor = ParallelExecutor(config, self.ai_provider)
        self.sector_manager = SectorManager(config, self.ai_provider)
        self.code_generator = CodeGenerator(config, self.ai_provider)
        self.intelligent_automation_agent = IntelligentAutomationAgent(config, self.ai_provider)
        self.search_agent = SearchAgent(config, self.ai_provider)
        
        # Initialize media capture
        if hasattr(config, 'database') and hasattr(config.database, 'media_path'):
            media_path = config.database.media_path
        else:
            media_path = 'data/media'
        self.media_capture = MediaCapture(media_path)
        
        # Workflow management
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_semaphore = asyncio.Semaphore(config.automation.max_parallel_workflows)
        self.agent_semaphore = asyncio.Semaphore(config.automation.max_parallel_agents)
        
        # Learning and optimization
        self.performance_metrics: Dict[str, Any] = {}
        self.self_healing_enabled = True
        
    async def initialize(self):
        """Initialize all components and agents."""
        self.logger.info("Initializing Multi-Agent Orchestrator...")
        
        # Initialize core components
        await self.database.initialize()
        await self.vector_store.initialize()
        await self.audit_logger.initialize()
        await self.ai_provider.initialize()
        
        # Initialize AI agents
        await self._initialize_agents()
        
        # Load existing workflows and metrics
        await self._load_existing_workflows()
        await self._load_performance_metrics()
        
        self.logger.info("Multi-Agent Orchestrator initialized successfully")
        
    async def _initialize_agents(self):
        """Initialize all AI agents."""
        # Initialize Planner Agent (AI-1: Brain)
        self.planner_agent = PlannerAgent(
            config=self.config.ai,
            vector_store=self.vector_store,
            audit_logger=self.audit_logger
        )
        await self.planner_agent.initialize()
        
        # Initialize Execution Agents (AI-2: Automation)
        for i in range(self.config.automation.max_parallel_agents):
            # Create media capture and selector drift detector instances
            media_capture = MediaCapture(self.config.database.media_path)
            await media_capture.initialize()
            
            selector_drift_detector = SelectorDriftDetector(self.config.automation)
            await selector_drift_detector.initialize()
            
            execution_agent = ExecutionAgent(
                config=self.config.automation,
                media_capture=media_capture,
                selector_drift_detector=selector_drift_detector,
                audit_logger=self.audit_logger
            )
            await execution_agent.initialize()
            self.execution_agents.append(execution_agent)
        
        # Initialize Conversational Agent (AI-3: Reasoning & Context)
        self.conversational_agent = ConversationalAgent(
            config=self.config.ai,
            vector_store=self.vector_store,
            audit_logger=self.audit_logger
        )
        await self.conversational_agent.initialize()
        
        # Initialize Search Agent
        self.search_agent = SearchAgent(
            config=self.config,
            audit_logger=self.audit_logger
        )
        await self.search_agent.initialize()
        
        # Initialize DOM Extraction Agent
        self.dom_extractor_agent = DOMExtractionAgent(
            config=self.config.automation,
            audit_logger=self.audit_logger
        )
        await self.dom_extractor_agent.initialize()
        
    @property
    def dom_extractor(self):
        """Get the DOM extractor agent."""
        return self.dom_extractor_agent
    
    @property
    def execution_agent(self):
        """Get the first available execution agent."""
        return self.execution_agents[0] if self.execution_agents else None
        
    async def _load_existing_workflows(self):
        """Load existing workflows from database."""
        workflows = await self.database.get_active_workflows()
        for workflow in workflows:
            self.active_workflows[workflow.id] = workflow
            
    async def _load_performance_metrics(self):
        """Load performance metrics from database."""
        self.performance_metrics = await self.database.get_performance_metrics()
        
    async def execute_workflow(self, workflow_request: Dict[str, Any]) -> str:
        """
        Execute a new workflow using the multi-agent system.
        
        Args:
            workflow_request: Dictionary containing workflow definition
            
        Returns:
            workflow_id: Unique identifier for the workflow
        """
        workflow_id = str(uuid4())
        
        async with self.workflow_semaphore:
            try:
                # Create workflow object
                workflow = Workflow(
                    id=workflow_id,
                    name=workflow_request.get("name", "Unnamed Workflow"),
                    description=workflow_request.get("description", ""),
                    domain=workflow_request.get("domain", "general"),
                    status=WorkflowStatus.PLANNING,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                # Store workflow
                await self.database.save_workflow(workflow)
                self.active_workflows[workflow_id] = workflow
                
                # Start workflow execution
                asyncio.create_task(self._execute_workflow_async(workflow_id, workflow_request))
                
                self.logger.info(f"Started workflow execution: {workflow_id}")
                return workflow_id
                
            except Exception as e:
                self.logger.error(f"Failed to start workflow: {e}", exc_info=True)
                raise
                
    async def _execute_workflow_async(self, workflow_id: str, workflow_request: Dict[str, Any]):
        """Execute workflow asynchronously."""
        try:
            # Step 1: Planning Phase (AI-1: Planner Agent)
            await self._update_workflow_status(workflow_id, WorkflowStatus.PLANNING)
            plan = await self._plan_workflow(workflow_id, workflow_request)
            
            # Step 2: Execution Phase (AI-2: Execution Agents)
            await self._update_workflow_status(workflow_id, WorkflowStatus.EXECUTING)
            execution_result = await self._execute_plan(workflow_id, plan)
            
            # Step 3: Learning and Optimization
            await self._learn_from_execution(workflow_id, execution_result)
            
            # Step 4: Update final status
            final_status = WorkflowStatus.COMPLETED if execution_result.success else WorkflowStatus.FAILED
            await self._update_workflow_status(workflow_id, final_status)
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            await self._update_workflow_status(workflow_id, WorkflowStatus.FAILED)
            
    async def _plan_workflow(self, workflow_id: str, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI-1: Planner Agent to break down workflow into tasks."""
        self.logger.info(f"Planning workflow: {workflow_id}")
        
        # Get workflow context
        workflow = self.active_workflows[workflow_id]
        
        # Use planner agent to create execution plan
        plan = await self.planner_agent.create_plan(
            workflow_request=workflow_request,
            workflow_context=workflow,
            performance_metrics=self.performance_metrics
        )
        
        # Store plan in database
        await self.database.save_workflow_plan(workflow_id, plan)
        
        return plan
        
    async def _execute_plan(self, workflow_id: str, plan: Dict[str, Any]) -> ExecutionResult:
        """Execute the plan using AI-2: Execution Agents."""
        self.logger.info(f"Executing plan for workflow: {workflow_id}")
        
        workflow = self.active_workflows[workflow_id]
        execution_log = ExecutionLog(workflow_id=workflow_id)
        
        try:
            # Execute tasks in parallel where possible
            tasks = plan.get("tasks", [])
            execution_tasks = []
            
            for task in tasks:
                if task.get("parallel", False):
                    # Execute in parallel
                    execution_task = asyncio.create_task(
                        self._execute_task_async(workflow_id, task)
                    )
                    execution_tasks.append(execution_task)
                else:
                    # Execute sequentially
                    result = await self._execute_task_async(workflow_id, task)
                    execution_log.add_step(result)
            
            # Wait for parallel tasks to complete
            if execution_tasks:
                parallel_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                for result in parallel_results:
                    if isinstance(result, Exception):
                        execution_log.add_error(str(result))
                    else:
                        execution_log.add_step(result)
            
            # Create execution result
            execution_result = ExecutionResult(
                workflow_id=workflow_id,
                success=execution_log.is_successful(),
                steps=execution_log.steps,
                errors=execution_log.errors,
                duration=execution_log.duration,
                created_at=datetime.utcnow()
            )
            
            # Store execution result
            await self.database.save_execution_result(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}", exc_info=True)
            return ExecutionResult(
                workflow_id=workflow_id,
                success=False,
                errors=[str(e)],
                created_at=datetime.utcnow()
            )
            
    async def _execute_task_async(self, workflow_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task using an available execution agent."""
        async with self.agent_semaphore:
            # Get available execution agent
            execution_agent = await self._get_available_execution_agent()
            
            try:
                # Execute task
                result = await execution_agent.execute_task(task)
                
                # Log execution
                await self.audit_logger.log_task_execution(
                    workflow_id=workflow_id,
                    task=task,
                    result=result
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "task_id": task.get("id"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
    async def _get_available_execution_agent(self) -> ExecutionAgent:
        """Get an available execution agent."""
        # Simple round-robin for now, could be enhanced with load balancing
        for agent in self.execution_agents:
            if not agent.is_busy():
                return agent
        
        # If all agents are busy, wait for one to become available
        while True:
            for agent in self.execution_agents:
                if not agent.is_busy():
                    return agent
            await asyncio.sleep(0.1)
            
    async def _learn_from_execution(self, workflow_id: str, execution_result: ExecutionResult):
        """Learn from execution results and update performance metrics."""
        if not self.self_healing_enabled:
            return
            
        try:
            # Update vector store with execution patterns
            await self.vector_store.store_execution_pattern(
                workflow_id=workflow_id,
                execution_result=execution_result
            )
            
            # Update performance metrics
            await self._update_performance_metrics(workflow_id, execution_result)
            
            # Trigger self-healing if needed
            if not execution_result.success:
                await self._trigger_self_healing(workflow_id, execution_result)
                
        except Exception as e:
            self.logger.error(f"Learning from execution failed: {e}", exc_info=True)
            
    async def _update_performance_metrics(self, workflow_id: str, execution_result: ExecutionResult):
        """Update performance metrics based on execution results."""
        metrics_key = f"workflow_{workflow_id}"
        
        self.performance_metrics[metrics_key] = {
            "success_rate": 1.0 if execution_result.success else 0.0,
            "duration": execution_result.duration,
            "step_count": len(execution_result.steps),
            "error_count": len(execution_result.errors),
            "last_execution": datetime.utcnow().isoformat()
        }
        
        # Store metrics in database
        await self.database.save_performance_metrics(metrics_key, self.performance_metrics[metrics_key])
        
    async def _trigger_self_healing(self, workflow_id: str, execution_result: ExecutionResult):
        """Trigger self-healing mechanisms for failed workflows."""
        self.logger.info(f"Triggering self-healing for workflow: {workflow_id}")
        
        # Analyze failure patterns
        failure_analysis = await self._analyze_failure_patterns(execution_result)
        
        # Update workflow with fixes
        if failure_analysis.get("fixes"):
            await self._apply_workflow_fixes(workflow_id, failure_analysis["fixes"])
            
    async def _analyze_failure_patterns(self, execution_result: ExecutionResult) -> Dict[str, Any]:
        """Analyze failure patterns to identify fixes."""
        # Use vector store to find similar failure patterns
        similar_failures = await self.vector_store.find_similar_failures(execution_result)
        
        # Use AI to suggest fixes
        fixes = await self.planner_agent.suggest_fixes(
            execution_result=execution_result,
            similar_failures=similar_failures
        )
        
        return {
            "similar_failures": similar_failures,
            "fixes": fixes
        }
        
    async def _apply_workflow_fixes(self, workflow_id: str, fixes: List[Dict[str, Any]]):
        """Apply suggested fixes to the workflow."""
        workflow = self.active_workflows[workflow_id]
        
        for fix in fixes:
            # Apply fix based on type
            if fix["type"] == "selector_update":
                await self._update_selectors(workflow_id, fix["data"])
            elif fix["type"] == "api_update":
                await self._update_api_calls(workflow_id, fix["data"])
            elif fix["type"] == "workflow_restructure":
                await self._restructure_workflow(workflow_id, fix["data"])
                
    async def _update_workflow_status(self, workflow_id: str, status: WorkflowStatus):
        """Update workflow status in database and memory."""
        workflow = self.active_workflows[workflow_id]
        workflow.status = status
        workflow.updated_at = datetime.utcnow()
        
        await self.database.update_workflow_status(workflow_id, status)
        
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return None
            
        workflow = self.active_workflows[workflow_id]
        execution_result = await self.database.get_latest_execution_result(workflow_id)
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "execution_result": execution_result.to_dict() if execution_result else None
        }
        
    async def chat_with_agent(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Chat with AI-3: Conversational Agent."""
        if not self.conversational_agent:
            raise RuntimeError("Conversational agent not initialized")
            
        response = await self.conversational_agent.chat(
            message=message,
            context=context,
            performance_metrics=self.performance_metrics
        )
        
        return response

    async def analyze_automation_requirements(self, user_request: str) -> Dict[str, Any]:
        """Analyze user request using advanced automation capabilities"""
        try:
            # Use the advanced capabilities to analyze requirements
            analysis = await self.advanced_capabilities.analyze_automation_requirements(user_request)
            
            # Get detailed capability information
            capabilities_info = []
            for capability_name in analysis.get("required_capabilities", []):
                capability = await self.advanced_capabilities.get_capability(capability_name)
                if capability:
                    capabilities_info.append({
                        "name": capability.name,
                        "description": capability.description,
                        "category": capability.category.value,
                        "complexity": capability.complexity,
                        "examples": capability.examples,
                        "selectors": capability.selectors,
                        "validation_rules": capability.validation_rules
                    })
            
            analysis["capabilities_info"] = capabilities_info
            
            # Log the analysis
            self.logger.info(f"Automation requirements analyzed: {analysis}")
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze automation requirements: {e}")
            return {
                "required_capabilities": ["click_operations", "type_operations"],
                "complexity_level": "simple",
                "estimated_duration": 300,
                "risk_level": "low",
                "compliance_requirements": [],
                "technical_requirements": [],
                "capabilities_info": []
            }

    async def generate_comprehensive_automation_plan(self, user_request: str) -> Dict[str, Any]:
        """Generate comprehensive automation plan using advanced capabilities"""
        try:
            # Generate plan using advanced capabilities
            plan = await self.advanced_capabilities.generate_automation_plan(user_request)
            
            # Validate the plan
            validation = await self.advanced_capabilities.validate_automation_plan(plan)
            plan["validation"] = validation
            
            # Add orchestrator-specific enhancements
            plan["orchestrator_info"] = {
                "agents_required": self._determine_required_agents(plan),
                "parallel_execution": self._can_execute_parallel(plan),
                "estimated_resources": self._estimate_resources(plan),
                "fallback_strategies": self._generate_fallback_strategies(plan)
            }
            
            self.logger.info(f"Comprehensive automation plan generated: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive automation plan: {e}")
            return {
                "analysis": {
                    "required_capabilities": ["click_operations", "type_operations"],
                    "complexity_level": "simple",
                    "estimated_duration": 300,
                    "risk_level": "low"
                },
                "steps": [],
                "capabilities_used": [],
                "estimated_completion_time": 300,
                "risk_assessment": "low",
                "compliance_checklist": [],
                "validation": {"is_feasible": True, "warnings": [], "risks": [], "recommendations": []},
                "orchestrator_info": {
                    "agents_required": ["executor"],
                    "parallel_execution": False,
                    "estimated_resources": {"cpu": 10, "memory": 20},
                    "fallback_strategies": ["manual_intervention"]
                }
            }

    def _determine_required_agents(self, plan: Dict[str, Any]) -> List[str]:
        """Determine which agents are required for the automation plan"""
        required_agents = []
        
        capabilities = plan.get("capabilities_used", [])
        for capability in capabilities:
            if hasattr(capability, 'category'):
                category = capability.category.value
                if category == "data_extraction":
                    required_agents.extend(["search", "dom_analysis"])
                elif category == "file_operations":
                    required_agents.extend(["executor"])
                elif category == "form_handling":
                    required_agents.extend(["executor", "intelligent_automation"])
                elif category == "api_integration":
                    required_agents.extend(["executor", "parallel_executor"])
                elif category in ["ecommerce", "banking", "healthcare", "education", "government"]:
                    required_agents.extend(["sector_manager", "executor"])
        
        # Always include planner and conversational agents
        required_agents.extend(["planner", "conversational"])
        
        # Remove duplicates
        return list(set(required_agents))

    def _can_execute_parallel(self, plan: Dict[str, Any]) -> bool:
        """Determine if the plan can be executed in parallel"""
        capabilities = plan.get("capabilities_used", [])
        return len(capabilities) > 1 and plan.get("analysis", {}).get("complexity_level") != "simple"

    def _estimate_resources(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for the automation"""
        complexity = plan.get("analysis", {}).get("complexity_level", "simple")
        duration = plan.get("estimated_completion_time", 300)
        
        base_resources = {
            "simple": {"cpu": 10, "memory": 20, "network": 5},
            "medium": {"cpu": 25, "memory": 40, "network": 15},
            "complex": {"cpu": 50, "memory": 80, "network": 30}
        }
        
        resources = base_resources.get(complexity, base_resources["simple"])
        
        # Scale based on duration
        if duration > 1800:  # More than 30 minutes
            resources = {k: v * 1.5 for k, v in resources.items()}
        elif duration > 3600:  # More than 1 hour
            resources = {k: v * 2 for k, v in resources.items()}
        
        return resources

    def _generate_fallback_strategies(self, plan: Dict[str, Any]) -> List[str]:
        """Generate fallback strategies for the automation plan"""
        strategies = []
        
        complexity = plan.get("analysis", {}).get("complexity_level", "simple")
        risk_level = plan.get("analysis", {}).get("risk_level", "low")
        
        if complexity == "complex":
            strategies.append("manual_intervention")
            strategies.append("partial_automation")
        
        if risk_level == "high":
            strategies.append("step_by_step_execution")
            strategies.append("human_approval_required")
        
        capabilities = plan.get("capabilities_used", [])
        for capability in capabilities:
            if hasattr(capability, 'category'):
                category = capability.category.value
                if category == "banking":
                    strategies.append("security_validation")
                elif category == "healthcare":
                    strategies.append("hipaa_compliance_check")
                elif category == "government":
                    strategies.append("compliance_verification")
        
        return strategies
        
    async def shutdown(self):
        """Shutdown the orchestrator and all agents."""
        self.logger.info("Shutting down Multi-Agent Orchestrator...")
        
        # Shutdown all agents
        if self.planner_agent:
            await self.planner_agent.shutdown()
            
        for agent in self.execution_agents:
            await agent.shutdown()
            
        if self.conversational_agent:
            await self.conversational_agent.shutdown()
            
        if self.search_agent:
            await self.search_agent.shutdown()
            
        if self.dom_extractor_agent:
            await self.dom_extractor_agent.shutdown()
            
        # Close database connections
        await self.database.close()
        await self.vector_store.shutdown()
        await self.audit_logger.shutdown()
        
        self.logger.info("Multi-Agent Orchestrator shutdown complete")


# Global orchestrator instance
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


def set_orchestrator(orchestrator: MultiAgentOrchestrator):
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator