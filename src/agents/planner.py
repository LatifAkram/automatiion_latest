"""
AI-1: Planner Agent (Brain)
==========================

The intelligent planner that breaks down complex workflows into executable tasks.
Uses AI to analyze requirements, detect data needs, and create optimal execution plans.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from ..core.ai_provider import AIProvider
from ..core.vector_store import VectorStore
from ..core.audit import AuditLogger
from ..models.workflow import Workflow, WorkflowStep
from ..models.task import Task, TaskType, TaskStatus


class PlannerAgent:
    """
    AI-1: Planner Agent - The intelligent brain that plans and orchestrates workflows.
    """
    
    def __init__(self, config: Any, vector_store: VectorStore, audit_logger: AuditLogger):
        self.config = config
        self.vector_store = vector_store
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # AI provider for planning decisions
        self.ai_provider = AIProvider(config)
        
        # Planning templates and patterns
        self.planning_templates = self._load_planning_templates()
        
    async def initialize(self):
        """Initialize the planner agent."""
        await self.ai_provider.initialize()
        self.logger.info("Planner Agent initialized")
        
    def _load_planning_templates(self) -> Dict[str, Any]:
        """Load planning templates for different workflow types."""
        return {
            "ecommerce": {
                "data_needs": ["product_catalog", "pricing", "inventory", "user_preferences"],
                "common_tasks": ["search_products", "add_to_cart", "checkout", "payment"],
                "parallel_opportunities": ["product_search", "price_comparison", "review_analysis"]
            },
            "advisory": {
                "data_needs": ["market_data", "financial_reports", "news", "expert_opinions"],
                "common_tasks": ["data_collection", "analysis", "report_generation", "recommendations"],
                "parallel_opportunities": ["data_gathering", "analysis", "report_preparation"]
            },
            "banking": {
                "data_needs": ["account_data", "transaction_history", "market_rates", "regulations"],
                "common_tasks": ["account_check", "transaction_processing", "compliance_check"],
                "parallel_opportunities": ["data_validation", "risk_assessment", "compliance_verification"]
            },
            "general": {
                "data_needs": ["context", "requirements", "constraints"],
                "common_tasks": ["data_gathering", "processing", "output_generation"],
                "parallel_opportunities": ["research", "analysis", "synthesis"]
            }
        }
        
    async def create_plan(self, workflow_request: Dict[str, Any], 
                         workflow_context: Workflow, 
                         performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an intelligent execution plan for a workflow.
        
        Args:
            workflow_request: The original workflow request
            workflow_context: The workflow context
            performance_metrics: Historical performance data
            
        Returns:
            Detailed execution plan with tasks, dependencies, and optimizations
        """
        self.logger.info(f"Creating plan for workflow: {workflow_context.id}")
        
        try:
            # Step 1: Analyze workflow requirements
            analysis = await self._analyze_workflow_requirements(workflow_request, workflow_context)
            
            # Step 2: Detect data needs and search requirements
            data_needs = await self._detect_data_needs(analysis)
            
            # Step 3: Identify parallel execution opportunities
            parallel_ops = await self._identify_parallel_opportunities(analysis, data_needs)
            
            # Step 4: Generate task breakdown
            tasks = await self._generate_task_breakdown(analysis, data_needs, parallel_ops)
            
            # Step 5: Optimize plan based on performance metrics
            optimized_tasks = await self._optimize_plan(tasks, performance_metrics)
            
            # Step 6: Create execution plan
            plan = {
                "workflow_id": workflow_context.id,
                "analysis": analysis,
                "data_needs": data_needs,
                "parallel_opportunities": parallel_ops,
                "tasks": optimized_tasks,
                "estimated_duration": self._estimate_duration(optimized_tasks),
                "success_probability": self._estimate_success_probability(optimized_tasks, performance_metrics),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store plan in vector store for learning
            await self.vector_store.store_plan(workflow_context.id, plan)
            
            # Log planning activity
            await self.audit_logger.log_planning_activity(
                workflow_id=workflow_context.id,
                plan=plan
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}", exc_info=True)
            raise
            
    async def _analyze_workflow_requirements(self, workflow_request: Dict[str, Any], 
                                           workflow_context: Workflow) -> Dict[str, Any]:
        """Analyze workflow requirements using AI."""
        
        prompt = f"""
        Analyze the following workflow request and provide a detailed breakdown:
        
        Workflow Name: {workflow_context.name}
        Description: {workflow_context.description}
        Domain: {workflow_context.domain}
        
        Request Details:
        {json.dumps(workflow_request, indent=2)}
        
        Please provide:
        1. Primary objectives and goals
        2. Required data sources and APIs
        3. User interactions needed
        4. Expected outputs
        5. Potential challenges and risks
        6. Success criteria
        7. Estimated complexity (low/medium/high)
        
        Respond in JSON format.
        """
        
        response = await self.ai_provider.generate_response(prompt)
        
        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            # Fallback to structured analysis
            return self._fallback_analysis(workflow_request, workflow_context)
            
    async def _detect_data_needs(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect what data needs to be gathered for the workflow."""
        
        data_needs = []
        
        # Extract data requirements from analysis
        if "required_data_sources" in analysis:
            for data_source in analysis["required_data_sources"]:
                data_needs.append({
                    "type": "data_source",
                    "source": data_source,
                    "priority": "high",
                    "search_required": True
                })
                
        # Detect URLs that need DOM extraction
        if "urls" in analysis:
            for url in analysis["urls"]:
                data_needs.append({
                    "type": "dom_extraction",
                    "url": url,
                    "priority": "medium",
                    "extraction_required": True
                })
                
        # Detect API calls needed
        if "apis" in analysis:
            for api in analysis["apis"]:
                data_needs.append({
                    "type": "api_call",
                    "api": api,
                    "priority": "high",
                    "authentication_required": api.get("auth_required", False)
                })
                
        return data_needs
        
    async def _identify_parallel_opportunities(self, analysis: Dict[str, Any], 
                                             data_needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify tasks that can be executed in parallel."""
        
        parallel_ops = []
        
        # Data gathering can often be parallelized
        data_gathering_tasks = [need for need in data_needs if need.get("search_required")]
        if len(data_gathering_tasks) > 1:
            parallel_ops.append({
                "type": "parallel_data_gathering",
                "tasks": data_gathering_tasks,
                "estimated_savings": "50-70%"
            })
            
        # Analysis tasks can be parallelized if independent
        if "analysis_tasks" in analysis:
            independent_analyses = [task for task in analysis["analysis_tasks"] 
                                  if task.get("independent", False)]
            if len(independent_analyses) > 1:
                parallel_ops.append({
                    "type": "parallel_analysis",
                    "tasks": independent_analyses,
                    "estimated_savings": "30-50%"
                })
                
        return parallel_ops
        
    async def _generate_task_breakdown(self, analysis: Dict[str, Any], 
                                     data_needs: List[Dict[str, Any]], 
                                     parallel_ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed task breakdown using AI."""
        
        prompt = f"""
        Create a detailed task breakdown for the following workflow:
        
        Analysis: {json.dumps(analysis, indent=2)}
        Data Needs: {json.dumps(data_needs, indent=2)}
        Parallel Opportunities: {json.dumps(parallel_ops, indent=2)}
        
        For each task, provide:
        1. Task ID and name
        2. Task type (search, automation, api_call, dom_extraction, etc.)
        3. Dependencies (what must complete before this task)
        4. Parallel execution flag
        5. Estimated duration
        6. Success criteria
        7. Retry strategy
        8. Required resources
        
        Respond with a JSON array of tasks.
        """
        
        response = await self.ai_provider.generate_response(prompt)
        
        try:
            tasks = json.loads(response)
            return self._validate_and_enhance_tasks(tasks)
        except json.JSONDecodeError:
            return self._fallback_task_breakdown(analysis, data_needs)
            
    def _validate_and_enhance_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance task definitions."""
        
        enhanced_tasks = []
        
        for i, task in enumerate(tasks):
            # Ensure required fields
            enhanced_task = {
                "id": task.get("id", f"task_{i}"),
                "name": task.get("name", f"Task {i}"),
                "type": task.get("type", "general"),
                "dependencies": task.get("dependencies", []),
                "parallel": task.get("parallel", False),
                "estimated_duration": task.get("estimated_duration", 60),
                "success_criteria": task.get("success_criteria", "task_completed"),
                "retry_strategy": task.get("retry_strategy", {"max_retries": 3, "backoff": "exponential"}),
                "required_resources": task.get("required_resources", []),
                "priority": task.get("priority", "medium"),
                "timeout": task.get("timeout", 300),
                "created_at": datetime.utcnow().isoformat()
            }
            
            enhanced_tasks.append(enhanced_task)
            
        return enhanced_tasks
        
    async def _optimize_plan(self, tasks: List[Dict[str, Any]], 
                           performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize the plan based on historical performance data."""
        
        optimized_tasks = []
        
        for task in tasks:
            optimized_task = task.copy()
            
            # Adjust timeouts based on historical performance
            task_type = task.get("type")
            if task_type in performance_metrics:
                avg_duration = performance_metrics[task_type].get("avg_duration", 60)
                optimized_task["timeout"] = max(avg_duration * 2, 60)
                
            # Adjust retry strategy based on success rates
            if task_type in performance_metrics:
                success_rate = performance_metrics[task_type].get("success_rate", 0.8)
                if success_rate < 0.7:
                    optimized_task["retry_strategy"]["max_retries"] = 5
                elif success_rate > 0.95:
                    optimized_task["retry_strategy"]["max_retries"] = 1
                    
            # Prioritize tasks based on dependencies
            if not task.get("dependencies"):
                optimized_task["priority"] = "high"
                
            optimized_tasks.append(optimized_task)
            
        return optimized_tasks
        
    def _estimate_duration(self, tasks: List[Dict[str, Any]]) -> int:
        """Estimate total workflow duration."""
        total_duration = 0
        
        # Calculate critical path
        task_durations = {task["id"]: task["estimated_duration"] for task in tasks}
        
        for task in tasks:
            if not task.get("dependencies"):
                # No dependencies, can start immediately
                total_duration = max(total_duration, task["estimated_duration"])
            else:
                # Has dependencies, add to critical path
                dependency_duration = max(task_durations.get(dep, 0) for dep in task["dependencies"])
                total_duration = max(total_duration, dependency_duration + task["estimated_duration"])
                
        return total_duration
        
    def _estimate_success_probability(self, tasks: List[Dict[str, Any]], 
                                    performance_metrics: Dict[str, Any]) -> float:
        """Estimate overall success probability based on task success rates."""
        
        if not tasks:
            return 1.0
            
        total_probability = 1.0
        
        for task in tasks:
            task_type = task.get("type", "general")
            task_success_rate = performance_metrics.get(task_type, {}).get("success_rate", 0.8)
            
            # Account for retries
            max_retries = task.get("retry_strategy", {}).get("max_retries", 3)
            retry_success_rate = 1 - ((1 - task_success_rate) ** (max_retries + 1))
            
            total_probability *= retry_success_rate
            
        return total_probability
        
    async def suggest_fixes(self, execution_result: Any, 
                          similar_failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest fixes for failed workflows based on similar patterns."""
        
        prompt = f"""
        Analyze the following execution failure and suggest fixes:
        
        Execution Result: {json.dumps(execution_result.to_dict(), indent=2)}
        Similar Failures: {json.dumps(similar_failures, indent=2)}
        
        Suggest specific fixes including:
        1. Selector updates for UI automation
        2. API call modifications
        3. Workflow restructuring
        4. Data source alternatives
        
        Respond with a JSON array of fixes.
        """
        
        response = await self.ai_provider.generate_response(prompt)
        
        try:
            fixes = json.loads(response)
            return fixes
        except json.JSONDecodeError:
            return self._fallback_fixes(execution_result)
            
    def _fallback_analysis(self, workflow_request: Dict[str, Any], 
                          workflow_context: Workflow) -> Dict[str, Any]:
        """Fallback analysis when AI analysis fails."""
        return {
            "primary_objectives": ["Complete the requested workflow"],
            "required_data_sources": [],
            "user_interactions": [],
            "expected_outputs": ["Workflow completion"],
            "potential_challenges": ["Unknown complexity"],
            "success_criteria": ["Workflow completes successfully"],
            "estimated_complexity": "medium"
        }
        
    def _fallback_task_breakdown(self, analysis: Dict[str, Any], 
                                data_needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback task breakdown when AI generation fails."""
        return [
            {
                "id": "task_1",
                "name": "Execute Workflow",
                "type": "general",
                "dependencies": [],
                "parallel": False,
                "estimated_duration": 60,
                "success_criteria": "workflow_completed",
                "retry_strategy": {"max_retries": 3, "backoff": "exponential"},
                "required_resources": [],
                "priority": "high",
                "timeout": 300,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
    def _fallback_fixes(self, execution_result: Any) -> List[Dict[str, Any]]:
        """Fallback fixes when AI suggestion fails."""
        return [
            {
                "type": "general_fix",
                "description": "Retry with increased timeout",
                "data": {"timeout_multiplier": 2}
            }
        ]
        
    async def shutdown(self):
        """Shutdown the planner agent."""
        await self.ai_provider.shutdown()
        self.logger.info("Planner Agent shutdown complete")