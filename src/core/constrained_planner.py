"""
Constrained Planner
==================

AI planner that stays inside the rails using:
- Frontier LLM (GPT/Claude/Gemini) constrained by schemas
- DAG of Step objects with pre/postconditions, fallbacks, timeouts, retries
- Confidence gating: if planner < τ or simulator disagrees → ask micro-clarification or switch to skill
- DAG Execution Loop with parallel execution and drift handling

DAG Execution Loop:
plan = planner(goal, tools)
while not plan.done:
  for node in plan.ready_parallel():
    spawn(executor(node))
  for result in gather():
    if result.drift: healer.patch(result)
    if result.needs_live: realtime.fetch(...)
    plan.update(result)
  if plan.conf < τ: micro_prompt_user()
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from semantic_dom_graph import SemanticDOMGraph
from self_healing_locators import SelfHealingLocatorStack
from shadow_dom_simulator import ShadowDOMSimulator, SimulationResult
# Import contracts with fallback
try:
    from models.contracts import StepContract, Action, ActionType, TargetSelector
except ImportError:
    from enum import Enum
    from dataclasses import dataclass
    from typing import Any, Optional, Dict
    
    class ActionType(Enum):
        CLICK = "click"
        TYPE = "type"
        SCROLL = "scroll"
        WAIT = "wait"
        NAVIGATE = "navigate"
    
    @dataclass
    class Action:
        type: ActionType
        selector: str
        value: Optional[Any] = None
        
    @dataclass
    class TargetSelector:
        selector: str
        confidence: float = 1.0
        method: str = "css"
        
    @dataclass  
    class StepContract:
        action: Action
        preconditions: Dict[str, Any]
        postconditions: Dict[str, Any]


class PlanStatus(str, Enum):
    """Plan execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """DAG node execution status."""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanNode:
    """A node in the execution DAG."""
    id: str
    step: StepContract
    dependencies: Set[str]
    dependents: Set[str]
    status: NodeStatus = NodeStatus.WAITING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retries: int = 0
    
    def __post_init__(self):
        if isinstance(self.dependencies, list):
            self.dependencies = set(self.dependencies)
        if isinstance(self.dependents, list):
            self.dependents = set(self.dependents)


@dataclass
class ExecutionPlan:
    """Complete execution plan with DAG structure."""
    id: str
    goal: str
    nodes: Dict[str, PlanNode]
    status: PlanStatus = PlanStatus.PENDING
    confidence: float = 1.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.total_steps = len(self.nodes)
    
    @property
    def done(self) -> bool:
        """Check if plan execution is done."""
        return self.status in [PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED]
    
    def get_ready_nodes(self) -> List[PlanNode]:
        """Get nodes ready for parallel execution."""
        ready_nodes = []
        
        for node in self.nodes.values():
            if node.status == NodeStatus.WAITING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.nodes[dep_id].status == NodeStatus.COMPLETED 
                    for dep_id in node.dependencies 
                    if dep_id in self.nodes
                )
                
                if deps_completed:
                    node.status = NodeStatus.READY
                    ready_nodes.append(node)
        
        return ready_nodes
    
    def update_node_result(self, node_id: str, result: Dict[str, Any], success: bool = True):
        """Update node execution result."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.result = result
        node.end_time = datetime.utcnow()
        
        if success:
            node.status = NodeStatus.COMPLETED
            self.completed_steps += 1
        else:
            node.status = NodeStatus.FAILED
            node.error = result.get('error', 'Unknown error')
            self.failed_steps += 1
        
        # Update overall plan status
        if self.completed_steps == self.total_steps:
            self.status = PlanStatus.COMPLETED
            self.completed_at = datetime.utcnow()
        elif self.failed_steps > 0 and not self._has_pending_nodes():
            self.status = PlanStatus.FAILED
            self.completed_at = datetime.utcnow()
    
    def _has_pending_nodes(self) -> bool:
        """Check if there are still pending nodes to execute."""
        return any(
            node.status in [NodeStatus.WAITING, NodeStatus.READY, NodeStatus.RUNNING]
            for node in self.nodes.values()
        )


class ConstrainedPlanner:
    """
    AI planner constrained by schemas with DAG execution and confidence gating.
    
    Uses frontier LLM to generate plans, validates with simulator,
    and executes with self-healing capabilities.
    """
    
    def __init__(self, 
                 semantic_graph: SemanticDOMGraph,
                 locator_stack: SelfHealingLocatorStack,
                 simulator: ShadowDOMSimulator,
                 config: Any = None):
        self.semantic_graph = semantic_graph
        self.locator_stack = locator_stack
        self.simulator = simulator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self._init_llm_clients()
        
        # Configuration
        self.confidence_threshold = 0.85
        self.simulation_threshold = 0.98
        self.max_retries = 3
        self.max_parallel_nodes = 5
        
        # Execution state
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_history: List[ExecutionPlan] = []
        
        # Planning prompts
        self.system_prompt = self._build_system_prompt()
        
    def _init_llm_clients(self):
        """Initialize LLM clients."""
        try:
            if OPENAI_AVAILABLE and self.config and hasattr(self.config, 'openai_api_key'):
                self.openai_client = openai.OpenAI(api_key=self.config.openai_api_key)
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        try:
            if ANTHROPIC_AVAILABLE and self.config and hasattr(self.config, 'anthropic_api_key'):
                self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        except Exception as e:
            self.logger.warning(f"Failed to initialize Anthropic client: {e}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM planning."""
        return """You are a web automation planner that generates precise step-by-step plans.

CONSTRAINTS:
1. Output ONLY valid JSON following the StepContract schema
2. Every step must have pre/postconditions, fallbacks, timeouts, retries
3. Use specific, actionable selectors (role, name, css, xpath)
4. Include proper error handling and fallback strategies
5. Steps must be executable in parallel where possible

STEP CONTRACT SCHEMA:
{
  "id": "unique_uuid",
  "goal": "specific_action_description", 
  "pre": ["precondition1", "precondition2"],
  "action": {
    "type": "click|type|keypress|hover|scroll|navigate|screenshot|extract|verify",
    "target": {
      "role": "button|textbox|link|etc",
      "name": "accessible_name",
      "text": "text_content",
      "css": "css_selector", 
      "xpath": "xpath_selector"
    },
    "value": "input_value_if_typing",
    "keys": ["key1", "key2"] 
  },
  "post": ["postcondition1", "postcondition2"],
  "fallbacks": [{"type": "keypress", "keys": ["Escape"]}],
  "timeout_ms": 8000,
  "retries": 2,
  "evidence": ["screenshot", "dom_diff", "event_log"]
}

EXAMPLE STEPS:
- Navigate: {"action": {"type": "navigate", "target": {"url": "https://example.com"}}}
- Click button: {"action": {"type": "click", "target": {"role": "button", "name": "Submit"}}}
- Type text: {"action": {"type": "type", "target": {"role": "textbox", "name": "Email"}, "value": "user@example.com"}}
- Extract data: {"action": {"type": "extract", "target": {"css": ".result-text"}}}

DEPENDENCIES:
- Use "dependencies" field to specify step order
- Steps without dependencies can run in parallel
- Use meaningful preconditions to ensure proper sequencing

RESPOND WITH:
{"plan": [step1, step2, ...], "confidence": 0.95, "reasoning": "explanation"}"""
    
    async def plan(self, goal: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """
        Generate execution plan for goal using LLM.
        
        Args:
            goal: High-level goal description
            context: Additional context (current page, user data, etc.)
            
        Returns:
            ExecutionPlan with DAG structure
        """
        try:
            self.logger.info(f"Planning for goal: {goal}")
            
            # Build planning prompt
            prompt = self._build_planning_prompt(goal, context)
            
            # Get plan from LLM
            llm_response = await self._query_llm(prompt)
            
            # Parse and validate response
            plan_data = self._parse_llm_response(llm_response)
            
            # Build execution plan
            execution_plan = self._build_execution_plan(goal, plan_data)
            
            # Validate plan with simulator
            simulation_result = await self._validate_plan(execution_plan)
            
            # Apply confidence gating
            if (execution_plan.confidence < self.confidence_threshold or 
                simulation_result.confidence < self.simulation_threshold):
                
                self.logger.warning(f"Plan confidence too low: {execution_plan.confidence}, simulation: {simulation_result.confidence}")
                
                # Try to improve plan or ask for clarification
                execution_plan = await self._improve_plan(execution_plan, simulation_result, goal, context)
            
            self.current_plan = execution_plan
            self.execution_history.append(execution_plan)
            
            self.logger.info(f"Generated plan with {len(execution_plan.nodes)} steps, confidence: {execution_plan.confidence}")
            return execution_plan
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            raise
    
    def _build_planning_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """Build prompt for LLM planning."""
        prompt_parts = [
            f"GOAL: {goal}",
            "",
            "CONTEXT:"
        ]
        
        if context:
            if 'current_url' in context:
                prompt_parts.append(f"Current URL: {context['current_url']}")
            
            if 'page_elements' in context:
                prompt_parts.append("Available elements:")
                for element in context['page_elements'][:10]:  # Limit to avoid token overflow
                    prompt_parts.append(f"- {element}")
            
            if 'user_data' in context:
                prompt_parts.append(f"User data: {json.dumps(context['user_data'], indent=2)}")
        
        prompt_parts.extend([
            "",
            "Generate a detailed step-by-step plan to achieve this goal.",
            "Each step must be precise and executable.",
            "Include proper error handling and fallback strategies.",
            "Consider parallel execution where possible."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _query_llm(self, prompt: str) -> str:
        """Query LLM for plan generation."""
        if not self.openai_client and not self.anthropic_client:
            raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        
        try:
            if self.openai_client:
                response = await self._query_openai(prompt)
                if response:
                    return response
            
            if self.anthropic_client:
                response = await self._query_anthropic(prompt)
                if response:
                    return response
            
            raise RuntimeError("All LLM providers failed to respond")
            
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            raise
    
    async def _query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI API."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.warning(f"OpenAI query failed: {e}")
            return None
    
    async def _query_anthropic(self, prompt: str) -> Optional[str]:
        """Query Anthropic API."""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.warning(f"Anthropic query failed: {e}")
            return None
    
    def _generate_mock_plan_response(self) -> str:
        """Generate mock plan response for development."""
        return json.dumps({
            "plan": [
                {
                    "id": str(uuid.uuid4()),
                    "goal": "navigate_to_page",
                    "pre": [],
                    "action": {
                        "type": "navigate",
                        "target": {"url": "https://example.com"}
                    },
                    "post": ["url_contains('example.com')"],
                    "fallbacks": [],
                    "timeout_ms": 10000,
                    "retries": 2,
                    "evidence": ["screenshot"]
                },
                {
                    "id": str(uuid.uuid4()),
                    "goal": "click_submit_button",
                    "pre": ["exists(role=button,name='Submit')", "visible('Submit')"],
                    "action": {
                        "type": "click",
                        "target": {
                            "role": "button",
                            "name": "Submit"
                        }
                    },
                    "post": ["form_submitted()"],
                    "fallbacks": [{"type": "keypress", "keys": ["Enter"]}],
                    "timeout_ms": 8000,
                    "retries": 2,
                    "evidence": ["screenshot", "dom_diff"]
                }
            ],
            "confidence": 0.9,
            "reasoning": "Simple navigation and click plan with proper fallbacks"
        })
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            
            # Validate required fields
            if "plan" not in data:
                raise ValueError("Response missing 'plan' field")
            
            if "confidence" not in data:
                data["confidence"] = 0.8  # Default confidence
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    def _build_execution_plan(self, goal: str, plan_data: Dict[str, Any]) -> ExecutionPlan:
        """Build ExecutionPlan from LLM response."""
        plan_id = str(uuid.uuid4())
        nodes = {}
        
        # Create nodes from steps
        for i, step_data in enumerate(plan_data["plan"]):
            # Ensure step has required fields
            if "id" not in step_data:
                step_data["id"] = str(uuid.uuid4())
            
            # Create StepContract
            step = StepContract(**step_data)
            
            # Create PlanNode
            dependencies = set(step_data.get("dependencies", []))
            
            node = PlanNode(
                id=step.id,
                step=step,
                dependencies=dependencies,
                dependents=set()
            )
            
            nodes[step.id] = node
        
        # Build dependent relationships
        for node in nodes.values():
            for dep_id in node.dependencies:
                if dep_id in nodes:
                    nodes[dep_id].dependents.add(node.id)
        
        execution_plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            nodes=nodes,
            confidence=plan_data.get("confidence", 0.8)
        )
        
        return execution_plan
    
    async def _validate_plan(self, plan: ExecutionPlan) -> SimulationResult:
        """Validate plan using simulator."""
        try:
            # Convert plan to list of steps for simulation
            steps = [node.step for node in plan.nodes.values()]
            
            # Simulate the plan
            result = self.simulator.simulate(steps)
            
            self.logger.info(f"Plan simulation result: ok={result.ok}, confidence={result.confidence}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plan validation failed: {e}")
            return SimulationResult(
                ok=False,
                violations=[f"Validation error: {str(e)}"],
                expected_changes=[],
                confidence=0.0,
                simulation_time_ms=0,
                postcondition_results={},
                side_effects=[]
            )
    
    async def _improve_plan(self, plan: ExecutionPlan, simulation_result: SimulationResult, 
                          goal: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """Improve plan based on simulation feedback."""
        try:
            # Build improvement prompt
            improvement_prompt = f"""
ORIGINAL GOAL: {goal}

PLAN ISSUES:
- Plan confidence: {plan.confidence}
- Simulation confidence: {simulation_result.confidence}
- Violations: {simulation_result.violations}

FAILED POSTCONDITIONS:
{[k for k, v in simulation_result.postcondition_results.items() if not v]}

Please generate an improved plan that addresses these issues.
Focus on:
1. More specific selectors
2. Better preconditions
3. Additional fallback strategies
4. Proper error handling
"""
            
            # Query LLM for improvement
            improved_response = await self._query_llm(improvement_prompt)
            improved_data = self._parse_llm_response(improved_response)
            
            # Build improved plan
            improved_plan = self._build_execution_plan(goal, improved_data)
            
            # If still not good enough, return original plan
            if improved_plan.confidence <= plan.confidence:
                self.logger.warning("Plan improvement did not increase confidence, using original")
                return plan
            
            return improved_plan
            
        except Exception as e:
            self.logger.error(f"Plan improvement failed: {e}")
            return plan
    
    async def execute_plan(self, plan: ExecutionPlan = None) -> Dict[str, Any]:
        """
        Execute plan using DAG execution loop.
        
        DAG Execution Loop:
        while not plan.done:
          for node in plan.ready_parallel():
            spawn(executor(node))
          for result in gather():
            if result.drift: healer.patch(result)
            if result.needs_live: realtime.fetch(...)
            plan.update(result)
          if plan.conf < τ: micro_prompt_user()
        """
        if plan is None:
            plan = self.current_plan
        
        if plan is None:
            raise ValueError("No plan to execute")
        
        try:
            self.logger.info(f"Starting plan execution: {plan.id}")
            plan.status = PlanStatus.RUNNING
            plan.started_at = datetime.utcnow()
            
            # Main execution loop
            while not plan.done:
                # Get nodes ready for parallel execution
                ready_nodes = plan.get_ready_nodes()
                
                if not ready_nodes:
                    if plan._has_pending_nodes():
                        # Deadlock situation - some nodes are waiting but none are ready
                        self.logger.error("Plan execution deadlock detected")
                        plan.status = PlanStatus.FAILED
                        break
                    else:
                        # All nodes processed
                        break
                
                # Limit parallel execution
                ready_nodes = ready_nodes[:self.max_parallel_nodes]
                
                # Execute nodes in parallel
                tasks = []
                for node in ready_nodes:
                    node.status = NodeStatus.RUNNING
                    node.start_time = datetime.utcnow()
                    task = asyncio.create_task(self._execute_node(node))
                    tasks.append((node, task))
                
                # Wait for all tasks to complete
                for node, task in tasks:
                    try:
                        result = await task
                        success = result.get('success', False)
                        plan.update_node_result(node.id, result, success)
                        
                        # Handle drift detection
                        if result.get('drift_detected'):
                            await self._handle_drift(node, result)
                        
                        # Handle real-time data needs
                        if result.get('needs_live_data'):
                            await self._fetch_live_data(node, result)
                        
                    except Exception as e:
                        self.logger.error(f"Node execution failed: {e}")
                        plan.update_node_result(node.id, {'error': str(e), 'success': False}, False)
                
                # Check plan confidence
                if plan.confidence < self.confidence_threshold:
                    await self._micro_prompt_user(plan)
            
            # Finalize execution
            if plan.status == PlanStatus.RUNNING:
                plan.status = PlanStatus.COMPLETED
                plan.completed_at = datetime.utcnow()
            
            execution_result = {
                'plan_id': plan.id,
                'status': plan.status,
                'total_steps': plan.total_steps,
                'completed_steps': plan.completed_steps,
                'failed_steps': plan.failed_steps,
                'duration_ms': int((plan.completed_at - plan.started_at).total_seconds() * 1000) if plan.completed_at else 0
            }
            
            self.logger.info(f"Plan execution completed: {execution_result}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            plan.status = PlanStatus.FAILED
            plan.completed_at = datetime.utcnow()
            raise
    
    async def _execute_node(self, node: PlanNode) -> Dict[str, Any]:
        """Execute a single plan node."""
        # This method is now handled by the SuperOmegaOrchestrator
        # which provides real execution with self-healing
        raise NotImplementedError("Node execution is handled by SuperOmegaOrchestrator")
    
    async def _handle_drift(self, node: PlanNode, result: Dict[str, Any]):
        """Handle selector drift detection."""
        self.logger.info(f"Handling drift for node {node.id}")
        # This would integrate with the self-healing locator stack
        pass
    
    async def _fetch_live_data(self, node: PlanNode, result: Dict[str, Any]):
        """Fetch live data as needed."""
        self.logger.info(f"Fetching live data for node {node.id}")
        # This would integrate with the real-time data fabric
        pass
    
    async def _micro_prompt_user(self, plan: ExecutionPlan):
        """Micro-prompt user for clarification when confidence is low."""
        self.logger.info(f"Plan confidence low ({plan.confidence}), requesting user clarification")
        # This would integrate with the conversational interface
        pass
    
    def get_plan_status(self, plan_id: str = None) -> Optional[Dict[str, Any]]:
        """Get current plan status."""
        plan = self.current_plan if plan_id is None else self._find_plan_by_id(plan_id)
        
        if not plan:
            return None
        
        return {
            'id': plan.id,
            'goal': plan.goal,
            'status': plan.status,
            'confidence': plan.confidence,
            'total_steps': plan.total_steps,
            'completed_steps': plan.completed_steps,
            'failed_steps': plan.failed_steps,
            'progress': plan.completed_steps / plan.total_steps if plan.total_steps > 0 else 0,
            'created_at': plan.created_at.isoformat(),
            'started_at': plan.started_at.isoformat() if plan.started_at else None,
            'completed_at': plan.completed_at.isoformat() if plan.completed_at else None
        }
    
    def _find_plan_by_id(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Find plan by ID in execution history."""
        for plan in self.execution_history:
            if plan.id == plan_id:
                return plan
        return None