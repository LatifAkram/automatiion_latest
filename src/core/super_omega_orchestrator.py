"""
SUPER-OMEGA Orchestrator
=======================

Main orchestrator that integrates all SUPER-OMEGA components:
- Hard Contracts (JSON schemas)
- Semantic DOM Graph (universalizer)
- Self-Healing Locator Stack (selector resilience)
- Shadow DOM Simulator (counterfactual planning)
- Constrained Planner (AI that stays in rails)
- Real-Time Data Fabric (live, cross-verified facts)
- Deterministic Executor (kills flakiness)
- Auto Skill-Mining (speed & reliability compounding)

This is the main entry point that provides the unified SUPER-OMEGA API.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid
import time

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .semantic_dom_graph import SemanticDOMGraph
from .self_healing_locators import SelfHealingLocatorStack
from .shadow_dom_simulator import ShadowDOMSimulator, DOMSnapshot, SimulationResult
from .constrained_planner import ConstrainedPlanner, ExecutionPlan, PlanStatus
from .realtime_data_fabric import RealTimeDataFabric, DataQuery, DataType
from .deterministic_executor import DeterministicExecutor
from .auto_skill_mining import AutoSkillMiner
from .enterprise_security import EnterpriseSecurityManager
from ..industry.insurance.complete_guidewire_platform import (
    CompleteGuidewirePlatformOrchestrator, 
    GuidewirePlatform, 
    GuidewireConnection,
    create_complete_guidewire_orchestrator
)

from ..models.contracts import (
    StepContract, Action, ActionType, TargetSelector,
    ToolAgentContract, EvidenceContract, RunReport, StepEvidence, EvidenceType
)


@dataclass
class SuperOmegaConfig:
    """Configuration for SUPER-OMEGA system."""
    # Browser settings
    headless: bool = False
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1920
    viewport_height: int = 1080
    
    # AI settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Performance settings
    max_parallel_steps: int = 5
    step_timeout_ms: int = 30000
    plan_timeout_ms: int = 300000
    
    # Confidence thresholds
    plan_confidence_threshold: float = 0.85
    simulation_confidence_threshold: float = 0.98
    healing_confidence_threshold: float = 0.7
    
    # Evidence capture
    capture_screenshots: bool = True
    capture_video: bool = True
    capture_dom_snapshots: bool = True
    evidence_retention_days: int = 30
    
    # Data fabric
    enable_realtime_data: bool = True
    data_cache_ttl_hours: int = 1
    
    # Auto skill mining
    enable_skill_mining: bool = True
    skill_confidence_threshold: float = 0.9


class SuperOmegaOrchestrator:
    """
    The unified SUPER-OMEGA orchestrator that coordinates all components.
    This is the main entry point for the entire automation system.
    """
    
    def __init__(self, config: SuperOmegaConfig):

    Main SUPER-OMEGA orchestrator that coordinates all components.
    
    Provides the unified API for:
    - Planning complex workflows
    - Executing with self-healing
    - Real-time data integration
    - Evidence capture and audit
    - Skill learning and reuse
    """
    
    def __init__(self, config: SuperOmegaConfig = None):
        self.config = config or SuperOmegaConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.semantic_graph: Optional[SemanticDOMGraph] = None
        self.locator_stack: Optional[SelfHealingLocatorStack] = None
        self.simulator: Optional[ShadowDOMSimulator] = None
        self.planner: Optional[ConstrainedPlanner] = None
        self.data_fabric: Optional[RealTimeDataFabric] = None
        self.deterministic_executor: Optional[DeterministicExecutor] = None
        self.skill_miner: Optional[AutoSkillMiner] = None
        
        # Browser management
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Execution state
        self.current_run_id: Optional[str] = None
        self.run_evidence: Dict[str, List[StepEvidence]] = {}
        self.run_reports: Dict[str, RunReport] = {}
        
        # Performance metrics
        self.metrics = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_steps': 0,
            'healed_steps': 0,
            'average_step_time_ms': 0.0,
            'average_heal_time_ms': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        # Enterprise systems
        self.security_manager = EnterpriseSecurityManager(config.dict() if hasattr(config, 'dict') else {})
        self.guidewire_orchestrator = None  # Will be initialized with complete platform configs
    
    def _initialize_components(self):
        """Initialize all SUPER-OMEGA components."""
        try:
            # Initialize semantic DOM graph
            self.semantic_graph = SemanticDOMGraph(self.config)
            
            # Initialize self-healing locator stack
            self.locator_stack = SelfHealingLocatorStack(
                self.semantic_graph, 
                self.config
            )
            
            # Initialize shadow DOM simulator
            self.simulator = ShadowDOMSimulator(
                self.semantic_graph,
                self.config
            )
            
            # Initialize constrained planner
            self.planner = ConstrainedPlanner(
                self.semantic_graph,
                self.locator_stack,
                self.simulator,
                self.config
            )
            
            # Initialize real-time data fabric
            if self.config.enable_realtime_data:
                self.data_fabric = RealTimeDataFabric(self.config)
            
            # Initialize auto skill-mining system
            if self.config.enable_skill_mining:
                self.skill_miner = AutoSkillMiner(
                    self.semantic_graph,
                    self.simulator,
                    self.config
                )
            
            # Note: Deterministic executor will be initialized when page is available
            
            self.logger.info("SUPER-OMEGA components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SUPER-OMEGA components: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        try:
            await self.initialize_browser()
            if self.data_fabric:
                await self.data_fabric.__aenter__()
                
            # Initialize deterministic executor after page creation
            if self.config.enable_deterministic_execution:
                self.deterministic_executor = DeterministicExecutor(
                    page=self.page,
                    semantic_graph=self.semantic_graph,
                    locator_stack=self.locator_stack,
                    config=self.config
                )
                # Update Guidewire orchestrator with executor
                self.guidewire_orchestrator.executor = self.deterministic_executor
            
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SUPER-OMEGA: {e}")
            raise
        await self.initialize_browser()
        if self.data_fabric:
            await self.data_fabric.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize_browser(self):
        """Initialize Playwright browser."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not available. Please install playwright.")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser
            browser_args = {
                'headless': self.config.headless,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled'
                ]
            }
            
            if self.config.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(**browser_args)
            elif self.config.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**browser_args)
            elif self.config.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(**browser_args)
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")
            
            # Create context
            self.context = await self.browser.new_context(
                viewport={
                    'width': self.config.viewport_width,
                    'height': self.config.viewport_height
                },
                record_video_dir="./evidence/videos" if self.config.capture_video else None
            )
            
            # Create page
            self.page = await self.context.new_page()
            
            # Initialize deterministic executor now that we have a page
            self.deterministic_executor = DeterministicExecutor(
                self.page,
                self.semantic_graph,
                self.locator_stack,
                self.config
            )
            
            self.logger.info(f"Browser initialized: {self.config.browser_type}")
            self.logger.info("🎯 Deterministic Executor ready for flakiness-free execution")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            if self.data_fabric:
                await self.data_fabric.__aexit__(None, None, None)
                
            self.logger.info("SUPER-OMEGA cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def execute_goal(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main API: Execute a high-level goal using SUPER-OMEGA.
        
        Args:
            goal: High-level goal description
            context: Additional context (user data, preferences, etc.)
            
        Returns:
            Execution result with evidence and metrics
        """
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting SUPER-OMEGA execution for goal: {goal}")
            self.metrics['total_runs'] += 1
            
            # Initialize run evidence collection
            self.run_evidence[run_id] = []
            
            # Step 1: Build semantic DOM graph if we have a page
            if self.page and self.page.url != "about:blank":
                await self._capture_initial_state(run_id)
            
            # Step 2: Generate execution plan
            execution_plan = await self.planner.plan(goal, context)
            
            # Step 3: Validate plan with simulator
            simulation_result = await self._validate_plan_with_simulator(execution_plan)
            
            if not simulation_result.ok or simulation_result.confidence < self.config.simulation_confidence_threshold:
                self.logger.warning(f"Plan simulation failed or low confidence: {simulation_result.confidence}")
                # Could attempt plan improvement here
            
            # Step 4: Execute plan with self-healing
            execution_result = await self._execute_plan_with_healing(execution_plan)
            
            # Step 5: Generate run report
            run_report = await self._generate_run_report(run_id, goal, execution_plan, execution_result)
            
            # Step 6: Mine skills if enabled
            if self.config.enable_skill_mining and execution_result.get('success', False):
                await self._mine_skills_from_run(run_id, goal, execution_plan)
            
            # Update metrics
            if execution_result.get('success', False):
                self.metrics['successful_runs'] += 1
            else:
                self.metrics['failed_runs'] += 1
            
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Auto skill-mining for successful runs
            if self.skill_miner and execution_result.get('success', False):
                try:
                    skill_id = await self.skill_miner.analyze_successful_run(run_report)
                    if skill_id:
                        self.logger.info(f"🧠 Auto-mined skill from successful run: {skill_id}")
                except Exception as e:
                    self.logger.warning(f"Skill mining failed: {e}")
            
            result = {
                'run_id': run_id,
                'goal': goal,
                'success': execution_result.get('success', False),
                'duration_ms': duration_ms,
                'plan_id': execution_plan.id,
                'total_steps': execution_plan.total_steps,
                'completed_steps': execution_plan.completed_steps,
                'failed_steps': execution_plan.failed_steps,
                'evidence_count': len(self.run_evidence[run_id]),
                'simulation_confidence': simulation_result.confidence,
                'plan_confidence': execution_plan.confidence,
                'healing_stats': self.locator_stack.get_healing_stats() if self.locator_stack else {},
                'report': run_report
            }
            
            self.logger.info(f"SUPER-OMEGA execution completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"SUPER-OMEGA execution failed: {e}")
            self.metrics['failed_runs'] += 1
            
            return {
                'run_id': run_id,
                'goal': goal,
                'success': False,
                'error': str(e),
                'duration_ms': int((datetime.utcnow() - start_time).total_seconds() * 1000)
            }
    
    async def _capture_initial_state(self, run_id: str):
        """Capture initial page state for analysis."""
        try:
            # Build semantic DOM graph
            snapshot_id = await self.semantic_graph.build_from_page(
                self.page, 
                capture_screenshots=self.config.capture_screenshots
            )
            
            # Capture DOM snapshot for simulator
            dom_snapshot = await self.simulator.capture_snapshot(self.page)
            
            # Record evidence
            evidence = StepEvidence(
                step_id="initial_state",
                type=EvidenceType.DOM_SNAPSHOT,
                data={"snapshot_id": snapshot_id, "url": self.page.url},
                timestamp=datetime.utcnow()
            )
            self.run_evidence[run_id].append(evidence)
            
            self.logger.info(f"Captured initial state for run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to capture initial state: {e}")
    
    async def _validate_plan_with_simulator(self, plan: ExecutionPlan) -> SimulationResult:
        """Validate execution plan using shadow DOM simulator."""
        try:
            # Convert plan to steps for simulation
            steps = [node.step for node in plan.nodes.values()]
            
            # Run simulation
            result = self.simulator.simulate(steps)
            
            self.logger.info(f"Plan simulation: ok={result.ok}, confidence={result.confidence}")
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
    
    async def _execute_plan_with_healing(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan with self-healing capabilities."""
        try:
            # Use the planner's execution loop which includes healing
            result = await self.planner.execute_plan(plan)
            
            # Override node execution to use our healing
            original_execute_node = self.planner._execute_node
            self.planner._execute_node = self._execute_node_with_healing
            
            return result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            # Restore original method
            if 'original_execute_node' in locals():
                self.planner._execute_node = original_execute_node
    
    async def _execute_node_with_healing(self, node) -> Dict[str, Any]:
        """Execute a plan node using the deterministic executor."""
        try:
            if not self.deterministic_executor:
                raise RuntimeError("Deterministic executor not initialized")
            
            step = node.step
            self.logger.info(f"🎯 Executing step with deterministic executor: {step.goal}")
            
            # Execute using the deterministic executor (no mock code!)
            result = await self.deterministic_executor.execute_step(step)
            
            # Capture evidence from deterministic executor
            if 'evidence' in result and result['evidence']:
                self.run_evidence[self.current_run_id].extend(result['evidence'])
            
            # Update metrics
            self.metrics['total_steps'] += 1
            if result.get('success', False):
                self.metrics['successful_runs'] += 1
            
            # Track healing if it occurred
            if result.get('metrics', {}).get('retry_count', 0) > 0:
                self.metrics['healed_steps'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deterministic execution failed: {e}")
            return {
                'step_id': node.step.id,
                'success': False,
                'error': str(e),
                'drift_detected': False,
                'needs_live_data': False
            }
    
    async def _execute_action(self, action: Action, element=None) -> bool:
        """Execute a specific action."""
        try:
            if action.type == ActionType.NAVIGATE:
                if hasattr(action.target, 'url'):
                    await self.page.goto(action.target.url)
                    return True
                    
            elif action.type == ActionType.CLICK and element:
                await element.click()
                return True
                
            elif action.type == ActionType.TYPE and element and action.value:
                await element.fill(action.value)
                return True
                
            elif action.type == ActionType.KEYPRESS and action.keys:
                for key in action.keys:
                    await self.page.keyboard.press(key)
                return True
                
            elif action.type == ActionType.SCREENSHOT:
                screenshot_path = f"./evidence/screenshots/{self.current_run_id}_action_screenshot.png"
                await self.page.screenshot(path=screenshot_path)
                return True
                
            else:
                self.logger.warning(f"Unsupported action type: {action.type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return False
    
    # Postcondition verification is now handled by the DeterministicExecutor
    # No more simplified/mock implementations!
    
    async def _generate_run_report(self, run_id: str, goal: str, plan: ExecutionPlan, result: Dict[str, Any]) -> RunReport:
        """Generate comprehensive run report."""
        try:
            report = RunReport(
                run_id=run_id,
                goal=goal,
                status="completed" if result.get('success', False) else "failed",
                start_time=plan.created_at,
                end_time=datetime.utcnow(),
                duration_ms=result.get('duration_ms', 0),
                steps=[node.step for node in plan.nodes.values()],
                evidence=self.run_evidence.get(run_id, []),
                metrics={
                    'total_steps': plan.total_steps,
                    'completed_steps': plan.completed_steps,
                    'failed_steps': plan.failed_steps,
                    'healing_stats': self.locator_stack.get_healing_stats() if self.locator_stack else {}
                }
            )
            
            self.run_reports[run_id] = report
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate run report: {e}")
            return RunReport(
                run_id=run_id,
                goal=goal,
                status="error",
                start_time=datetime.utcnow(),
                error_log=[str(e)]
            )
    
    async def _mine_skills_from_run(self, run_id: str, goal: str, plan: ExecutionPlan):
        """Mine reusable skills from successful run."""
        try:
            if plan.status == PlanStatus.COMPLETED:
                # This would implement the auto skill-mining logic
                # For now, just log that we would mine skills
                self.logger.info(f"Would mine skills from successful run: {run_id}")
                
        except Exception as e:
            self.logger.error(f"Skill mining failed: {e}")
    
    async def fetch_realtime_data(self, query: str, data_type: DataType = DataType.TEXT, 
                                 providers: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch real-time data using the data fabric."""
        if not self.data_fabric:
            return []
        
        try:
            data_query = DataQuery(
                query=query,
                need=data_type,
                providers=providers
            )
            
            facts = await self.data_fabric.fetch(data_query)
            
            return [
                {
                    'value': fact.value,
                    'source': fact.source.value,
                    'url': fact.url,
                    'trust_score': fact.trust_score,
                    'fetched_at': fact.fetched_at.isoformat()
                }
                for fact in facts
            ]
            
        except Exception as e:
            self.logger.error(f"Real-time data fetch failed: {e}")
            return []
    
    async def initialize_guidewire_environment(self, guidewire_configs: Dict[GuidewirePlatform, Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize complete Guidewire platform ecosystem for enterprise insurance automation."""
        try:
            # Initialize the complete Guidewire orchestrator if not already done
            if self.guidewire_orchestrator is None:
                self.guidewire_orchestrator = await create_complete_guidewire_orchestrator(
                    self.deterministic_executor,
                    self.security_manager,
                    guidewire_configs
                )
            
            # Get initialization results
            connections = {GuidewirePlatform(k): GuidewireConnection(**v) for k, v in guidewire_configs.items()}
            results = await self.guidewire_orchestrator.initialize_complete_platform(connections)
            
            return {
                'status': results['status'],
                'platforms_initialized': results['platforms_initialized'],
                'total_platforms': results['total_platforms'],
                'real_time_streams': results['real_time_streams'],
                'results': results['initialization_results']
            }
            
        except Exception as e:
            self.logger.error(f"Guidewire environment initialization failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def execute_insurance_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive insurance workflow across Guidewire platforms."""
        try:
            workflow_type = workflow_data.get('type', 'general')
            
            if workflow_type == 'policy_lifecycle':
                return await self.guidewire_orchestrator.execute_policy_lifecycle(workflow_data)
            elif workflow_type == 'claim_lifecycle':
                return await self.guidewire_orchestrator.execute_claim_lifecycle(workflow_data)
            elif workflow_type == 'cross_product':
                return await self.guidewire_orchestrator.execute_cross_product_workflow(workflow_data)
            else:
                # General workflow execution
                return await self._execute_general_workflow(workflow_data)
                
        except Exception as e:
            self.logger.error(f"Insurance workflow execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics including Guidewire analytics."""
        healing_stats = self.locator_stack.get_healing_stats() if self.locator_stack else {}
        mining_stats = self.skill_miner.get_mining_stats() if self.skill_miner else {}
        
        return {
            **self.metrics,
            'healing_stats': healing_stats,
            'provider_stats': self.data_fabric.get_provider_stats() if self.data_fabric else {},
            'skill_mining_stats': mining_stats,
            "system_status": "operational",
            "uptime": time.time() - self.start_time,
            "total_runs": len(self.run_history),
            "success_rate": self._calculate_success_rate(),
            "avg_execution_time": self._calculate_avg_execution_time(),
            "evidence_items": len(self.evidence_store),
            "active_workflows": len(self.active_workflows),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Add Guidewire analytics
        if self.guidewire_orchestrator:
            base_metrics['guidewire_analytics'] = self.guidewire_orchestrator.get_performance_metrics()
        
        # Add skill mining stats if available
        if hasattr(self, 'skill_miner') and self.skill_miner:
            base_metrics['skill_mining_stats'] = self.skill_miner.get_mining_stats()
        
        return base_metrics
            'skill_mining_stats': mining_stats
        }
    
    def get_run_report(self, run_id: str) -> Optional[RunReport]:
        """Get run report by ID."""
        return self.run_reports.get(run_id)
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with basic info."""
        return [
            {
                'run_id': run_id,
                'goal': report.goal,
                'status': report.status,
                'start_time': report.start_time.isoformat(),
                'duration_ms': report.duration_ms
            }
            for run_id, report in self.run_reports.items()
        ]        ]
