"""
MULTI-TAB ORCHESTRATION SYSTEM
==============================

Enterprise-grade multi-tab automation for complex workflows that require
coordination across multiple browser tabs and windows.

✅ FEATURES:
- Multi-tab workflow coordination
- Tab-specific session management
- Cross-tab data sharing
- Parallel execution with synchronization
- Tab lifecycle management
- Error handling across tabs
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from playwright.async_api import Page, BrowserContext, Browser

logger = logging.getLogger(__name__)

class TabState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    CLOSED = "closed"

class TabPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TabTask:
    """Individual task to be executed in a specific tab"""
    task_id: str
    tab_id: str
    action: str
    target: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: TabPriority = TabPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    
@dataclass
class TabInfo:
    """Information about a managed tab"""
    tab_id: str
    page: Page
    url: str
    title: str
    state: TabState
    created_at: float
    last_activity: float
    session_data: Dict[str, Any] = field(default_factory=dict)
    task_queue: List[TabTask] = field(default_factory=list)
    
@dataclass
class WorkflowStep:
    """Step in a multi-tab workflow"""
    step_id: str
    step_type: str  # 'sequential', 'parallel', 'conditional'
    tabs_required: List[str]
    tasks: List[TabTask]
    synchronization_points: List[str] = field(default_factory=list)
    timeout: float = 60.0

class MultiTabOrchestrator:
    """Enterprise multi-tab orchestration system"""
    
    def __init__(self, browser_context: BrowserContext):
        self.context = browser_context
        self.tabs: Dict[str, TabInfo] = {}
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.shared_data: Dict[str, Any] = {}
        self.synchronization_events: Dict[str, asyncio.Event] = {}
        self.active_workflows: Dict[str, bool] = {}
        
    async def create_tab(self, tab_id: str, url: str = "about:blank") -> TabInfo:
        """Create a new managed tab"""
        try:
            page = await self.context.new_page()
            await page.goto(url)
            
            tab_info = TabInfo(
                tab_id=tab_id,
                page=page,
                url=url,
                title=await page.title(),
                state=TabState.READY,
                created_at=time.time(),
                last_activity=time.time()
            )
            
            self.tabs[tab_id] = tab_info
            logger.info(f"Created tab {tab_id} with URL: {url}")
            return tab_info
            
        except Exception as e:
            logger.error(f"Failed to create tab {tab_id}: {e}")
            raise
    
    async def close_tab(self, tab_id: str):
        """Close a managed tab"""
        if tab_id in self.tabs:
            try:
                await self.tabs[tab_id].page.close()
                self.tabs[tab_id].state = TabState.CLOSED
                logger.info(f"Closed tab {tab_id}")
            except Exception as e:
                logger.error(f"Failed to close tab {tab_id}: {e}")
    
    async def execute_workflow(self, workflow_id: str, workflow_steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Execute a complex multi-tab workflow"""
        self.workflows[workflow_id] = workflow_steps
        self.active_workflows[workflow_id] = True
        
        results = {
            'workflow_id': workflow_id,
            'steps_completed': 0,
            'total_steps': len(workflow_steps),
            'step_results': [],
            'success': False,
            'error': None,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            for step_idx, step in enumerate(workflow_steps):
                if not self.active_workflows.get(workflow_id, False):
                    break
                
                logger.info(f"Executing workflow step {step_idx + 1}/{len(workflow_steps)}: {step.step_id}")
                
                step_result = await self._execute_workflow_step(step)
                results['step_results'].append(step_result)
                
                if step_result['success']:
                    results['steps_completed'] += 1
                else:
                    results['error'] = step_result.get('error', 'Step failed')
                    break
            
            results['success'] = results['steps_completed'] == results['total_steps']
            results['execution_time'] = time.time() - start_time
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Workflow {workflow_id} failed: {e}")
        
        finally:
            self.active_workflows[workflow_id] = False
        
        return results
    
    async def _execute_workflow_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_result = {
            'step_id': step.step_id,
            'step_type': step.step_type,
            'success': False,
            'tasks_completed': 0,
            'total_tasks': len(step.tasks),
            'task_results': [],
            'error': None
        }
        
        try:
            # Ensure required tabs exist
            await self._ensure_tabs_exist(step.tabs_required)
            
            if step.step_type == 'sequential':
                # Execute tasks sequentially
                for task in step.tasks:
                    task_result = await self._execute_tab_task(task)
                    step_result['task_results'].append(task_result)
                    
                    if task_result['success']:
                        step_result['tasks_completed'] += 1
                    else:
                        step_result['error'] = task_result.get('error', 'Task failed')
                        break
            
            elif step.step_type == 'parallel':
                # Execute tasks in parallel
                tasks_coroutines = [self._execute_tab_task(task) for task in step.tasks]
                task_results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
                
                for i, result in enumerate(task_results):
                    if isinstance(result, Exception):
                        task_result = {'success': False, 'error': str(result), 'task_id': step.tasks[i].task_id}
                    else:
                        task_result = result
                        if task_result['success']:
                            step_result['tasks_completed'] += 1
                    
                    step_result['task_results'].append(task_result)
            
            elif step.step_type == 'conditional':
                # Execute tasks based on conditions
                for task in step.tasks:
                    if await self._evaluate_task_condition(task):
                        task_result = await self._execute_tab_task(task)
                        step_result['task_results'].append(task_result)
                        
                        if task_result['success']:
                            step_result['tasks_completed'] += 1
            
            # Handle synchronization points
            for sync_point in step.synchronization_points:
                await self._wait_for_synchronization(sync_point)
            
            step_result['success'] = step_result['tasks_completed'] > 0
            
        except Exception as e:
            step_result['error'] = str(e)
            logger.error(f"Step {step.step_id} failed: {e}")
        
        return step_result
    
    async def _execute_tab_task(self, task: TabTask) -> Dict[str, Any]:
        """Execute a task in a specific tab"""
        task_result = {
            'task_id': task.task_id,
            'tab_id': task.tab_id,
            'action': task.action,
            'success': False,
            'error': None,
            'result_data': {},
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            if task.tab_id not in self.tabs:
                raise Exception(f"Tab {task.tab_id} not found")
            
            tab_info = self.tabs[task.tab_id]
            tab_info.state = TabState.EXECUTING
            tab_info.last_activity = time.time()
            
            # Execute the task based on action type
            if task.action == 'navigate':
                await tab_info.page.goto(task.data.get('url', ''))
                task_result['result_data'] = {'url': tab_info.page.url}
                
            elif task.action == 'click':
                selector = task.data.get('selector', task.target)
                element = tab_info.page.locator(selector)
                await element.click(timeout=task.timeout * 1000)
                task_result['result_data'] = {'clicked': selector}
                
            elif task.action == 'input':
                selector = task.data.get('selector', task.target)
                text = task.data.get('text', '')
                element = tab_info.page.locator(selector)
                await element.fill(text)
                task_result['result_data'] = {'input': text, 'selector': selector}
                
            elif task.action == 'extract':
                selector = task.data.get('selector', task.target)
                attribute = task.data.get('attribute', 'textContent')
                element = tab_info.page.locator(selector)
                value = await element.get_attribute(attribute) if attribute != 'textContent' else await element.text_content()
                task_result['result_data'] = {'extracted': value, 'selector': selector}
                
                # Store in shared data if specified
                if 'store_as' in task.data:
                    self.shared_data[task.data['store_as']] = value
                
            elif task.action == 'wait':
                wait_type = task.data.get('wait_type', 'time')
                if wait_type == 'time':
                    await asyncio.sleep(task.data.get('duration', 1))
                elif wait_type == 'element':
                    selector = task.data.get('selector', task.target)
                    await tab_info.page.wait_for_selector(selector, timeout=task.timeout * 1000)
                elif wait_type == 'url':
                    url_pattern = task.data.get('url_pattern', '')
                    await tab_info.page.wait_for_url(url_pattern, timeout=task.timeout * 1000)
                
                task_result['result_data'] = {'wait_completed': True}
            
            elif task.action == 'screenshot':
                screenshot_path = task.data.get('path', f'screenshots/tab_{task.tab_id}_{int(time.time())}.png')
                await tab_info.page.screenshot(path=screenshot_path)
                task_result['result_data'] = {'screenshot': screenshot_path}
            
            elif task.action == 'execute_script':
                script = task.data.get('script', '')
                result = await tab_info.page.evaluate(script)
                task_result['result_data'] = {'script_result': result}
            
            elif task.action == 'share_data':
                # Share data between tabs
                key = task.data.get('key', '')
                value = task.data.get('value', '')
                self.shared_data[key] = value
                task_result['result_data'] = {'shared': {key: value}}
            
            task_result['success'] = True
            tab_info.state = TabState.READY
            
        except Exception as e:
            task_result['error'] = str(e)
            logger.error(f"Task {task.task_id} failed in tab {task.tab_id}: {e}")
            if task.tab_id in self.tabs:
                self.tabs[task.tab_id].state = TabState.ERROR
        
        finally:
            task_result['execution_time'] = time.time() - start_time
        
        return task_result
    
    async def _ensure_tabs_exist(self, tab_ids: List[str]):
        """Ensure all required tabs exist"""
        for tab_id in tab_ids:
            if tab_id not in self.tabs:
                await self.create_tab(tab_id)
    
    async def _evaluate_task_condition(self, task: TabTask) -> bool:
        """Evaluate if a conditional task should be executed"""
        condition = task.data.get('condition', {})
        
        if 'shared_data_key' in condition:
            key = condition['shared_data_key']
            expected_value = condition.get('expected_value')
            actual_value = self.shared_data.get(key)
            return actual_value == expected_value
        
        if 'tab_state' in condition:
            tab_id = condition['tab_id']
            expected_state = TabState(condition['tab_state'])
            actual_state = self.tabs.get(tab_id, {}).state if tab_id in self.tabs else None
            return actual_state == expected_state
        
        return True  # Default to execute if no condition specified
    
    async def _wait_for_synchronization(self, sync_point: str):
        """Wait for a synchronization point"""
        if sync_point not in self.synchronization_events:
            self.synchronization_events[sync_point] = asyncio.Event()
        
        # Set the event to allow other waiting coroutines to proceed
        self.synchronization_events[sync_point].set()
        
        # Wait briefly to allow synchronization
        await asyncio.sleep(0.1)
    
    def get_tab_info(self, tab_id: str) -> Optional[TabInfo]:
        """Get information about a specific tab"""
        return self.tabs.get(tab_id)
    
    def get_all_tabs_info(self) -> Dict[str, TabInfo]:
        """Get information about all managed tabs"""
        return self.tabs.copy()
    
    def get_shared_data(self, key: str = None) -> Any:
        """Get shared data between tabs"""
        if key:
            return self.shared_data.get(key)
        return self.shared_data.copy()
    
    def set_shared_data(self, key: str, value: Any):
        """Set shared data between tabs"""
        self.shared_data[key] = value
    
    async def cleanup(self):
        """Clean up all managed tabs"""
        for tab_id in list(self.tabs.keys()):
            await self.close_tab(tab_id)
        
        self.tabs.clear()
        self.workflows.clear()
        self.shared_data.clear()
        self.synchronization_events.clear()

# Workflow builder utilities
class WorkflowBuilder:
    """Helper class to build complex multi-tab workflows"""
    
    def __init__(self):
        self.steps: List[WorkflowStep] = []
    
    def add_sequential_step(self, step_id: str, tasks: List[TabTask], tabs_required: List[str] = None) -> 'WorkflowBuilder':
        """Add a sequential execution step"""
        if tabs_required is None:
            tabs_required = list(set(task.tab_id for task in tasks))
        
        step = WorkflowStep(
            step_id=step_id,
            step_type='sequential',
            tabs_required=tabs_required,
            tasks=tasks
        )
        self.steps.append(step)
        return self
    
    def add_parallel_step(self, step_id: str, tasks: List[TabTask], tabs_required: List[str] = None) -> 'WorkflowBuilder':
        """Add a parallel execution step"""
        if tabs_required is None:
            tabs_required = list(set(task.tab_id for task in tasks))
        
        step = WorkflowStep(
            step_id=step_id,
            step_type='parallel',
            tabs_required=tabs_required,
            tasks=tasks
        )
        self.steps.append(step)
        return self
    
    def add_conditional_step(self, step_id: str, tasks: List[TabTask], tabs_required: List[str] = None) -> 'WorkflowBuilder':
        """Add a conditional execution step"""
        if tabs_required is None:
            tabs_required = list(set(task.tab_id for task in tasks))
        
        step = WorkflowStep(
            step_id=step_id,
            step_type='conditional',
            tabs_required=tabs_required,
            tasks=tasks
        )
        self.steps.append(step)
        return self
    
    def add_synchronization_point(self, sync_point: str) -> 'WorkflowBuilder':
        """Add a synchronization point to the last step"""
        if self.steps:
            self.steps[-1].synchronization_points.append(sync_point)
        return self
    
    def build(self) -> List[WorkflowStep]:
        """Build the workflow steps"""
        return self.steps.copy()

# Global orchestrator instance
_global_orchestrator: Optional[MultiTabOrchestrator] = None

async def get_multi_tab_orchestrator(browser_context: BrowserContext = None) -> MultiTabOrchestrator:
    """Get or create the global multi-tab orchestrator"""
    global _global_orchestrator
    
    if _global_orchestrator is None and browser_context is not None:
        _global_orchestrator = MultiTabOrchestrator(browser_context)
    
    return _global_orchestrator

async def cleanup_multi_tab_orchestrator():
    """Clean up the global multi-tab orchestrator"""
    global _global_orchestrator
    
    if _global_orchestrator is not None:
        await _global_orchestrator.cleanup()
        _global_orchestrator = None