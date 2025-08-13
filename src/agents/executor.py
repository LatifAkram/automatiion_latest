"""
AI-2: Execution Agent (Automation)
=================================

The execution agent that handles web automation, API calls, and task execution.
Uses Playwright/Selenium for browser automation and can run multiple workflows in parallel.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import aiohttp
import requests

from ..core.audit import AuditLogger
from ..models.execution import ExecutionStep, ExecutionResult
from ..utils.media_capture import MediaCapture
from ..utils.selector_drift import SelectorDriftDetector


class ExecutionAgent:
    """
    AI-2: Execution Agent - Handles automation and task execution.
    """
    
    def __init__(self, agent_id: str, config: Any, audit_logger: AuditLogger):
        self.agent_id = agent_id
        self.config = config
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Browser automation
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Media capture
        self.media_capture = MediaCapture(config.media_path)
        
        # Selector drift detection
        self.drift_detector = SelectorDriftDetector()
        
        # Agent state
        self.is_busy = False
        self.current_task: Optional[Dict[str, Any]] = None
        self.session_start_time: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize the execution agent."""
        self.logger.info(f"Initializing Execution Agent: {self.agent_id}")
        
        # Initialize browser
        await self._initialize_browser()
        
        # Initialize media capture
        await self.media_capture.initialize()
        
        # Initialize drift detector
        await self.drift_detector.initialize()
        
        self.session_start_time = datetime.utcnow()
        self.logger.info(f"Execution Agent {self.agent_id} initialized successfully")
        
    async def _initialize_browser(self):
        """Initialize browser for automation."""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser based on configuration
            if self.config.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(
                    headless=self.config.headless,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-accelerated-2d-canvas",
                        "--no-first-run",
                        "--no-zygote",
                        "--disable-gpu"
                    ]
                )
            elif self.config.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(
                    headless=self.config.headless
                )
            elif self.config.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(
                    headless=self.config.headless
                )
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")
                
            # Create browser context
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            # Create new page
            self.page = await self.context.new_page()
            
            # Set default timeout
            self.page.set_default_timeout(self.config.browser_timeout)
            
            self.logger.info(f"Browser initialized: {self.config.browser_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}", exc_info=True)
            raise
            
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single task.
        
        Args:
            task: Task definition with type, parameters, and configuration
            
        Returns:
            Execution result with success status, data, and artifacts
        """
        self.is_busy = True
        self.current_task = task
        
        try:
            self.logger.info(f"Executing task: {task.get('id', 'unknown')} - {task.get('name', 'unnamed')}")
            
            # Start execution step
            step = ExecutionStep(
                task_id=task.get("id"),
                task_name=task.get("name"),
                task_type=task.get("type"),
                start_time=datetime.utcnow()
            )
            
            # Execute based on task type
            task_type = task.get("type", "general")
            
            if task_type == "web_automation":
                result = await self._execute_web_automation(task)
            elif task_type == "api_call":
                result = await self._execute_api_call(task)
            elif task_type == "dom_extraction":
                result = await self._execute_dom_extraction(task)
            elif task_type == "data_processing":
                result = await self._execute_data_processing(task)
            elif task_type == "file_operation":
                result = await self._execute_file_operation(task)
            else:
                result = await self._execute_general_task(task)
                
            # Complete execution step
            step.end_time = datetime.utcnow()
            step.success = result.get("success", False)
            step.data = result.get("data", {})
            step.error = result.get("error")
            
            # Capture media if enabled
            if self.config.capture_screenshots and task_type == "web_automation":
                screenshot_path = await self.media_capture.capture_screenshot(
                    self.page, task.get("id"), "task_completion"
                )
                step.artifacts["screenshot"] = screenshot_path
                
            # Log execution
            await self.audit_logger.log_task_execution(
                agent_id=self.agent_id,
                task=task,
                step=step
            )
            
            return {
                "success": step.success,
                "data": step.data,
                "error": step.error,
                "artifacts": step.artifacts,
                "duration": (step.end_time - step.start_time).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {},
                "artifacts": {},
                "duration": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        finally:
            self.is_busy = False
            self.current_task = None
            
    async def _execute_web_automation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web automation tasks using Playwright."""
        try:
            actions = task.get("actions", [])
            url = task.get("url")
            
            if url:
                await self.page.goto(url, wait_until="networkidle")
                
            results = []
            
            for action in actions:
                action_type = action.get("type")
                
                if action_type == "click":
                    selector = action.get("selector")
                    await self._click_element(selector, action)
                    
                elif action_type == "type":
                    selector = action.get("selector")
                    text = action.get("text", "")
                    await self._type_text(selector, text, action)
                    
                elif action_type == "select":
                    selector = action.get("selector")
                    value = action.get("value")
                    await self._select_option(selector, value, action)
                    
                elif action_type == "wait":
                    duration = action.get("duration", 1000)
                    await self.page.wait_for_timeout(duration)
                    
                elif action_type == "wait_for_element":
                    selector = action.get("selector")
                    await self.page.wait_for_selector(selector)
                    
                elif action_type == "extract_text":
                    selector = action.get("selector")
                    text = await self.page.text_content(selector)
                    results.append({"type": "text", "selector": selector, "value": text})
                    
                elif action_type == "extract_attribute":
                    selector = action.get("selector")
                    attribute = action.get("attribute")
                    value = await self.page.get_attribute(selector, attribute)
                    results.append({"type": "attribute", "selector": selector, "attribute": attribute, "value": value})
                    
                elif action_type == "screenshot":
                    path = await self.media_capture.capture_screenshot(
                        self.page, task.get("id"), action.get("name", "screenshot")
                    )
                    results.append({"type": "screenshot", "path": path})
                    
            return {
                "success": True,
                "data": {
                    "url": url,
                    "actions_performed": len(actions),
                    "extracted_data": results
                }
            }
            
        except Exception as e:
            self.logger.error(f"Web automation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    async def _click_element(self, selector: str, action: Dict[str, Any]):
        """Click an element with drift detection and retry logic."""
        max_retries = action.get("max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                # Check for selector drift
                if await self.drift_detector.detect_drift(selector, self.page):
                    new_selector = await self.drift_detector.suggest_alternative(selector, self.page)
                    if new_selector:
                        selector = new_selector
                        
                await self.page.click(selector)
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
                
    async def _type_text(self, selector: str, text: str, action: Dict[str, Any]):
        """Type text into an element."""
        await self.page.fill(selector, text)
        
    async def _select_option(self, selector: str, value: str, action: Dict[str, Any]):
        """Select an option from a dropdown."""
        await self.page.select_option(selector, value)
        
    async def _execute_api_call(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API calls."""
        try:
            method = task.get("method", "GET")
            url = task.get("url")
            headers = task.get("headers", {})
            data = task.get("data", {})
            params = task.get("params", {})
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=params) as response:
                        response_data = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        response_data = await response.json()
                elif method.upper() == "PUT":
                    async with session.put(url, headers=headers, json=data) as response:
                        response_data = await response.json()
                elif method.upper() == "DELETE":
                    async with session.delete(url, headers=headers) as response:
                        response_data = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
            return {
                "success": True,
                "data": {
                    "status_code": response.status,
                    "response": response_data,
                    "headers": dict(response.headers)
                }
            }
            
        except Exception as e:
            self.logger.error(f"API call failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    async def _execute_dom_extraction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from DOM elements."""
        try:
            url = task.get("url")
            selectors = task.get("selectors", {})
            
            await self.page.goto(url, wait_until="networkidle")
            
            extracted_data = {}
            
            for key, selector in selectors.items():
                try:
                    if selector.get("type") == "text":
                        value = await self.page.text_content(selector["selector"])
                    elif selector.get("type") == "attribute":
                        value = await self.page.get_attribute(selector["selector"], selector["attribute"])
                    elif selector.get("type") == "multiple":
                        elements = await self.page.query_selector_all(selector["selector"])
                        value = [await el.text_content() for el in elements]
                    else:
                        value = await self.page.text_content(selector["selector"])
                        
                    extracted_data[key] = value
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract {key}: {e}")
                    extracted_data[key] = None
                    
            return {
                "success": True,
                "data": {
                    "url": url,
                    "extracted_data": extracted_data
                }
            }
            
        except Exception as e:
            self.logger.error(f"DOM extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    async def _execute_data_processing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using various algorithms."""
        try:
            data = task.get("data", {})
            operation = task.get("operation")
            
            if operation == "filter":
                condition = task.get("condition")
                filtered_data = [item for item in data if eval(condition, {"item": item})]
                result = filtered_data
                
            elif operation == "transform":
                transform_func = task.get("transform_function")
                transformed_data = [eval(transform_func, {"item": item}) for item in data]
                result = transformed_data
                
            elif operation == "aggregate":
                field = task.get("field")
                agg_type = task.get("aggregation_type", "sum")
                
                if agg_type == "sum":
                    result = sum(item.get(field, 0) for item in data)
                elif agg_type == "average":
                    values = [item.get(field, 0) for item in data]
                    result = sum(values) / len(values) if values else 0
                elif agg_type == "count":
                    result = len(data)
                else:
                    raise ValueError(f"Unsupported aggregation type: {agg_type}")
                    
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
            return {
                "success": True,
                "data": {
                    "operation": operation,
                    "result": result,
                    "input_count": len(data) if isinstance(data, list) else 1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    async def _execute_file_operation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform file operations."""
        try:
            operation = task.get("operation")
            file_path = task.get("file_path")
            
            if operation == "read":
                with open(file_path, 'r') as f:
                    content = f.read()
                result = {"content": content}
                
            elif operation == "write":
                content = task.get("content", "")
                with open(file_path, 'w') as f:
                    f.write(content)
                result = {"file_path": file_path}
                
            elif operation == "delete":
                Path(file_path).unlink()
                result = {"deleted": file_path}
                
            elif operation == "copy":
                source = task.get("source")
                destination = task.get("destination")
                import shutil
                shutil.copy2(source, destination)
                result = {"copied": {"from": source, "to": destination}}
                
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
                
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"File operation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    async def _execute_general_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general tasks that don't fit other categories."""
        try:
            # For now, just return success with task info
            return {
                "success": True,
                "data": {
                    "task_type": task.get("type"),
                    "task_name": task.get("name"),
                    "message": "General task executed successfully"
                }
            }
            
        except Exception as e:
            self.logger.error(f"General task failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
            
    def is_busy(self) -> bool:
        """Check if the agent is currently busy."""
        return self.is_busy
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the execution agent."""
        return {
            "agent_id": self.agent_id,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "browser_type": self.config.browser_type,
            "uptime": (datetime.utcnow() - self.session_start_time).total_seconds() if self.session_start_time else 0
        }
        
    async def shutdown(self):
        """Shutdown the execution agent."""
        self.logger.info(f"Shutting down Execution Agent: {self.agent_id}")
        
        try:
            # Close browser
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
                
            # Shutdown media capture
            await self.media_capture.shutdown()
            
            # Shutdown drift detector
            await self.drift_detector.shutdown()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            
        self.logger.info(f"Execution Agent {self.agent_id} shutdown complete")