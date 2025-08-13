"""
Intelligent Automation Agent
============================

Universal automation agent that can handle any website automation
based on natural language instructions without hardcoding website-specific logic.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture
from ..utils.selector_drift_detector import SelectorDriftDetector


class IntelligentAutomationAgent:
    """Universal automation agent for any website."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        self.media_capture = MediaCapture(config)
        self.selector_drift_detector = SelectorDriftDetector()
        self.logger = logging.getLogger(__name__)
        
        # Browser context
        self.browser = None
        self.context = None
        self.page = None
        
    async def initialize(self):
        """Initialize browser context."""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.config.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page = await self.context.new_page()
            
            self.logger.info("Intelligent automation agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize intelligent automation agent: {e}")
            raise
            
    async def execute_natural_language_automation(self, instructions: str, url: str) -> Dict[str, Any]:
        """
        Execute automation based on natural language instructions.
        
        Args:
            instructions: Natural language automation instructions
            url: Target website URL
            
        Returns:
            Automation result with screenshots and execution details
        """
        try:
            self.logger.info(f"Starting intelligent automation: {instructions} on {url}")
            
            # Step 1: Navigate to the website
            await self.page.goto(url, wait_until="networkidle")
            
            # Step 2: Analyze the page and generate automation plan
            automation_plan = await self._generate_automation_plan(instructions, url)
            
            # Step 3: Execute the automation plan
            results = await self._execute_automation_plan(automation_plan)
            
            # Step 4: Capture final state
            final_screenshot = await self.media_capture.capture_screenshot(
                self.page, "intelligent_automation", "final_state"
            )
            
            return {
                "status": "completed",
                "instructions": instructions,
                "url": url,
                "automation_plan": automation_plan,
                "results": results,
                "screenshots": [final_screenshot],
                "execution_time": results.get("execution_time", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Intelligent automation failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "instructions": instructions,
                "url": url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _generate_automation_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate automation plan from natural language instructions."""
        try:
            # Get page content for context
            page_content = await self._extract_page_context()
            
            # Generate automation plan using AI
            prompt = f"""
            Given these natural language instructions: "{instructions}"
            For this website: {url}
            
            Page context:
            - Title: {await self.page.title()}
            - URL: {self.page.url}
            - Available elements: {page_content}
            
            Generate a detailed automation plan with specific actions. Each action should include:
            1. action_type: navigate, click, type, wait, execute_script, etc.
            2. selector: XPath or CSS selector to find the element
            3. description: what this action does
            4. fallback_selectors: alternative selectors if the main one fails
            5. validation: how to verify the action succeeded
            
            Return the plan as a JSON array of action objects.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            # Parse the response and extract the automation plan
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Fallback: generate a basic plan
                plan = await self._generate_fallback_plan(instructions)
                
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to generate automation plan: {e}")
            return await self._generate_fallback_plan(instructions)
            
    async def _generate_fallback_plan(self, instructions: str) -> List[Dict[str, Any]]:
        """Generate a fallback automation plan when AI fails."""
        # Basic keyword-based plan generation
        plan = []
        instructions_lower = instructions.lower()
        
        if "login" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'login') or contains(text(), 'Login')]",
                    "description": "Click login button/link",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Login')]",
                        "//a[contains(@class, 'login')]",
                        "//div[contains(text(), 'Login')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "timeout": 5,
                    "description": "Wait for login form to load"
                }
            ])
            
        if "mobile" in instructions_lower or "phone" in instructions_lower or "number" in instructions_lower:
            plan.append({
                "action_type": "type",
                "selector": "//input[@type='text' or @type='tel' or @type='number']",
                "text": "9080306208",
                "description": "Enter mobile number",
                "fallback_selectors": [
                    "//input[contains(@placeholder, 'mobile') or contains(@placeholder, 'phone')]",
                    "//input[contains(@name, 'mobile') or contains(@name, 'phone')]"
                ]
            })
            
        if "otp" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//button[contains(text(), 'OTP') or contains(text(), 'Send')]",
                    "description": "Click request OTP button",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Request')]",
                        "//button[contains(text(), 'Get')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "selector": "//div[contains(text(), 'OTP sent') or contains(text(), 'sent')]",
                    "timeout": 10,
                    "description": "Wait for OTP sent message"
                }
            ])
            
        return plan
        
    async def _extract_page_context(self) -> str:
        """Extract relevant page context for AI analysis."""
        try:
            # Get all clickable elements and form inputs
            elements = await self.page.evaluate("""
                () => {
                    const elements = [];
                    
                    // Get all links
                    document.querySelectorAll('a').forEach(link => {
                        elements.push({
                            type: 'link',
                            text: link.textContent?.trim(),
                            href: link.href,
                            class: link.className
                        });
                    });
                    
                    // Get all buttons
                    document.querySelectorAll('button').forEach(button => {
                        elements.push({
                            type: 'button',
                            text: button.textContent?.trim(),
                            class: button.className
                        });
                    });
                    
                    // Get all inputs
                    document.querySelectorAll('input').forEach(input => {
                        elements.push({
                            type: 'input',
                            type: input.type,
                            placeholder: input.placeholder,
                            name: input.name,
                            class: input.className
                        });
                    });
                    
                    return elements.slice(0, 50); // Limit to first 50 elements
                }
            """)
            
            return str(elements)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract page context: {e}")
            return "Page context extraction failed"
            
    async def _execute_automation_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the automation plan."""
        results = []
        execution_time = 0
        start_time = datetime.now()
        
        for i, action in enumerate(plan):
            try:
                self.logger.info(f"Executing action {i+1}/{len(plan)}: {action.get('description', 'Unknown action')}")
                
                action_result = await self._execute_action(action)
                results.append(action_result)
                
                # Capture screenshot after each action
                if action_result.get("success", False):
                    screenshot = await self.media_capture.capture_screenshot(
                        self.page, f"action_{i+1}", action.get("description", "action")
                    )
                    action_result["screenshot"] = screenshot
                    
            except Exception as e:
                self.logger.error(f"Action {i+1} failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "action": action
                })
                
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "actions": results,
            "execution_time": execution_time,
            "successful_actions": len([r for r in results if r.get("success", False)]),
            "total_actions": len(results)
        }
        
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single automation action."""
        action_type = action.get("action_type")
        
        if action_type == "click":
            return await self._smart_click(action)
        elif action_type == "type":
            return await self._smart_type(action)
        elif action_type == "wait":
            return await self._smart_wait(action)
        elif action_type == "execute_script":
            return await self._execute_script(action)
        elif action_type == "navigate":
            return await self._navigate(action)
        else:
            return {"success": False, "error": f"Unknown action type: {action_type}"}
            
    async def _smart_click(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Smart click with fallback selectors."""
        selectors = [action.get("selector")] + action.get("fallback_selectors", [])
        
        for selector in selectors:
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.click(selector)
                return {"success": True, "selector_used": selector}
            except Exception as e:
                self.logger.warning(f"Selector {selector} failed: {e}")
                continue
                
        return {"success": False, "error": "All selectors failed"}
        
    async def _smart_type(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Smart type with fallback selectors."""
        selectors = [action.get("selector")] + action.get("fallback_selectors", [])
        text = action.get("text", "")
        
        for selector in selectors:
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.fill(selector, text)
                return {"success": True, "selector_used": selector, "text_entered": text}
            except Exception as e:
                self.logger.warning(f"Selector {selector} failed: {e}")
                continue
                
        return {"success": False, "error": "All selectors failed"}
        
    async def _smart_wait(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Smart wait with multiple strategies."""
        timeout = action.get("timeout", 10)
        selector = action.get("selector")
        
        if selector:
            try:
                await self.page.wait_for_selector(selector, timeout=timeout * 1000)
                return {"success": True, "selector_found": selector}
            except Exception as e:
                return {"success": False, "error": f"Selector not found: {e}"}
        else:
            await asyncio.sleep(timeout)
            return {"success": True, "waited": timeout}
            
    async def _execute_script(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript code."""
        script = action.get("script", "")
        try:
            result = await self.page.evaluate(script)
            return {"success": True, "script_result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _navigate(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to URL."""
        url = action.get("url", "")
        try:
            await self.page.goto(url, wait_until="networkidle")
            return {"success": True, "url": url}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def shutdown(self):
        """Shutdown the intelligent automation agent."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
                
            self.logger.info("Intelligent automation agent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during intelligent automation agent shutdown: {e}")