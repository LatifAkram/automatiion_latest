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
from ..utils.selector_drift import SelectorDriftDetector


class IntelligentAutomationAgent:
    """Universal automation agent for any website."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        
        # Debug: Print config type and attributes
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Config type: {type(config)}")
        self.logger.info(f"Config attributes: {dir(config)}")
        
        # Initialize media capture with correct path
        if hasattr(config, 'database') and hasattr(config.database, 'media_path'):
            media_path = config.database.media_path
            self.logger.info(f"Using database media path: {media_path}")
        else:
            media_path = 'data/media'
            self.logger.info(f"Using default media path: {media_path}")
        
        self.media_capture = MediaCapture(media_path)
        # Initialize selector drift detector with automation config
        if hasattr(config, 'automation'):
            self.selector_drift_detector = SelectorDriftDetector(config.automation)
        else:
            # Fallback to a basic config
            self.selector_drift_detector = SelectorDriftDetector(config)
        
        # Browser context
        self.browser = None
        self.context = None
        self.page = None
        
    async def initialize(self):
        """Initialize browser context."""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            # Get headless setting from automation config
            headless = getattr(self.config.automation, 'headless', True)
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
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
            # Use sector specialists for intelligent plan generation
            from .sector_specialists import SectorManager
            
            sector_manager = SectorManager(self.config, self.ai_provider)
            universal_plan = await sector_manager.generate_universal_plan(instructions, url)
            
            self.logger.info(f"Generated plan for sector: {universal_plan['sector']}")
            return universal_plan['plan']
            
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
            
    async def execute_automation_plan(self, instructions: str, url: str) -> Dict[str, Any]:
        """Execute automation plan with parallel DOM analysis."""
        try:
            self.logger.info(f"Executing automation plan for: {instructions}")
            
            # Navigate to URL
            await self.page.goto(url, wait_until='networkidle')
            await asyncio.sleep(2)
            
            # Perform parallel DOM analysis
            dom_analysis = await self._parallel_dom_analysis()
            self.logger.info(f"DOM analysis completed: {len(dom_analysis)} elements found")
            
            # Generate intelligent selectors based on DOM analysis
            intelligent_selectors = await self._generate_intelligent_selectors(instructions, dom_analysis)
            
            # Execute automation with intelligent selectors
            results = await self._execute_with_intelligent_selectors(instructions, intelligent_selectors)
            
            return {
                "status": "completed",
                "summary": f"Automation completed successfully with {len(results)} actions",
                "results": results,
                "dom_analysis": dom_analysis,
                "intelligent_selectors": intelligent_selectors
            }
            
        except Exception as e:
            self.logger.error(f"Automation execution failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "summary": "Automation failed"
            }

    async def _parallel_dom_analysis(self) -> Dict[str, Any]:
        """Perform parallel DOM analysis using multiple strategies."""
        try:
            # Parallel DOM analysis tasks
            tasks = [
                self._analyze_form_elements(),
                self._analyze_interactive_elements(),
                self._analyze_navigation_elements(),
                self._analyze_content_elements(),
                self._analyze_structural_elements()
            ]
            
            # Execute all analysis tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            dom_analysis = {
                "forms": results[0] if not isinstance(results[0], Exception) else [],
                "interactive": results[1] if not isinstance(results[1], Exception) else [],
                "navigation": results[2] if not isinstance(results[2], Exception) else [],
                "content": results[3] if not isinstance(results[3], Exception) else [],
                "structural": results[4] if not isinstance(results[4], Exception) else []
            }
            
            return dom_analysis
            
        except Exception as e:
            self.logger.error(f"Parallel DOM analysis failed: {e}")
            return {}

    async def _analyze_form_elements(self) -> List[Dict[str, Any]]:
        """Analyze form elements in parallel."""
        try:
            # Multiple selector strategies for forms
            selectors = [
                "input[type='text'], input[type='email'], input[type='tel'], input[type='number'], input[type='password']",
                "textarea",
                "select",
                "button[type='submit'], input[type='submit']",
                "form"
            ]
            
            form_elements = []
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            tag_name = await element.get_attribute('tagName')
                            input_type = await element.get_attribute('type')
                            placeholder = await element.get_attribute('placeholder')
                            name = await element.get_attribute('name')
                            id_attr = await element.get_attribute('id')
                            
                            form_elements.append({
                                "tag": tag_name,
                                "type": input_type,
                                "placeholder": placeholder,
                                "name": name,
                                "id": id_attr,
                                "selector": selector,
                                "xpath": await self._get_xpath(element)
                            })
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
                    
            return form_elements
            
        except Exception as e:
            self.logger.error(f"Form analysis failed: {e}")
            return []

    async def _analyze_interactive_elements(self) -> List[Dict[str, Any]]:
        """Analyze interactive elements in parallel."""
        try:
            selectors = [
                "button",
                "a",
                "input[type='button'], input[type='submit']",
                "[onclick]",
                "[role='button']"
            ]
            
            interactive_elements = []
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            tag_name = await element.get_attribute('tagName')
                            text_content = await element.text_content()
                            onclick = await element.get_attribute('onclick')
                            role = await element.get_attribute('role')
                            
                            interactive_elements.append({
                                "tag": tag_name,
                                "text": text_content,
                                "onclick": onclick,
                                "role": role,
                                "selector": selector,
                                "xpath": await self._get_xpath(element)
                            })
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
                    
            return interactive_elements
            
        except Exception as e:
            self.logger.error(f"Interactive analysis failed: {e}")
            return []

    async def _analyze_navigation_elements(self) -> List[Dict[str, Any]]:
        """Analyze navigation elements in parallel."""
        try:
            selectors = [
                "nav",
                "a[href]",
                "[role='navigation']",
                ".nav, .navigation, .menu",
                "ul li a"
            ]
            
            nav_elements = []
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            tag_name = await element.get_attribute('tagName')
                            href = await element.get_attribute('href')
                            text_content = await element.text_content()
                            
                            nav_elements.append({
                                "tag": tag_name,
                                "href": href,
                                "text": text_content,
                                "selector": selector,
                                "xpath": await self._get_xpath(element)
                            })
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
                    
            return nav_elements
            
        except Exception as e:
            self.logger.error(f"Navigation analysis failed: {e}")
            return []

    async def _analyze_content_elements(self) -> List[Dict[str, Any]]:
        """Analyze content elements in parallel."""
        try:
            selectors = [
                "h1, h2, h3, h4, h5, h6",
                "p",
                "div[class*='content'], div[class*='text']",
                "span[class*='content'], span[class*='text']"
            ]
            
            content_elements = []
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            tag_name = await element.get_attribute('tagName')
                            text_content = await element.text_content()
                            class_name = await element.get_attribute('class')
                            
                            content_elements.append({
                                "tag": tag_name,
                                "text": text_content,
                                "class": class_name,
                                "selector": selector,
                                "xpath": await self._get_xpath(element)
                            })
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
                    
            return content_elements
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return []

    async def _analyze_structural_elements(self) -> List[Dict[str, Any]]:
        """Analyze structural elements in parallel."""
        try:
            selectors = [
                "header",
                "footer",
                "main",
                "section",
                "article",
                "aside"
            ]
            
            structural_elements = []
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            tag_name = await element.get_attribute('tagName')
                            class_name = await element.get_attribute('class')
                            id_attr = await element.get_attribute('id')
                            
                            structural_elements.append({
                                "tag": tag_name,
                                "class": class_name,
                                "id": id_attr,
                                "selector": selector,
                                "xpath": await self._get_xpath(element)
                            })
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
                    
            return structural_elements
            
        except Exception as e:
            self.logger.error(f"Structural analysis failed: {e}")
            return []

    async def _get_xpath(self, element) -> str:
        """Get XPath for an element."""
        try:
            return await element.evaluate('(element) => { return getXPath(element); }')
        except:
            return ""

    async def _generate_intelligent_selectors(self, instructions: str, dom_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate intelligent selectors based on DOM analysis and instructions."""
        try:
            intelligent_selectors = {
                "input_selectors": [],
                "button_selectors": [],
                "navigation_selectors": [],
                "content_selectors": []
            }
            
            # Analyze instructions for keywords
            instructions_lower = instructions.lower()
            
            # Generate input selectors
            if any(word in instructions_lower for word in ['login', 'email', 'phone', 'mobile', 'password', 'username']):
                for form_element in dom_analysis.get("forms", []):
                    if form_element.get("tag") == "INPUT":
                        input_type = form_element.get("type", "")
                        placeholder = form_element.get("placeholder", "").lower()
                        name = form_element.get("name", "").lower()
                        id_attr = form_element.get("id", "").lower()
                        
                        # Smart selector generation
                        if input_type in ['text', 'email', 'tel', 'number']:
                            selectors = []
                            
                            # Priority 1: Specific attributes
                            if id_attr:
                                selectors.append(f"#{id_attr}")
                            if name:
                                selectors.append(f"[name='{form_element.get('name')}']")
                            if placeholder:
                                selectors.append(f"[placeholder='{form_element.get('placeholder')}']")
                            
                            # Priority 2: Type-based selectors
                            selectors.append(f"input[type='{input_type}']")
                            
                            # Priority 3: XPath
                            if form_element.get("xpath"):
                                selectors.append(form_element.get("xpath"))
                            
                            intelligent_selectors["input_selectors"].extend(selectors)
            
            # Generate button selectors
            if any(word in instructions_lower for word in ['click', 'submit', 'button', 'login', 'search']):
                for interactive_element in dom_analysis.get("interactive", []):
                    if interactive_element.get("tag") in ['BUTTON', 'INPUT']:
                        text_content = interactive_element.get("text", "").lower()
                        
                        # Smart button selector generation
                        selectors = []
                        
                        # Priority 1: Text-based selectors
                        if text_content:
                            selectors.append(f"button:has-text('{text_content}')")
                            selectors.append(f"input[value='{text_content}']")
                        
                        # Priority 2: XPath
                        if interactive_element.get("xpath"):
                            selectors.append(interactive_element.get("xpath"))
                        
                        intelligent_selectors["button_selectors"].extend(selectors)
            
            return intelligent_selectors
            
        except Exception as e:
            self.logger.error(f"Intelligent selector generation failed: {e}")
            return {}

    async def _execute_with_intelligent_selectors(self, instructions: str, intelligent_selectors: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Execute automation using intelligent selectors."""
        try:
            results = []
            instructions_lower = instructions.lower()
            
            # Execute input actions
            if any(word in instructions_lower for word in ['login', 'email', 'phone', 'mobile', 'password', 'username']):
                for selector in intelligent_selectors.get("input_selectors", []):
                    try:
                        element = await self.page.wait_for_selector(selector, timeout=3000)
                        if element:
                            await element.click()
                            await element.fill("test_input")
                            results.append({
                                "action": "input",
                                "selector": selector,
                                "status": "success"
                            })
                            break
                    except Exception as e:
                        continue
            
            # Execute button actions
            if any(word in instructions_lower for word in ['click', 'submit', 'button', 'login', 'search']):
                for selector in intelligent_selectors.get("button_selectors", []):
                    try:
                        element = await self.page.wait_for_selector(selector, timeout=3000)
                        if element:
                            await element.click()
                            results.append({
                                "action": "click",
                                "selector": selector,
                                "status": "success"
                            })
                            break
                    except Exception as e:
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Intelligent execution failed: {e}")
            return []
            
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