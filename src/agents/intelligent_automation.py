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
from ..agents.ai_dom_analyzer import AIDOMAnalyzer
from ..utils.advanced_learning import AdvancedLearningSystem


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
        
        # Initialize AI DOM analyzer (will be fully initialized in initialize method)
        self.ai_dom_analyzer = None
        
        # Initialize advanced learning system (will be fully initialized in initialize method)
        self.advanced_learning = None
        
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
            
            # Initialize AI DOM analyzer with required dependencies
            from ..core.vector_store import VectorStore
            vector_store = VectorStore(self.config.database)
            await vector_store.initialize()
            
            self.ai_dom_analyzer = AIDOMAnalyzer(
                config=self.config,
                ai_provider=self.ai_provider,
                vector_store=vector_store,
                selector_drift_detector=self.selector_drift_detector
            )
            
            # Initialize advanced learning system
            self.advanced_learning = AdvancedLearningSystem(
                config=self.config,
                ai_provider=self.ai_provider,
                vector_store=vector_store
            )
            
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
            
            # Step 2: AI-powered DOM analysis and automation plan generation
            ai_dom_analysis = await self.ai_dom_analyzer.analyze_dom_with_ai(self.page, instructions)
            automation_plan = await self._generate_automation_plan_with_ai(instructions, url, ai_dom_analysis)
            
            # Step 3: Predict success and optimize plan
            success_prediction = await self.advanced_learning.predict_automation_success({
                "domain": url.split("//")[1].split("/")[0] if "//" in url else url,
                "automation_type": "web_automation",
                "complexity": "medium",
                "selectors": automation_plan.get("steps", []),
                "ai_confidence": automation_plan.get("ai_analysis", {}).get("confidence_score", 0.5)
            })
            
            # Optimize plan using learned patterns
            optimization_result = await self.advanced_learning.optimize_automation_plan(automation_plan)
            optimized_plan = optimization_result.get("optimized_plan", automation_plan)
            
            # Step 4: Execute the optimized automation plan
            results = await self._execute_automation_plan_with_learning(optimized_plan, success_prediction)
            
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
            
    async def _generate_automation_plan_with_ai(self, instructions: str, url: str, ai_dom_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automation plan using AI-powered DOM analysis."""
        try:
            # Extract intelligent selectors from AI analysis
            intelligent_selectors = ai_dom_analysis.get("intelligent_selectors", {})
            ai_analysis = ai_dom_analysis.get("ai_analysis", {})
            learned_patterns = ai_dom_analysis.get("learned_patterns", {})
            
            # Use AI to generate plan with intelligent selectors
            prompt = f"""
            Generate a detailed automation plan using AI-analyzed DOM elements:
            
            Instructions: {instructions}
            URL: {url}
            
            AI DOM Analysis Results:
            - Login Elements: {len(ai_analysis.get('login_elements', []))}
            - Form Elements: {len(ai_analysis.get('form_elements', []))}
            - Navigation Elements: {len(ai_analysis.get('navigation_elements', []))}
            - Interactive Elements: {len(ai_analysis.get('interactive_elements', []))}
            
            Intelligent Selectors Available:
            - Input Selectors: {intelligent_selectors.get('input_selectors', [])}
            - Button Selectors: {intelligent_selectors.get('button_selectors', [])}
            - Navigation Selectors: {intelligent_selectors.get('navigation_selectors', [])}
            
            Learned Patterns Applied:
            - Successful Selectors: {len(learned_patterns.get('successful_selectors', []))}
            - Domain-Specific Patterns: {len(learned_patterns.get('domain_specific', []))}
            
            Create an intelligent automation plan that:
            1. Uses the best available selectors from AI analysis
            2. Incorporates learned patterns for better success
            3. Includes fallback strategies for each step
            4. Provides comprehensive error handling
            5. Uses advanced learning and auto-heal capabilities
            
            Return as JSON with structure:
            {{
                "steps": [
                    {{
                        "step": 1,
                        "action": "navigate",
                        "description": "Navigate to the website",
                        "url": "{url}",
                        "expected_result": "Page loaded successfully",
                        "ai_confidence": 0.95
                    }},
                    {{
                        "step": 2,
                        "action": "click",
                        "description": "Click login button using AI-identified selector",
                        "primary_selector": "best_selector_from_ai",
                        "fallback_selectors": ["alternative1", "alternative2"],
                        "ai_confidence": 0.88,
                        "learned_pattern": "successful_pattern_from_vector_store"
                    }}
                ],
                "ai_analysis": {{
                    "total_elements_analyzed": "count",
                    "confidence_score": "overall_confidence",
                    "learning_applied": "yes/no"
                }},
                "validation": [
                    {{
                        "type": "ai_validated_element",
                        "selector": "ai_generated_selector",
                        "description": "AI-validated element presence"
                    }}
                ],
                "error_handling": [
                    {{
                        "condition": "selector_failure",
                        "action": "auto_heal_with_ai",
                        "fallback_strategy": "use_learned_patterns"
                    }}
                ],
                "learning_opportunities": [
                    {{
                        "pattern": "new_pattern_to_learn",
                        "context": "when_to_apply"
                    }}
                ]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                import json
                plan = json.loads(response)
                self.logger.info(f"Generated AI-powered automation plan with {len(plan.get('steps', []))} steps")
                return plan
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse AI response as JSON, using fallback plan")
                return await self._generate_fallback_plan_with_ai(instructions, url, ai_dom_analysis)
                
        except Exception as e:
            self.logger.error(f"Failed to generate AI automation plan: {e}")
            return await self._generate_fallback_plan_with_ai(instructions, url, ai_dom_analysis)

    async def _generate_fallback_plan_with_ai(self, instructions: str, url: str, ai_dom_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback automation plan using AI analysis data."""
        try:
            intelligent_selectors = ai_dom_analysis.get("intelligent_selectors", {})
            
            # Create basic plan using available selectors
            steps = []
            
            # Add navigation step
            steps.append({
                "step": 1,
                "action": "navigate",
                "description": f"Navigate to {url}",
                "url": url,
                "expected_result": "Page loaded successfully",
                "ai_confidence": 0.9
            })
            
            # Add steps based on available selectors
            step_number = 2
            
            # Input actions
            for selector in intelligent_selectors.get("input_selectors", [])[:3]:  # Limit to 3 inputs
                steps.append({
                    "step": step_number,
                    "action": "input",
                    "description": f"Fill input field using selector: {selector}",
                    "primary_selector": selector,
                    "fallback_selectors": intelligent_selectors.get("input_selectors", [])[1:],
                    "ai_confidence": 0.8,
                    "learned_pattern": "input_field_pattern"
                })
                step_number += 1
            
            # Button actions
            for selector in intelligent_selectors.get("button_selectors", [])[:2]:  # Limit to 2 buttons
                steps.append({
                    "step": step_number,
                    "action": "click",
                    "description": f"Click button using selector: {selector}",
                    "primary_selector": selector,
                    "fallback_selectors": intelligent_selectors.get("button_selectors", [])[1:],
                    "ai_confidence": 0.85,
                    "learned_pattern": "button_click_pattern"
                })
                step_number += 1
            
            return {
                "steps": steps,
                "ai_analysis": {
                    "total_elements_analyzed": len(ai_dom_analysis.get("page_structure", {}).get("elements", [])),
                    "confidence_score": 0.75,
                    "learning_applied": "yes"
                },
                "validation": [
                    {
                        "type": "element_present",
                        "selector": "body",
                        "description": "Verify page loaded"
                    }
                ],
                "error_handling": [
                    {
                        "condition": "selector_failure",
                        "action": "try_fallback_selectors",
                        "fallback_strategy": "use_learned_patterns"
                    }
                ],
                "learning_opportunities": [
                    {
                        "pattern": "new_selector_pattern",
                        "context": "when_selector_fails"
                    }
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate fallback AI plan: {e}")
            return await self._generate_fallback_plan(instructions)
        
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

    async def _execute_automation_plan_with_learning(self, automation_plan: Dict[str, Any], success_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automation plan with advanced learning and auto-heal capabilities."""
        try:
            self.logger.info("Executing automation plan with advanced learning")
            
            start_time = datetime.utcnow()
            steps = automation_plan.get("steps", [])
            completed_steps = []
            errors = []
            selectors_used = []
            
            # Execute each step with learning and auto-heal
            for i, step in enumerate(steps):
                try:
                    self.logger.info(f"Executing step {i+1}/{len(steps)}: {step.get('action', 'unknown')}")
                    
                    # Execute step with auto-heal
                    step_result = await self._execute_step_with_auto_heal(step, success_prediction)
                    
                    if step_result["success"]:
                        completed_steps.append(step_result)
                        selectors_used.extend(step_result.get("selectors_used", []))
                        
                        # Capture screenshot after successful step
                        screenshot = await self.media_capture.capture_screenshot(
                            self.page, f"step_{i+1}_{step.get('action', 'unknown')}", "automation_progress"
                        )
                        step_result["screenshot"] = screenshot
                        
                    else:
                        errors.append({
                            "step": i+1,
                            "action": step.get("action"),
                            "error": step_result.get("error"),
                            "selector": step.get("primary_selector")
                        })
                        
                        # Try auto-heal for failed step
                        healed_result = await self._auto_heal_failed_step(step, step_result)
                        if healed_result["success"]:
                            completed_steps.append(healed_result)
                            selectors_used.extend(healed_result.get("selectors_used", []))
                            self.logger.info(f"Step {i+1} healed successfully")
                        else:
                            self.logger.error(f"Step {i+1} failed even after auto-heal")
                    
                    # Small delay between steps
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Step {i+1} execution failed: {e}")
                    errors.append({
                        "step": i+1,
                        "action": step.get("action"),
                        "error": str(e),
                        "selector": step.get("primary_selector")
                    })
            
            # Calculate execution metrics
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            success = len(errors) == 0
            
            # Prepare execution result for learning
            execution_result = {
                "success": success,
                "execution_time": execution_time,
                "completed_steps": completed_steps,
                "all_steps": steps,
                "errors": errors,
                "selectors_used": selectors_used,
                "domain": automation_plan.get("domain", ""),
                "automation_type": "web_automation",
                "ai_confidence": success_prediction.get("success_probability", 0.5)
            }
            
            # Learn from execution
            learning_result = await self.advanced_learning.learn_from_execution(execution_result)
            
            return {
                "success": success,
                "execution_time": execution_time,
                "completed_steps": len(completed_steps),
                "total_steps": len(steps),
                "errors": errors,
                "screenshots": [step.get("screenshot") for step in completed_steps if step.get("screenshot")],
                "learning_applied": learning_result.get("learning_applied", False),
                "performance_improvement": learning_result.get("performance_improvement", 0),
                "suggestions": learning_result.get("suggestions", []),
                "success_prediction": success_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Automation execution with learning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "completed_steps": 0,
                "total_steps": 0
            }

    async def _execute_step_with_auto_heal(self, step: Dict[str, Any], success_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step with auto-heal capabilities."""
        try:
            action = step.get("action", "")
            primary_selector = step.get("primary_selector", "")
            fallback_selectors = step.get("fallback_selectors", [])
            
            # Try primary selector first
            if primary_selector:
                try:
                    result = await self._execute_action(action, primary_selector, step)
                    if result["success"]:
                        return {
                            "success": True,
                            "action": action,
                            "selector_used": primary_selector,
                            "selectors_used": [primary_selector],
                            "result": result
                        }
                except Exception as e:
                    self.logger.warning(f"Primary selector failed: {e}")
            
            # Try fallback selectors
            for fallback_selector in fallback_selectors:
                try:
                    result = await self._execute_action(action, fallback_selector, step)
                    if result["success"]:
                        return {
                            "success": True,
                            "action": action,
                            "selector_used": fallback_selector,
                            "selectors_used": [primary_selector, fallback_selector],
                            "result": result
                        }
                except Exception as e:
                    self.logger.warning(f"Fallback selector failed: {e}")
            
            # All selectors failed
            return {
                "success": False,
                "action": action,
                "error": "All selectors failed",
                "selectors_used": [primary_selector] + fallback_selectors
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": step.get("action"),
                "error": str(e),
                "selectors_used": []
            }

    async def _auto_heal_failed_step(self, step: Dict[str, Any], step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-heal a failed step using advanced learning."""
        try:
            failed_selector = step_result.get("selector_used", step.get("primary_selector", ""))
            
            # Context for auto-heal
            context = {
                "domain": step.get("domain", ""),
                "element_type": step.get("action", ""),
                "page_context": "automation_execution",
                "step_number": step.get("step", 0)
            }
            
            # Try auto-healing the selector
            healed_selector = await self.advanced_learning.auto_heal_selector(failed_selector, context)
            
            if healed_selector:
                # Try executing with healed selector
                result = await self._execute_action(step.get("action", ""), healed_selector, step)
                if result["success"]:
                    return {
                        "success": True,
                        "action": step.get("action"),
                        "selector_used": healed_selector,
                        "selectors_used": [failed_selector, healed_selector],
                        "healed": True,
                        "result": result
                    }
            
            return {
                "success": False,
                "action": step.get("action"),
                "error": "Auto-heal failed",
                "selectors_used": [failed_selector]
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": step.get("action"),
                "error": f"Auto-heal error: {str(e)}",
                "selectors_used": []
            }

    async def _execute_action(self, action: str, selector: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action with a selector."""
        try:
            if action == "click":
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    await element.click()
                    return {"success": True, "action": "click"}
                else:
                    return {"success": False, "error": "Element not found"}
                    
            elif action == "type":
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    text = step.get("text", "test_input")
                    await element.fill(text)
                    return {"success": True, "action": "type", "text": text}
                else:
                    return {"success": False, "error": "Element not found"}
                    
            elif action == "input":
                element = await self.page.wait_for_selector(selector, timeout=5000)
                if element:
                    text = step.get("text", "test_input")
                    await element.fill(text)
                    return {"success": True, "action": "input", "text": text}
                else:
                    return {"success": False, "error": "Element not found"}
                    
            elif action == "wait":
                await asyncio.sleep(step.get("timeout", 2))
                return {"success": True, "action": "wait"}
                
            elif action == "navigate":
                url = step.get("url", "")
                if url:
                    await self.page.goto(url, wait_until="networkidle")
                    return {"success": True, "action": "navigate", "url": url}
                else:
                    return {"success": False, "error": "No URL provided"}
                    
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
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