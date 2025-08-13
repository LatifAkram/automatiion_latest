"""
AI-2: DOM Analysis Agent
========================

Fetches and analyzes DOM from website URLs based on planned steps.
Uses multiple AI providers (GPT, Claude, Gemini, Local LLM) for speed and accuracy.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from playwright.async_api import Page, ElementHandle

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture
from ..utils.selector_drift import SelectorDriftDetector


class DOMAnalysisResult:
    """Result of DOM analysis."""
    
    def __init__(self, url: str, elements: List[Dict], selectors: List[Dict], 
                 page_structure: Dict, analysis_metadata: Dict):
        self.url = url
        self.elements = elements
        self.selectors = selectors
        self.page_structure = page_structure
        self.analysis_metadata = analysis_metadata
        self.timestamp = datetime.utcnow().isoformat()


class AIDOMAnalysisAgent:
    """AI-2: DOM Analysis Agent for intelligent element detection and selector generation."""
    
    def __init__(self, config, ai_provider: AIProvider, media_capture: MediaCapture):
        self.config = config
        self.ai_provider = ai_provider
        self.media_capture = media_capture
        self.selector_drift_detector = SelectorDriftDetector(config.automation)
        self.logger = logging.getLogger(__name__)
        
        # Analysis cache for performance
        self.analysis_cache = {}
        self.selector_cache = {}
        
    async def initialize(self):
        """Initialize the DOM analysis agent."""
        self.logger.info("Initializing AI-2 DOM Analysis Agent...")
        await self.selector_drift_detector.initialize()
        self.logger.info("AI-2 DOM Analysis Agent initialized successfully")
    
    async def analyze_dom_for_automation(self, page: Page, planned_steps: List[Dict], 
                                       context: Dict[str, Any]) -> DOMAnalysisResult:
        """
        Analyze DOM based on planned automation steps.
        
        Args:
            page: Playwright page object
            planned_steps: Steps from AI-1 planner
            context: Additional context information
            
        Returns:
            DOMAnalysisResult with comprehensive analysis
        """
        try:
            self.logger.info(f"AI-2: Analyzing DOM for {len(planned_steps)} planned steps")
            
            # Step 1: Extract comprehensive page information
            page_info = await self._extract_page_information(page)
            
            # Step 2: Analyze DOM structure based on planned steps
            dom_analysis = await self._analyze_dom_structure(page, planned_steps, context)
            
            # Step 3: Generate intelligent selectors
            selectors = await self._generate_intelligent_selectors(page, planned_steps, dom_analysis)
            
            # Step 4: Validate and optimize selectors
            validated_selectors = await self._validate_and_optimize_selectors(page, selectors)
            
            # Step 5: Capture page state
            screenshot = await self.media_capture.capture_screenshot(page, "dom_analysis", "current_state")
            
            # Create analysis result
            result = DOMAnalysisResult(
                url=page.url,
                elements=dom_analysis["elements"],
                selectors=validated_selectors,
                page_structure=page_info,
                analysis_metadata={
                    "planned_steps": len(planned_steps),
                    "elements_found": len(dom_analysis["elements"]),
                    "selectors_generated": len(validated_selectors),
                    "analysis_time": datetime.utcnow().isoformat(),
                    "screenshot_path": screenshot,
                    "ai_providers_used": ["gpt", "claude", "gemini", "local_llm"]
                }
            )
            
            self.logger.info(f"AI-2: DOM analysis completed - {len(validated_selectors)} selectors generated")
            return result
            
        except Exception as e:
            self.logger.error(f"AI-2: DOM analysis failed: {e}")
            return self._create_fallback_result(page.url)
    
    async def _extract_page_information(self, page: Page) -> Dict[str, Any]:
        """Extract comprehensive page information."""
        try:
            # Get page metadata
            page_metadata = await page.evaluate("""
                () => ({
                    title: document.title,
                    url: window.location.href,
                    domain: window.location.hostname,
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    userAgent: navigator.userAgent,
                    language: navigator.language,
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
                })
            """)
            
            # Get all elements with their properties
            elements_data = await page.evaluate("""
                () => {
                    const elements = [];
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_ELEMENT,
                        null,
                        false
                    );
                    
                    let node;
                    while (node = walker.nextNode()) {
                        const element = node;
                        const rect = element.getBoundingClientRect();
                        const computedStyle = window.getComputedStyle(element);
                        
                        elements.push({
                            tagName: element.tagName.toLowerCase(),
                            id: element.id,
                            className: element.className,
                            textContent: element.textContent?.trim().substring(0, 200),
                            attributes: Array.from(element.attributes).map(attr => ({
                                name: attr.name,
                                value: attr.value
                            })),
                            isVisible: rect.width > 0 && rect.height > 0,
                            position: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            },
                            computedStyle: {
                                backgroundColor: computedStyle.backgroundColor,
                                color: computedStyle.color,
                                fontSize: computedStyle.fontSize,
                                fontWeight: computedStyle.fontWeight,
                                display: computedStyle.display,
                                visibility: computedStyle.visibility,
                                opacity: computedStyle.opacity
                            },
                            aria: {
                                label: element.getAttribute('aria-label'),
                                describedby: element.getAttribute('aria-describedby'),
                                role: element.getAttribute('role')
                            }
                        });
                    }
                    return elements;
                }
            """)
            
            return {
                "metadata": page_metadata,
                "elements": elements_data,
                "total_elements": len(elements_data),
                "visible_elements": len([e for e in elements_data if e["isVisible"]])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract page information: {e}")
            return {"error": str(e)}
    
    async def _analyze_dom_structure(self, page: Page, planned_steps: List[Dict], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DOM structure based on planned steps using multiple AI providers."""
        try:
            # Create analysis tasks for different AI providers
            analysis_tasks = []
            
            # Task 1: Analyze with GPT
            gpt_task = asyncio.create_task(
                self._analyze_with_ai_provider("gpt", page, planned_steps, context)
            )
            analysis_tasks.append(("gpt", gpt_task))
            
            # Task 2: Analyze with Claude
            claude_task = asyncio.create_task(
                self._analyze_with_ai_provider("claude", page, planned_steps, context)
            )
            analysis_tasks.append(("claude", claude_task))
            
            # Task 3: Analyze with Gemini
            gemini_task = asyncio.create_task(
                self._analyze_with_ai_provider("gemini", page, planned_steps, context)
            )
            analysis_tasks.append(("gemini", gemini_task))
            
            # Task 4: Analyze with Local LLM
            local_task = asyncio.create_task(
                self._analyze_with_ai_provider("local_llm", page, planned_steps, context)
            )
            analysis_tasks.append(("local_llm", local_task))
            
            # Wait for all analyses to complete
            results = {}
            for provider, task in analysis_tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=30)
                    results[provider] = result
                except asyncio.TimeoutError:
                    self.logger.warning(f"AI-2: {provider} analysis timed out")
                    results[provider] = {"error": "timeout"}
                except Exception as e:
                    self.logger.warning(f"AI-2: {provider} analysis failed: {e}")
                    results[provider] = {"error": str(e)}
            
            # Combine and validate results
            combined_analysis = await self._combine_ai_analyses(results, planned_steps)
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"DOM structure analysis failed: {e}")
            return {"error": str(e), "elements": []}
    
    async def _analyze_with_ai_provider(self, provider: str, page: Page, planned_steps: List[Dict], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DOM with a specific AI provider."""
        try:
            # Get page content for analysis
            page_content = await page.content()
            page_info = await self._extract_page_information(page)
            
            # Create analysis prompt
            prompt = f"""
            Analyze this web page for automation based on planned steps:
            
            Page URL: {page.url}
            Page Title: {page_info.get('metadata', {}).get('title', 'Unknown')}
            Planned Steps: {json.dumps(planned_steps, indent=2)}
            Context: {json.dumps(context, indent=2)}
            
            Page Content (first 2000 chars): {page_content[:2000]}
            
            Visible Elements: {len(page_info.get('elements', []))} elements found
            
            Analyze and identify:
            1. Elements needed for each planned step
            2. Best selectors for each element (ID, class, XPath, text-based)
            3. Element relationships and hierarchy
            4. Potential automation challenges
            5. Recommended interaction strategies
            
            Return as JSON:
            {{
                "elements_for_steps": [
                    {{
                        "step_number": 1,
                        "step_action": "click",
                        "elements": [
                            {{
                                "tagName": "button",
                                "textContent": "Login",
                                "selectors": ["#login-btn", ".login-button", "//button[text()='Login']"],
                                "confidence": 0.95,
                                "interaction_type": "click"
                            }}
                        ]
                    }}
                ],
                "page_structure": {{
                    "main_sections": ["header", "main", "footer"],
                    "interactive_areas": ["navigation", "forms", "buttons"],
                    "data_containers": ["tables", "lists", "cards"]
                }},
                "automation_challenges": ["challenge1", "challenge2"],
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            # Use specific AI provider
            if provider == "gpt":
                response = await self.ai_provider._call_openai(prompt, 2000, 0.3)
            elif provider == "claude":
                response = await self.ai_provider._call_anthropic(prompt, 2000, 0.3)
            elif provider == "gemini":
                response = await self.ai_provider._call_google(prompt, 2000, 0.3)
            elif provider == "local_llm":
                response = await self.ai_provider._call_local_llm(prompt, 2000, 0.3)
            else:
                response = await self.ai_provider.generate_response(prompt, 2000, 0.3)
            
            # Parse response
            try:
                analysis = json.loads(response)
                analysis["provider"] = provider
                analysis["analysis_time"] = datetime.utcnow().isoformat()
                return analysis
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "provider": provider}
                
        except Exception as e:
            return {"error": str(e), "provider": provider}
    
    async def _combine_ai_analyses(self, results: Dict[str, Dict], planned_steps: List[Dict]) -> Dict[str, Any]:
        """Combine analyses from multiple AI providers."""
        try:
            # Filter successful analyses
            successful_analyses = {k: v for k, v in results.items() if "error" not in v}
            
            if not successful_analyses:
                return {"error": "All AI analyses failed", "elements": []}
            
            # Combine elements from all providers
            all_elements = []
            for provider, analysis in successful_analyses.items():
                if "elements_for_steps" in analysis:
                    for step_elements in analysis["elements_for_steps"]:
                        all_elements.extend(step_elements.get("elements", []))
            
            # Remove duplicates and rank by confidence
            unique_elements = self._deduplicate_elements(all_elements)
            
            # Create combined analysis
            combined = {
                "elements": unique_elements,
                "providers_used": list(successful_analyses.keys()),
                "total_elements": len(unique_elements),
                "analysis_quality": len(successful_analyses) / len(results),
                "combined_at": datetime.utcnow().isoformat()
            }
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Failed to combine AI analyses: {e}")
            return {"error": str(e), "elements": []}
    
    def _deduplicate_elements(self, elements: List[Dict]) -> List[Dict]:
        """Remove duplicate elements and rank by confidence."""
        seen = set()
        unique_elements = []
        
        for element in elements:
            # Create unique key based on element properties
            key = f"{element.get('tagName', '')}-{element.get('textContent', '')[:50]}"
            
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)
        
        # Sort by confidence
        unique_elements.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return unique_elements
    
    async def _generate_intelligent_selectors(self, page: Page, planned_steps: List[Dict], 
                                           dom_analysis: Dict[str, Any]) -> List[Dict]:
        """Generate intelligent selectors for automation."""
        try:
            selectors = []
            
            for step in planned_steps:
                step_selectors = await self._generate_selectors_for_step(page, step, dom_analysis)
                selectors.extend(step_selectors)
            
            return selectors
            
        except Exception as e:
            self.logger.error(f"Selector generation failed: {e}")
            return []
    
    async def _generate_selectors_for_step(self, page: Page, step: Dict, 
                                         dom_analysis: Dict[str, Any]) -> List[Dict]:
        """Generate selectors for a specific step."""
        try:
            action = step.get("action", "")
            description = step.get("description", "")
            
            # Find relevant elements for this step
            relevant_elements = []
            for element in dom_analysis.get("elements", []):
                if self._element_matches_step(element, step):
                    relevant_elements.append(element)
            
            # Generate selectors for each relevant element
            step_selectors = []
            for element in relevant_elements:
                selectors = await self._generate_element_selectors(page, element, action)
                step_selectors.extend(selectors)
            
            return step_selectors
            
        except Exception as e:
            self.logger.error(f"Step selector generation failed: {e}")
            return []
    
    def _element_matches_step(self, element: Dict, step: Dict) -> bool:
        """Check if element matches the planned step."""
        action = step.get("action", "").lower()
        description = step.get("description", "").lower()
        element_text = element.get("textContent", "").lower()
        
        # Simple matching logic
        if action == "click":
            return any(word in element_text for word in ["button", "click", "submit", "login", "search"])
        elif action == "type":
            return element.get("tagName") in ["input", "textarea"]
        elif action == "select":
            return element.get("tagName") == "select"
        else:
            return True
    
    async def _generate_element_selectors(self, page: Page, element: Dict, action: str) -> List[Dict]:
        """Generate multiple selectors for an element."""
        selectors = []
        
        # ID selector
        if element.get("id"):
            selectors.append({
                "type": "id",
                "value": f"#{element['id']}",
                "confidence": 0.95,
                "priority": 1
            })
        
        # Class selector
        if element.get("className"):
            classes = element["className"].split()
            for class_name in classes:
                if class_name.strip():
                    selectors.append({
                        "type": "class",
                        "value": f".{class_name}",
                        "confidence": 0.8,
                        "priority": 2
                    })
        
        # XPath selector
        xpath = await self._generate_xpath_selector(page, element)
        if xpath:
            selectors.append({
                "type": "xpath",
                "value": xpath,
                "confidence": 0.9,
                "priority": 3
            })
        
        # Text-based selector
        if element.get("textContent"):
            text = element["textContent"].strip()
            if text and len(text) < 50:
                selectors.append({
                    "type": "text",
                    "value": f"//{element['tagName']}[text()='{text}']",
                    "confidence": 0.7,
                    "priority": 4
                })
        
        # Aria selector
        if element.get("aria", {}).get("label"):
            selectors.append({
                "type": "aria",
                "value": f"[aria-label='{element['aria']['label']}']",
                "confidence": 0.85,
                "priority": 2
            })
        
        return selectors
    
    async def _generate_xpath_selector(self, page: Page, element: Dict) -> Optional[str]:
        """Generate XPath selector for an element."""
        try:
            # This would use Playwright's built-in XPath generation
            # For now, return a simple XPath
            tag_name = element.get("tagName", "div")
            text_content = element.get("textContent", "")
            
            if text_content:
                return f"//{tag_name}[contains(text(), '{text_content[:20]}')]"
            else:
                return f"//{tag_name}"
                
        except Exception as e:
            self.logger.error(f"XPath generation failed: {e}")
            return None
    
    async def _validate_and_optimize_selectors(self, page: Page, selectors: List[Dict]) -> List[Dict]:
        """Validate and optimize selectors."""
        try:
            validated_selectors = []
            
            for selector in selectors:
                # Test selector on page
                is_valid = await self._test_selector(page, selector["value"])
                
                if is_valid:
                    # Apply selector drift detection
                    optimized = await self.selector_drift_detector.optimize_selector(selector)
                    validated_selectors.append(optimized)
            
            # Sort by priority and confidence
            validated_selectors.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
            
            return validated_selectors
            
        except Exception as e:
            self.logger.error(f"Selector validation failed: {e}")
            return selectors
    
    async def _test_selector(self, page: Page, selector: str) -> bool:
        """Test if a selector works on the page."""
        try:
            element = await page.query_selector(selector)
            return element is not None
        except Exception:
            return False
    
    def _create_fallback_result(self, url: str) -> DOMAnalysisResult:
        """Create fallback result when analysis fails."""
        return DOMAnalysisResult(
            url=url,
            elements=[],
            selectors=[],
            page_structure={"error": "Analysis failed"},
            analysis_metadata={
                "error": "DOM analysis failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )