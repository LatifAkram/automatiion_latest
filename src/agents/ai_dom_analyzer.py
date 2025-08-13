"""
AI-Powered DOM Analyzer
=======================

Advanced DOM analysis using AI for intelligent element detection,
learning from past interactions, and auto-healing capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

from ..core.ai_provider import AIProvider
from ..utils.selector_drift import SelectorDriftDetector
from ..core.vector_store import VectorStore


class AIDOMAnalyzer:
    """AI-powered DOM analyzer with learning and auto-heal capabilities."""
    
    def __init__(self, config, ai_provider: AIProvider, vector_store: VectorStore, selector_drift_detector: SelectorDriftDetector):
        self.config = config
        self.ai_provider = ai_provider
        self.vector_store = vector_store
        self.selector_drift_detector = selector_drift_detector
        self.logger = logging.getLogger(__name__)
        
        # Learning cache
        self.element_patterns = {}
        self.successful_selectors = {}
        self.failed_selectors = {}
        
    async def analyze_dom_with_ai(self, page, instructions: str) -> Dict[str, Any]:
        """Analyze DOM using AI for intelligent element detection."""
        try:
            self.logger.info("Starting AI-powered DOM analysis")
            
            # Get page content and structure
            page_content = await self._extract_page_content(page)
            
            # AI-powered element analysis
            ai_analysis = await self._ai_analyze_elements(page_content, instructions)
            
            # Learn from previous successful patterns
            learned_patterns = await self._apply_learned_patterns(instructions, page_content)
            
            # Generate intelligent selectors
            intelligent_selectors = await self._generate_ai_selectors(ai_analysis, learned_patterns, instructions)
            
            # Auto-heal and validate selectors
            validated_selectors = await self._auto_heal_selectors(intelligent_selectors, page)
            
            return {
                "ai_analysis": ai_analysis,
                "learned_patterns": learned_patterns,
                "intelligent_selectors": validated_selectors,
                "page_structure": page_content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"AI DOM analysis failed: {e}", exc_info=True)
            return {}

    async def _extract_page_content(self, page) -> Dict[str, Any]:
        """Extract comprehensive page content for AI analysis."""
        try:
            # Get page HTML
            html_content = await page.content()
            
            # Extract all elements with their attributes
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
                        
                        elements.push({
                            tagName: element.tagName.toLowerCase(),
                            id: element.id,
                            className: element.className,
                            textContent: element.textContent?.trim().substring(0, 100),
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
                            computedStyle: window.getComputedStyle(element)
                        });
                    }
                    return elements;
                }
            """)
            
            # Get page metadata
            page_metadata = await page.evaluate("""
                () => ({
                    title: document.title,
                    url: window.location.href,
                    domain: window.location.hostname,
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    }
                })
            """)
            
            return {
                "html": html_content,
                "elements": elements_data,
                "metadata": page_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Page content extraction failed: {e}")
            return {}

    async def _ai_analyze_elements(self, page_content: Dict[str, Any], instructions: str) -> Dict[str, Any]:
        """Use AI to analyze elements and understand their purpose."""
        try:
            # Prepare context for AI analysis
            context = {
                "instructions": instructions,
                "page_title": page_content.get("metadata", {}).get("title", ""),
                "domain": page_content.get("metadata", {}).get("domain", ""),
                "elements_count": len(page_content.get("elements", [])),
                "visible_elements": [e for e in page_content.get("elements", []) if e.get("isVisible", False)]
            }
            
            # AI prompt for element analysis
            prompt = f"""
            Analyze this web page for automation purposes:
            
            Instructions: {instructions}
            Page Title: {context['page_title']}
            Domain: {context['domain']}
            Total Elements: {context['elements_count']}
            Visible Elements: {len(context['visible_elements'])}
            
            Please analyze the page and identify:
            1. Login/authentication elements (username, password, login button)
            2. Form input fields (text, email, phone, search)
            3. Navigation elements (menus, links, buttons)
            4. Content areas (headers, text, images)
            5. Interactive elements (buttons, links, dropdowns)
            
            For each identified element, provide:
            - Element purpose and function
            - Best selector strategy (ID, class, XPath, text-based)
            - Confidence score (0-1)
            - Alternative selectors as fallbacks
            
            Return as JSON with structure:
            {{
                "login_elements": [...],
                "form_elements": [...],
                "navigation_elements": [...],
                "content_elements": [...],
                "interactive_elements": [...],
                "recommended_actions": [...]
            }}
            """
            
            # Get AI analysis
            ai_response = await self.ai_provider.generate_response(prompt)
            
            try:
                # Clean the response to extract JSON
                cleaned_response = self._extract_json_from_response(ai_response)
                ai_analysis = json.loads(cleaned_response)
                self.logger.info(f"AI analysis completed: {len(ai_analysis)} categories")
                return ai_analysis
            except json.JSONDecodeError as e:
                self.logger.warning(f"AI response not valid JSON, using fallback analysis: {e}")
                self.logger.warning(f"Raw response: {ai_response[:200]}...")
                return self._fallback_element_analysis(page_content, instructions)
                
        except Exception as e:
            self.logger.error(f"AI element analysis failed: {e}")
            return self._fallback_element_analysis(page_content, instructions)

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from AI response that might contain extra text."""
        try:
            import re
            
            # First, try to extract from code blocks with json language
            code_block_pattern = r'```json\s*(\{.*?\})\s*```'
            code_match = re.search(code_block_pattern, response, re.DOTALL)
            
            if code_match:
                return code_match.group(1)
            
            # Try to extract from code blocks without language specification
            code_block_pattern2 = r'```\s*(\{.*?\})\s*```'
            code_match2 = re.search(code_block_pattern2, response, re.DOTALL)
            
            if code_match2:
                return code_match2.group(1)
            
            # Try to find JSON object with curly braces (from start to end)
            json_pattern = r'^\s*\{.*\}\s*$'
            if re.match(json_pattern, response, re.DOTALL):
                return response.strip()
            
            # Try to find JSON object anywhere in the response
            json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_match = re.search(json_pattern2, response, re.DOTALL)
            
            if json_match:
                return json_match.group(0)
            
            # If still no JSON found, return the original response
            return response
            
        except Exception as e:
            self.logger.warning(f"Failed to extract JSON from response: {e}")
            return response

    async def _apply_learned_patterns(self, instructions: str, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned patterns from previous successful automations."""
        try:
            # Search vector store for similar patterns
            similar_patterns = await self.vector_store.search(
                query=f"automation pattern: {instructions}",
                limit=5
            )
            
            learned_patterns = {
                "successful_selectors": [],
                "element_patterns": [],
                "domain_specific": [],
                "common_failures": []
            }
            
            for pattern in similar_patterns:
                pattern_data = pattern.get("metadata", {})
                
                # Extract successful selectors
                if pattern_data.get("success_rate", 0) > 0.8:
                    learned_patterns["successful_selectors"].extend(
                        pattern_data.get("selectors", [])
                    )
                
                # Extract element patterns
                if pattern_data.get("element_patterns"):
                    learned_patterns["element_patterns"].extend(
                        pattern_data.get("element_patterns", [])
                    )
                
                # Domain-specific patterns
                if pattern_data.get("domain") == page_content.get("metadata", {}).get("domain"):
                    learned_patterns["domain_specific"].append(pattern_data)
                
                # Common failures to avoid
                if pattern_data.get("success_rate", 0) < 0.3:
                    learned_patterns["common_failures"].extend(
                        pattern_data.get("failed_selectors", [])
                    )
            
            self.logger.info(f"Applied {len(learned_patterns['successful_selectors'])} learned patterns")
            return learned_patterns
            
        except Exception as e:
            self.logger.error(f"Learned patterns application failed: {e}")
            return {}

    async def _generate_ai_selectors(self, ai_analysis: Dict[str, Any], learned_patterns: Dict[str, Any], instructions: str) -> Dict[str, List[str]]:
        """Generate intelligent selectors using AI and learned patterns."""
        try:
            intelligent_selectors = {
                "input_selectors": [],
                "button_selectors": [],
                "navigation_selectors": [],
                "content_selectors": [],
                "fallback_selectors": []
            }
            
            # Combine AI analysis with learned patterns
            all_elements = []
            
            # Add AI-identified elements
            for category, elements in ai_analysis.items():
                if isinstance(elements, list):
                    all_elements.extend(elements)
            
            # Add learned patterns
            for pattern in learned_patterns.get("successful_selectors", []):
                # Ensure pattern is a dictionary
                if isinstance(pattern, dict):
                    all_elements.append({
                        "type": "learned",
                        "selector": pattern.get("selector"),
                        "confidence": pattern.get("success_rate", 0.5),
                        "context": pattern.get("context", "")
                    })
                elif isinstance(pattern, str):
                    # If pattern is a string, treat it as a selector
                    all_elements.append({
                        "type": "learned",
                        "selector": pattern,
                        "confidence": 0.5,
                        "context": "string_pattern"
                    })
                else:
                    self.logger.warning(f"Unknown pattern type: {type(pattern)}")
            
            # Generate selectors for each element
            for element in all_elements:
                selectors = await self._generate_element_selectors(element, instructions)
                
                # Categorize selectors
                element_type = element.get("type", "unknown")
                if "input" in element_type or "form" in element_type:
                    intelligent_selectors["input_selectors"].extend(selectors)
                elif "button" in element_type or "click" in element_type:
                    intelligent_selectors["button_selectors"].extend(selectors)
                elif "nav" in element_type or "link" in element_type:
                    intelligent_selectors["navigation_selectors"].extend(selectors)
                else:
                    intelligent_selectors["content_selectors"].extend(selectors)
            
            # Add fallback selectors
            intelligent_selectors["fallback_selectors"] = [
                "body",
                "main",
                "div",
                "span"
            ]
            
            return intelligent_selectors
            
        except Exception as e:
            self.logger.error(f"AI selector generation failed: {e}")
            return {}

    async def _generate_element_selectors(self, element: Dict[str, Any], instructions: str) -> List[str]:
        """Generate multiple selector strategies for a single element."""
        try:
            selectors = []
            
            # Ensure element is a dictionary
            if not isinstance(element, dict):
                self.logger.warning(f"Element is not a dictionary: {type(element)}")
                return []
            
            # Priority 1: ID-based selectors
            if element.get("id"):
                selectors.append(f"#{element['id']}")
            
            # Priority 2: Data attributes
            if element.get("data_testid"):
                selectors.append(f"[data-testid='{element['data_testid']}']")
            if element.get("data_cy"):
                selectors.append(f"[data-cy='{element['data_cy']}']")
            
            # Priority 3: Name-based selectors
            if element.get("name"):
                selectors.append(f"[name='{element['name']}']")
            
            # Priority 4: Class-based selectors (with specificity)
            if element.get("className"):
                classes = element["className"].split()
                if len(classes) == 1:
                    selectors.append(f".{classes[0]}")
                else:
                    # Use multiple classes for specificity
                    selectors.append(f".{'.'.join(classes[:2])}")
            
            # Priority 5: Text-based selectors
            if element.get("textContent"):
                text = element["textContent"].strip()
                if len(text) > 0 and len(text) < 50:
                    selectors.append(f"text='{text}'")
                    selectors.append(f":has-text('{text}')")
            
            # Priority 6: XPath selectors
            if element.get("xpath"):
                selectors.append(element["xpath"])
            
            # Priority 7: Position-based selectors (as last resort)
            if element.get("position"):
                pos = element["position"]
                if pos["width"] > 0 and pos["height"] > 0:
                    selectors.append(f"xpath=//*[@x='{pos['x']}' and @y='{pos['y']}']")
            
            return selectors
            
        except Exception as e:
            self.logger.error(f"Element selector generation failed: {e}")
            return []

    async def _auto_heal_selectors(self, intelligent_selectors: Dict[str, List[str]], page) -> Dict[str, List[str]]:
        """Auto-heal and validate selectors using drift detection."""
        try:
            validated_selectors = {}
            
            for category, selectors in intelligent_selectors.items():
                validated_selectors[category] = []
                
                for selector in selectors:
                    try:
                        # Test selector with short timeout
                        element = await page.wait_for_selector(selector, timeout=1000)
                        if element:
                            validated_selectors[category].append(selector)
                            
                            # Store successful selector for learning
                            await self._store_successful_selector(selector, category)
                        else:
                            # Try auto-healing
                            healed_selector = await self.selector_drift_detector.auto_heal_selector(selector, page)
                            if healed_selector:
                                validated_selectors[category].append(healed_selector)
                                await self._store_healed_selector(selector, healed_selector, category)
                            
                    except Exception as e:
                        # Store failed selector for learning
                        await self._store_failed_selector(selector, category, str(e))
                        continue
            
            self.logger.info(f"Auto-healing completed: {sum(len(v) for v in validated_selectors.values())} valid selectors")
            return validated_selectors
            
        except Exception as e:
            self.logger.error(f"Auto-healing failed: {e}")
            return intelligent_selectors

    async def _store_successful_selector(self, selector: str, category: str):
        """Store successful selector for learning."""
        try:
            pattern_data = {
                "selector": selector,
                "category": category,
                "success_rate": 1.0,
                "last_used": datetime.utcnow().isoformat(),
                "usage_count": 1
            }
            
            await self.vector_store.add_document(
                content=f"Successful selector pattern: {selector} for {category}",
                metadata=pattern_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store successful selector: {e}")

    async def _store_healed_selector(self, original_selector: str, healed_selector: str, category: str):
        """Store healed selector for learning."""
        try:
            pattern_data = {
                "original_selector": original_selector,
                "healed_selector": healed_selector,
                "category": category,
                "healing_success": True,
                "last_used": datetime.utcnow().isoformat()
            }
            
            await self.vector_store.add_document(
                content=f"Selector healing pattern: {original_selector} -> {healed_selector}",
                metadata=pattern_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store healed selector: {e}")

    async def _store_failed_selector(self, selector: str, category: str, error: str):
        """Store failed selector for learning."""
        try:
            pattern_data = {
                "selector": selector,
                "category": category,
                "error": error,
                "success_rate": 0.0,
                "last_used": datetime.utcnow().isoformat()
            }
            
            await self.vector_store.add_document(
                content=f"Failed selector pattern: {selector} for {category} - {error}",
                metadata=pattern_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store failed selector: {e}")

    def _fallback_element_analysis(self, page_content: Dict[str, Any], instructions: str) -> Dict[str, Any]:
        """Fallback element analysis when AI fails."""
        try:
            elements = page_content.get("elements", [])
            
            # Basic categorization
            login_elements = [e for e in elements if self._is_login_element(e)]
            form_elements = [e for e in elements if self._is_form_element(e)]
            navigation_elements = [e for e in elements if self._is_navigation_element(e)]
            interactive_elements = [e for e in elements if self._is_interactive_element(e)]
            
            return {
                "login_elements": login_elements,
                "form_elements": form_elements,
                "navigation_elements": navigation_elements,
                "interactive_elements": interactive_elements,
                "content_elements": elements[:10]  # First 10 elements as content
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis failed: {e}")
            return {}

    def _is_login_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is a login-related element."""
        text = element.get("textContent", "").lower()
        id_attr = element.get("id", "").lower()
        class_name = element.get("className", "").lower()
        
        login_keywords = ["login", "signin", "username", "password", "email", "mobile", "phone"]
        return any(keyword in text or keyword in id_attr or keyword in class_name for keyword in login_keywords)

    def _is_form_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is a form element."""
        tag_name = element.get("tagName", "").lower()
        return tag_name in ["input", "textarea", "select", "form"]

    def _is_navigation_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is a navigation element."""
        tag_name = element.get("tagName", "").lower()
        text = element.get("textContent", "").lower()
        
        nav_keywords = ["menu", "nav", "home", "about", "contact", "search"]
        return tag_name == "a" or any(keyword in text for keyword in nav_keywords)

    def _is_interactive_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is an interactive element."""
        tag_name = element.get("tagName", "").lower()
        return tag_name in ["button", "a", "input"] and element.get("isVisible", False)