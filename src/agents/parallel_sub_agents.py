"""
Parallel Sub-Agents System
=========================

Multiple specialized agents working simultaneously to complete
automation tasks faster and more efficiently.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from ..core.ai_provider import AIProvider
from ..core.vector_store import VectorStore


class URLExtractionAgent:
    """Agent responsible for extracting URLs from instructions and AI responses."""
    
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
        
    async def extract_urls_from_instruction(self, instruction: str) -> List[str]:
        """Extract URLs from user instruction."""
        try:
            # Use regex to find URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, instruction)
            
            # Also look for common website patterns
            website_patterns = [
                r'www\.\w+\.\w+',
                r'\w+\.com',
                r'\w+\.org',
                r'\w+\.net',
                r'\w+\.io'
            ]
            
            for pattern in website_patterns:
                matches = re.findall(pattern, instruction)
                for match in matches:
                    if not match.startswith('http'):
                        urls.append(f'https://{match}')
            
            # Remove duplicates and validate
            unique_urls = list(set(urls))
            valid_urls = []
            
            for url in unique_urls:
                if self._is_valid_url(url):
                    valid_urls.append(url)
            
            self.logger.info(f"Extracted {len(valid_urls)} URLs from instruction")
            return valid_urls
            
        except Exception as e:
            self.logger.error(f"URL extraction failed: {e}")
            return []
    
    async def extract_urls_from_ai_response(self, ai_response: str) -> List[str]:
        """Extract URLs from AI response."""
        try:
            # Use AI to extract URLs from response
            prompt = f"""
            Extract all URLs from this AI response. Return only the URLs as a JSON array.
            
            AI Response: {ai_response}
            
            Return format: ["url1", "url2", "url3"]
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                import json
                urls = json.loads(response)
                if isinstance(urls, list):
                    return [url for url in urls if self._is_valid_url(url)]
            except json.JSONDecodeError:
                # Fallback to regex extraction
                return await self.extract_urls_from_instruction(ai_response)
                
        except Exception as e:
            self.logger.error(f"AI URL extraction failed: {e}")
            return await self.extract_urls_from_instruction(ai_response)
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is properly formatted."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class ParallelDOMAnalysisAgent:
    """Agent for parallel DOM analysis using multiple strategies."""
    
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
        
    async def analyze_dom_parallel(self, page, instructions: str) -> Dict[str, Any]:
        """Analyze DOM using multiple parallel strategies."""
        try:
            self.logger.info("Starting parallel DOM analysis")
            
            # Create parallel analysis tasks
            tasks = [
                self._analyze_forms_parallel(page),
                self._analyze_interactive_elements_parallel(page),
                self._analyze_navigation_parallel(page),
                self._analyze_content_parallel(page),
                self._analyze_structure_parallel(page),
                self._ai_analyze_context_parallel(page, instructions)
            ]
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            dom_analysis = {
                "forms": results[0] if not isinstance(results[0], Exception) else [],
                "interactive": results[1] if not isinstance(results[1], Exception) else [],
                "navigation": results[2] if not isinstance(results[2], Exception) else [],
                "content": results[3] if not isinstance(results[3], Exception) else [],
                "structure": results[4] if not isinstance(results[4], Exception) else [],
                "ai_context": results[5] if not isinstance(results[5], Exception) else {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Parallel DOM analysis completed: {len(dom_analysis)} categories")
            return dom_analysis
            
        except Exception as e:
            self.logger.error(f"Parallel DOM analysis failed: {e}")
            return {}
    
    async def _analyze_forms_parallel(self, page) -> List[Dict[str, Any]]:
        """Analyze form elements in parallel."""
        try:
            # Multiple parallel form analysis strategies
            tasks = [
                self._get_input_elements(page),
                self._get_form_elements(page),
                self._get_submit_elements(page)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            forms = []
            for result in results:
                if not isinstance(result, Exception):
                    forms.extend(result)
            
            return forms
            
        except Exception as e:
            self.logger.error(f"Form analysis failed: {e}")
            return []
    
    async def _get_input_elements(self, page) -> List[Dict[str, Any]]:
        """Get input elements."""
        try:
            elements = await page.query_selector_all("input, textarea, select")
            form_elements = []
            
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
                        "category": "input"
                    })
                except Exception:
                    continue
            
            return form_elements
            
        except Exception as e:
            self.logger.error(f"Input elements analysis failed: {e}")
            return []
    
    async def _get_form_elements(self, page) -> List[Dict[str, Any]]:
        """Get form elements."""
        try:
            elements = await page.query_selector_all("form")
            forms = []
            
            for element in elements:
                try:
                    action = await element.get_attribute('action')
                    method = await element.get_attribute('method')
                    id_attr = await element.get_attribute('id')
                    
                    forms.append({
                        "tag": "FORM",
                        "action": action,
                        "method": method,
                        "id": id_attr,
                        "category": "form"
                    })
                except Exception:
                    continue
            
            return forms
            
        except Exception as e:
            self.logger.error(f"Form elements analysis failed: {e}")
            return []
    
    async def _get_submit_elements(self, page) -> List[Dict[str, Any]]:
        """Get submit elements."""
        try:
            elements = await page.query_selector_all("button[type='submit'], input[type='submit']")
            submits = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    value = await element.get_attribute('value')
                    id_attr = await element.get_attribute('id')
                    
                    submits.append({
                        "tag": "SUBMIT",
                        "text": text_content,
                        "value": value,
                        "id": id_attr,
                        "category": "submit"
                    })
                except Exception:
                    continue
            
            return submits
            
        except Exception as e:
            self.logger.error(f"Submit elements analysis failed: {e}")
            return []
    
    async def _analyze_interactive_elements_parallel(self, page) -> List[Dict[str, Any]]:
        """Analyze interactive elements in parallel."""
        try:
            tasks = [
                self._get_buttons(page),
                self._get_links(page),
                self._get_clickable_elements(page)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            interactive = []
            for result in results:
                if not isinstance(result, Exception):
                    interactive.extend(result)
            
            return interactive
            
        except Exception as e:
            self.logger.error(f"Interactive analysis failed: {e}")
            return []
    
    async def _get_buttons(self, page) -> List[Dict[str, Any]]:
        """Get button elements."""
        try:
            elements = await page.query_selector_all("button")
            buttons = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    onclick = await element.get_attribute('onclick')
                    id_attr = await element.get_attribute('id')
                    
                    buttons.append({
                        "tag": "BUTTON",
                        "text": text_content,
                        "onclick": onclick,
                        "id": id_attr,
                        "category": "button"
                    })
                except Exception:
                    continue
            
            return buttons
            
        except Exception as e:
            self.logger.error(f"Button analysis failed: {e}")
            return []
    
    async def _get_links(self, page) -> List[Dict[str, Any]]:
        """Get link elements."""
        try:
            elements = await page.query_selector_all("a[href]")
            links = []
            
            for element in elements:
                try:
                    href = await element.get_attribute('href')
                    text_content = await element.text_content()
                    id_attr = await element.get_attribute('id')
                    
                    links.append({
                        "tag": "A",
                        "href": href,
                        "text": text_content,
                        "id": id_attr,
                        "category": "link"
                    })
                except Exception:
                    continue
            
            return links
            
        except Exception as e:
            self.logger.error(f"Link analysis failed: {e}")
            return []
    
    async def _get_clickable_elements(self, page) -> List[Dict[str, Any]]:
        """Get clickable elements."""
        try:
            elements = await page.query_selector_all("[onclick], [role='button']")
            clickable = []
            
            for element in elements:
                try:
                    onclick = await element.get_attribute('onclick')
                    role = await element.get_attribute('role')
                    text_content = await element.text_content()
                    
                    clickable.append({
                        "tag": "CLICKABLE",
                        "onclick": onclick,
                        "role": role,
                        "text": text_content,
                        "category": "clickable"
                    })
                except Exception:
                    continue
            
            return clickable
            
        except Exception as e:
            self.logger.error(f"Clickable analysis failed: {e}")
            return []
    
    async def _analyze_navigation_parallel(self, page) -> List[Dict[str, Any]]:
        """Analyze navigation elements in parallel."""
        try:
            tasks = [
                self._get_nav_elements(page),
                self._get_menu_elements(page),
                self._get_breadcrumb_elements(page)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            navigation = []
            for result in results:
                if not isinstance(result, Exception):
                    navigation.extend(result)
            
            return navigation
            
        except Exception as e:
            self.logger.error(f"Navigation analysis failed: {e}")
            return []
    
    async def _get_nav_elements(self, page) -> List[Dict[str, Any]]:
        """Get navigation elements."""
        try:
            elements = await page.query_selector_all("nav, [role='navigation']")
            navs = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    id_attr = await element.get_attribute('id')
                    
                    navs.append({
                        "tag": "NAV",
                        "text": text_content,
                        "id": id_attr,
                        "category": "navigation"
                    })
                except Exception:
                    continue
            
            return navs
            
        except Exception as e:
            self.logger.error(f"Nav analysis failed: {e}")
            return []
    
    async def _get_menu_elements(self, page) -> List[Dict[str, Any]]:
        """Get menu elements."""
        try:
            elements = await page.query_selector_all(".menu, .nav, [class*='menu'], [class*='nav']")
            menus = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    class_name = await element.get_attribute('class')
                    
                    menus.append({
                        "tag": "MENU",
                        "text": text_content,
                        "class": class_name,
                        "category": "menu"
                    })
                except Exception:
                    continue
            
            return menus
            
        except Exception as e:
            self.logger.error(f"Menu analysis failed: {e}")
            return []
    
    async def _get_breadcrumb_elements(self, page) -> List[Dict[str, Any]]:
        """Get breadcrumb elements."""
        try:
            elements = await page.query_selector_all("[class*='breadcrumb'], [role='navigation']")
            breadcrumbs = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    class_name = await element.get_attribute('class')
                    
                    breadcrumbs.append({
                        "tag": "BREADCRUMB",
                        "text": text_content,
                        "class": class_name,
                        "category": "breadcrumb"
                    })
                except Exception:
                    continue
            
            return breadcrumbs
            
        except Exception as e:
            self.logger.error(f"Breadcrumb analysis failed: {e}")
            return []
    
    async def _analyze_content_parallel(self, page) -> List[Dict[str, Any]]:
        """Analyze content elements in parallel."""
        try:
            tasks = [
                self._get_headers(page),
                self._get_paragraphs(page),
                self._get_images(page)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            content = []
            for result in results:
                if not isinstance(result, Exception):
                    content.extend(result)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return []
    
    async def _get_headers(self, page) -> List[Dict[str, Any]]:
        """Get header elements."""
        try:
            elements = await page.query_selector_all("h1, h2, h3, h4, h5, h6")
            headers = []
            
            for element in elements:
                try:
                    tag_name = await element.get_attribute('tagName')
                    text_content = await element.text_content()
                    
                    headers.append({
                        "tag": tag_name,
                        "text": text_content,
                        "category": "header"
                    })
                except Exception:
                    continue
            
            return headers
            
        except Exception as e:
            self.logger.error(f"Header analysis failed: {e}")
            return []
    
    async def _get_paragraphs(self, page) -> List[Dict[str, Any]]:
        """Get paragraph elements."""
        try:
            elements = await page.query_selector_all("p")
            paragraphs = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    class_name = await element.get_attribute('class')
                    
                    paragraphs.append({
                        "tag": "P",
                        "text": text_content,
                        "class": class_name,
                        "category": "paragraph"
                    })
                except Exception:
                    continue
            
            return paragraphs
            
        except Exception as e:
            self.logger.error(f"Paragraph analysis failed: {e}")
            return []
    
    async def _get_images(self, page) -> List[Dict[str, Any]]:
        """Get image elements."""
        try:
            elements = await page.query_selector_all("img")
            images = []
            
            for element in elements:
                try:
                    src = await element.get_attribute('src')
                    alt = await element.get_attribute('alt')
                    
                    images.append({
                        "tag": "IMG",
                        "src": src,
                        "alt": alt,
                        "category": "image"
                    })
                except Exception:
                    continue
            
            return images
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return []
    
    async def _analyze_structure_parallel(self, page) -> List[Dict[str, Any]]:
        """Analyze structural elements in parallel."""
        try:
            tasks = [
                self._get_main_elements(page),
                self._get_section_elements(page),
                self._get_aside_elements(page)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            structure = []
            for result in results:
                if not isinstance(result, Exception):
                    structure.extend(result)
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}")
            return []
    
    async def _get_main_elements(self, page) -> List[Dict[str, Any]]:
        """Get main elements."""
        try:
            elements = await page.query_selector_all("main")
            mains = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    id_attr = await element.get_attribute('id')
                    
                    mains.append({
                        "tag": "MAIN",
                        "text": text_content,
                        "id": id_attr,
                        "category": "main"
                    })
                except Exception:
                    continue
            
            return mains
            
        except Exception as e:
            self.logger.error(f"Main analysis failed: {e}")
            return []
    
    async def _get_section_elements(self, page) -> List[Dict[str, Any]]:
        """Get section elements."""
        try:
            elements = await page.query_selector_all("section")
            sections = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    id_attr = await element.get_attribute('id')
                    
                    sections.append({
                        "tag": "SECTION",
                        "text": text_content,
                        "id": id_attr,
                        "category": "section"
                    })
                except Exception:
                    continue
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Section analysis failed: {e}")
            return []
    
    async def _get_aside_elements(self, page) -> List[Dict[str, Any]]:
        """Get aside elements."""
        try:
            elements = await page.query_selector_all("aside")
            asides = []
            
            for element in elements:
                try:
                    text_content = await element.text_content()
                    id_attr = await element.get_attribute('id')
                    
                    asides.append({
                        "tag": "ASIDE",
                        "text": text_content,
                        "id": id_attr,
                        "category": "aside"
                    })
                except Exception:
                    continue
            
            return asides
            
        except Exception as e:
            self.logger.error(f"Aside analysis failed: {e}")
            return []
    
    async def _ai_analyze_context_parallel(self, page, instructions: str) -> Dict[str, Any]:
        """AI-powered context analysis in parallel."""
        try:
            # Get page metadata
            metadata = await page.evaluate("""
                () => ({
                    title: document.title,
                    url: window.location.href,
                    domain: window.location.hostname
                })
            """)
            
            # AI analysis prompt
            prompt = f"""
            Analyze this web page context for automation:
            
            Instructions: {instructions}
            Page Title: {metadata.get('title', '')}
            Domain: {metadata.get('domain', '')}
            URL: {metadata.get('url', '')}
            
            Provide analysis in JSON format:
            {{
                "page_type": "type of page (login, search, ecommerce, etc.)",
                "primary_actions": ["list of main actions user can perform"],
                "key_elements": ["list of important elements for automation"],
                "complexity": "simple/medium/complex",
                "automation_priority": ["list of elements to automate first"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "page_type": "unknown",
                    "primary_actions": [],
                    "key_elements": [],
                    "complexity": "medium",
                    "automation_priority": []
                }
                
        except Exception as e:
            self.logger.error(f"AI context analysis failed: {e}")
            return {}


class CodeGenerationAgent:
    """Agent for generating automation code in parallel."""
    
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
        
    async def generate_code_parallel(self, automation_plan: Dict[str, Any], dom_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate code in multiple formats simultaneously."""
        try:
            self.logger.info("Starting parallel code generation")
            
            # Generate code in multiple formats simultaneously
            tasks = [
                self._generate_playwright_code(automation_plan, dom_analysis),
                self._generate_selenium_code(automation_plan, dom_analysis),
                self._generate_cypress_code(automation_plan, dom_analysis)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            code_results = {
                "playwright": results[0] if not isinstance(results[0], Exception) else "",
                "selenium": results[1] if not isinstance(results[1], Exception) else "",
                "cypress": results[2] if not isinstance(results[2], Exception) else ""
            }
            
            self.logger.info("Parallel code generation completed")
            return code_results
            
        except Exception as e:
            self.logger.error(f"Parallel code generation failed: {e}")
            return {}
    
    async def _generate_playwright_code(self, automation_plan: Dict[str, Any], dom_analysis: Dict[str, Any]) -> str:
        """Generate Playwright code."""
        try:
            steps = automation_plan.get("steps", [])
            
            prompt = f"""
            Generate Playwright code for this automation plan:
            
            Steps: {steps}
            DOM Analysis: {dom_analysis}
            
            Return only the Playwright code without explanations:
            
            ```python
            from playwright.async_api import async_playwright
            import asyncio
            
            async def main():
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    
                    # Automation steps here
                    
                    await browser.close()
            
            asyncio.run(main())
            ```
            """
            
            response = await self.ai_provider.generate_response(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Playwright code generation failed: {e}")
            return ""
    
    async def _generate_selenium_code(self, automation_plan: Dict[str, Any], dom_analysis: Dict[str, Any]) -> str:
        """Generate Selenium code."""
        try:
            steps = automation_plan.get("steps", [])
            
            prompt = f"""
            Generate Selenium code for this automation plan:
            
            Steps: {steps}
            DOM Analysis: {dom_analysis}
            
            Return only the Selenium code without explanations:
            
            ```python
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            driver = webdriver.Chrome()
            
            # Automation steps here
            
            driver.quit()
            ```
            """
            
            response = await self.ai_provider.generate_response(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Selenium code generation failed: {e}")
            return ""
    
    async def _generate_cypress_code(self, automation_plan: Dict[str, Any], dom_analysis: Dict[str, Any]) -> str:
        """Generate Cypress code."""
        try:
            steps = automation_plan.get("steps", [])
            
            prompt = f"""
            Generate Cypress code for this automation plan:
            
            Steps: {steps}
            DOM Analysis: {dom_analysis}
            
            Return only the Cypress code without explanations:
            
            ```javascript
            describe('Automation Test', () => {{
                it('should perform automation steps', () => {{
                    // Automation steps here
                }})
            }})
            ```
            """
            
            response = await self.ai_provider.generate_response(prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Cypress code generation failed: {e}")
            return ""


class ParallelSubAgentOrchestrator:
    """Orchestrator for managing parallel sub-agents."""
    
    def __init__(self, ai_provider: AIProvider, vector_store: VectorStore):
        self.ai_provider = ai_provider
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-agents
        self.url_agent = URLExtractionAgent(ai_provider)
        self.dom_agent = ParallelDOMAnalysisAgent(ai_provider)
        self.code_agent = CodeGenerationAgent(ai_provider)
        
    async def execute_parallel_automation(self, instructions: str, url: str, page) -> Dict[str, Any]:
        """Execute automation using parallel sub-agents."""
        try:
            self.logger.info("Starting parallel automation execution")
            
            # Step 1: Extract URLs in parallel with initial analysis
            url_extraction_task = self.url_agent.extract_urls_from_instruction(instructions)
            
            # Step 2: Start DOM analysis immediately if URL is available
            dom_analysis_task = None
            if url:
                dom_analysis_task = self.dom_agent.analyze_dom_parallel(page, instructions)
            
            # Wait for URL extraction
            extracted_urls = await url_extraction_task
            
            # Use extracted URL if no URL provided
            target_url = url if url else (extracted_urls[0] if extracted_urls else None)
            
            # Step 3: Generate automation plan using AI
            plan_generation_task = self._generate_automation_plan_parallel(instructions, target_url)
            
            # Step 4: Wait for DOM analysis and plan generation
            results = await asyncio.gather(
                dom_analysis_task if dom_analysis_task else asyncio.sleep(0),
                plan_generation_task,
                return_exceptions=True
            )
            
            dom_analysis = results[0] if not isinstance(results[0], Exception) else {}
            automation_plan = results[1] if not isinstance(results[1], Exception) else {}
            
            # Step 5: Generate code in parallel with execution
            code_generation_task = self.code_agent.generate_code_parallel(automation_plan, dom_analysis)
            
            # Step 6: Execute automation plan
            execution_task = self._execute_automation_plan_parallel(automation_plan, page)
            
            # Wait for both code generation and execution
            final_results = await asyncio.gather(
                code_generation_task,
                execution_task,
                return_exceptions=True
            )
            
            generated_code = final_results[0] if not isinstance(final_results[0], Exception) else {}
            execution_result = final_results[1] if not isinstance(final_results[1], Exception) else {}
            
            return {
                "urls_extracted": extracted_urls,
                "target_url": target_url,
                "dom_analysis": dom_analysis,
                "automation_plan": automation_plan,
                "generated_code": generated_code,
                "execution_result": execution_result,
                "parallel_execution": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Parallel automation execution failed: {e}")
            return {
                "error": str(e),
                "parallel_execution": False
            }
    
    async def _generate_automation_plan_parallel(self, instructions: str, url: str) -> Dict[str, Any]:
        """Generate automation plan using AI."""
        try:
            prompt = f"""
            Generate an automation plan for these instructions:
            
            Instructions: {instructions}
            URL: {url}
            
            Return a JSON plan with this structure:
            {{
                "steps": [
                    {{
                        "step": 1,
                        "action": "navigate",
                        "description": "Navigate to the website",
                        "url": "{url}",
                        "expected_result": "Page loaded successfully"
                    }}
                ],
                "estimated_duration": "estimated time in seconds",
                "complexity": "simple/medium/complex",
                "success_probability": 0.85
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "steps": [],
                    "estimated_duration": 30,
                    "complexity": "medium",
                    "success_probability": 0.5
                }
                
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return {"steps": [], "error": str(e)}
    
    async def _execute_automation_plan_parallel(self, automation_plan: Dict[str, Any], page) -> Dict[str, Any]:
        """Execute automation plan."""
        try:
            steps = automation_plan.get("steps", [])
            completed_steps = []
            errors = []
            
            for i, step in enumerate(steps):
                try:
                    action = step.get("action", "")
                    
                    if action == "navigate":
                        url = step.get("url", "")
                        if url:
                            await page.goto(url, wait_until="networkidle")
                            completed_steps.append(step)
                    
                    elif action == "click":
                        selector = step.get("selector", "")
                        if selector:
                            element = await page.wait_for_selector(selector, timeout=5000)
                            if element:
                                await element.click()
                                completed_steps.append(step)
                            else:
                                errors.append(f"Element not found: {selector}")
                    
                    elif action == "type":
                        selector = step.get("selector", "")
                        text = step.get("text", "")
                        if selector and text:
                            element = await page.wait_for_selector(selector, timeout=5000)
                            if element:
                                await element.fill(text)
                                completed_steps.append(step)
                            else:
                                errors.append(f"Element not found: {selector}")
                    
                    elif action == "wait":
                        duration = step.get("duration", 2)
                        await asyncio.sleep(duration)
                        completed_steps.append(step)
                    
                except Exception as e:
                    errors.append(f"Step {i+1} failed: {str(e)}")
            
            return {
                "success": len(errors) == 0,
                "completed_steps": len(completed_steps),
                "total_steps": len(steps),
                "errors": errors,
                "execution_time": len(steps) * 2  # Rough estimate
            }
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "completed_steps": 0,
                "total_steps": 0
            }