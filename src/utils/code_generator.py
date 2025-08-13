"""
Code Generator Utility
======================

Generates automation code in multiple formats (Playwright, Selenium, Cypress).
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.ai_provider import AIProvider


class CodeGenerator:
    """Generates automation code from execution results."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
        
    async def generate_automation_code(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automation code in multiple formats."""
        try:
            self.logger.info("Generating automation code")
            
            # Generate Playwright code
            playwright_code = await self._generate_playwright_code(execution_result)
            
            # Generate Selenium code
            selenium_code = await self._generate_selenium_code(execution_result)
            
            # Generate Cypress code
            cypress_code = await self._generate_cypress_code(execution_result)
            
            return {
                "playwright": playwright_code,
                "selenium": selenium_code,
                "cypress": cypress_code,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {"error": str(e)}
            
    async def _generate_playwright_code(self, execution_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate Playwright automation code."""
        try:
            # Extract automation details
            automation_details = execution_result.get("result", {}).get("automation_details", {})
            actions = automation_details.get("actions", [])
            url = execution_result.get("result", {}).get("url", "")
            
            prompt = f"""
            Generate Playwright automation code for this execution:
            
            URL: {url}
            Actions: {actions}
            Execution Details: {automation_details}
            
            Generate:
            1. Complete Playwright test file with proper imports
            2. Page object model structure
            3. Error handling and assertions
            4. Screenshot capture
            5. Video recording setup
            
            Return as structured code with comments.
            """
            
            code = await self.ai_provider.generate_response(prompt)
            
            return {
                "code": code,
                "framework": "playwright",
                "language": "typescript",
                "features": ["screenshots", "video", "error_handling", "assertions"]
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _generate_selenium_code(self, execution_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate Selenium automation code."""
        try:
            # Extract automation details
            automation_details = execution_result.get("result", {}).get("automation_details", {})
            actions = automation_details.get("actions", [])
            url = execution_result.get("result", {}).get("url", "")
            
            prompt = f"""
            Generate Selenium automation code for this execution:
            
            URL: {url}
            Actions: {actions}
            Execution Details: {automation_details}
            
            Generate:
            1. Complete Selenium test file with proper imports
            2. WebDriver setup and configuration
            3. Page object model structure
            4. Explicit waits and error handling
            5. Screenshot capture
            6. Test reporting setup
            
            Return as structured code with comments.
            """
            
            code = await self.ai_provider.generate_response(prompt)
            
            return {
                "code": code,
                "framework": "selenium",
                "language": "python",
                "features": ["screenshots", "explicit_waits", "error_handling", "page_objects"]
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _generate_cypress_code(self, execution_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate Cypress automation code."""
        try:
            # Extract automation details
            automation_details = execution_result.get("result", {}).get("automation_details", {})
            actions = automation_details.get("actions", [])
            url = execution_result.get("result", {}).get("url", "")
            
            prompt = f"""
            Generate Cypress automation code for this execution:
            
            URL: {url}
            Actions: {actions}
            Execution Details: {automation_details}
            
            Generate:
            1. Complete Cypress test file with proper structure
            2. Custom commands for common actions
            3. Page object model using Cypress
            4. Assertions and error handling
            5. Screenshot and video capture
            6. Test configuration
            
            Return as structured code with comments.
            """
            
            code = await self.ai_provider.generate_response(prompt)
            
            return {
                "code": code,
                "framework": "cypress",
                "language": "javascript",
                "features": ["screenshots", "video", "custom_commands", "assertions"]
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def generate_test_report(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        try:
            prompt = f"""
            Generate a comprehensive test report for this automation execution:
            
            Execution Result: {execution_result}
            
            Include:
            1. Test summary and results
            2. Performance metrics
            3. Screenshots and media
            4. Error analysis
            5. Recommendations
            6. Generated code snippets
            
            Return as structured JSON report.
            """
            
            report = await self.ai_provider.generate_response(prompt)
            
            return {
                "report": report,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def generate_documentation(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation for the automation."""
        try:
            prompt = f"""
            Generate documentation for this automation:
            
            Execution Result: {execution_result}
            
            Include:
            1. Overview and purpose
            2. Prerequisites and setup
            3. Step-by-step execution guide
            4. Configuration options
            5. Troubleshooting guide
            6. Maintenance instructions
            
            Return as structured documentation.
            """
            
            documentation = await self.ai_provider.generate_response(prompt)
            
            return {
                "documentation": documentation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}