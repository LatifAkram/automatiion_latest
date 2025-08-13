"""
Advanced Automation Capabilities
================================

Enterprise-level automation capabilities covering all aspects of RPA and automation.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from playwright.async_api import Page, ElementHandle
import aiohttp
from dataclasses import dataclass

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture


class AutomationCategory(Enum):
    """Categories of automation capabilities."""
    WEB_AUTOMATION = "web_automation"
    DATA_EXTRACTION = "data_extraction"
    FORM_HANDLING = "form_handling"
    API_INTEGRATION = "api_integration"
    FILE_PROCESSING = "file_processing"
    EMAIL_AUTOMATION = "email_automation"
    DOCUMENT_PROCESSING = "document_processing"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    ERROR_HANDLING = "error_handling"
    COMPLIANCE = "compliance"


class ComplexityLevel(Enum):
    """Complexity levels for automation tasks."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class AutomationCapability:
    """Represents an automation capability."""
    name: str
    description: str
    category: AutomationCategory
    complexity: ComplexityLevel
    examples: List[str]
    selectors: List[str]
    validation_rules: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: int
    risk_level: str
    compliance_requirements: List[str]


class AdvancedAutomationCapabilities:
    """Advanced automation capabilities for enterprise-level automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        self.media_capture = MediaCapture(config.database.media_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize capabilities
        self.capabilities = self._initialize_capabilities()
        
        # Performance tracking
        self.execution_history = []
        self.error_patterns = {}
        self.success_patterns = {}
        
    def _initialize_capabilities(self) -> Dict[str, AutomationCapability]:
        """Initialize all automation capabilities."""
        capabilities = {}
        
        # Web Automation Capabilities
        capabilities["web_navigation"] = AutomationCapability(
            name="Web Navigation",
            description="Advanced web navigation with intelligent waiting and error handling",
            category=AutomationCategory.WEB_AUTOMATION,
            complexity=ComplexityLevel.MEDIUM,
            examples=["Navigate to specific pages", "Handle redirects", "Wait for page loads"],
            selectors=["//a[@href]", "//button", "//input[@type='submit']"],
            validation_rules={"timeout": 30, "retry_attempts": 3},
            dependencies=[],
            estimated_duration=10,
            risk_level="low",
            compliance_requirements=[]
        )
        
        capabilities["form_automation"] = AutomationCapability(
            name="Form Automation",
            description="Intelligent form filling with validation and error handling",
            category=AutomationCategory.FORM_HANDLING,
            complexity=ComplexityLevel.COMPLEX,
            examples=["Fill complex forms", "Handle dynamic fields", "Validate inputs"],
            selectors=["//input", "//select", "//textarea", "//button"],
            validation_rules={"required_fields": True, "format_validation": True},
            dependencies=["web_navigation"],
            estimated_duration=30,
            risk_level="medium",
            compliance_requirements=["data_validation"]
        )
        
        capabilities["data_extraction"] = AutomationCapability(
            name="Data Extraction",
            description="Advanced data extraction from web pages, tables, and documents",
            category=AutomationCategory.DATA_EXTRACTION,
            complexity=ComplexityLevel.COMPLEX,
            examples=["Extract table data", "Parse JSON responses", "Handle pagination"],
            selectors=["//table", "//div[@class='data']", "//span[@data-value]"],
            validation_rules={"data_format": True, "completeness_check": True},
            dependencies=["web_navigation"],
            estimated_duration=20,
            risk_level="low",
            compliance_requirements=["data_integrity"]
        )
        
        capabilities["api_integration"] = AutomationCapability(
            name="API Integration",
            description="REST API integration with authentication and error handling",
            category=AutomationCategory.API_INTEGRATION,
            complexity=ComplexityLevel.ENTERPRISE,
            examples=["Call REST APIs", "Handle authentication", "Process responses"],
            selectors=[],
            validation_rules={"authentication": True, "rate_limiting": True},
            dependencies=[],
            estimated_duration=15,
            risk_level="medium",
            compliance_requirements=["api_security", "data_encryption"]
        )
        
        capabilities["file_processing"] = AutomationCapability(
            name="File Processing",
            description="Advanced file operations including Excel, PDF, and CSV processing",
            category=AutomationCategory.FILE_PROCESSING,
            complexity=ComplexityLevel.COMPLEX,
            examples=["Read Excel files", "Generate PDF reports", "Process CSV data"],
            selectors=[],
            validation_rules={"file_format": True, "backup_creation": True},
            dependencies=[],
            estimated_duration=25,
            risk_level="low",
            compliance_requirements=["data_backup"]
        )
        
        capabilities["email_automation"] = AutomationCapability(
            name="Email Automation",
            description="Email sending, receiving, and processing automation",
            category=AutomationCategory.EMAIL_AUTOMATION,
            complexity=ComplexityLevel.ENTERPRISE,
            examples=["Send automated emails", "Process email responses", "Handle attachments"],
            selectors=[],
            validation_rules={"email_validation": True, "attachment_handling": True},
            dependencies=[],
            estimated_duration=20,
            risk_level="medium",
            compliance_requirements=["email_security", "data_privacy"]
        )
        
        capabilities["workflow_orchestration"] = AutomationCapability(
            name="Workflow Orchestration",
            description="Complex workflow orchestration with conditional logic and parallel execution",
            category=AutomationCategory.WORKFLOW_ORCHESTRATION,
            complexity=ComplexityLevel.ENTERPRISE,
            examples=["Parallel task execution", "Conditional workflows", "Error recovery"],
            selectors=[],
            validation_rules={"workflow_validation": True, "rollback_capability": True},
            dependencies=["web_navigation", "form_automation", "data_extraction"],
            estimated_duration=60,
            risk_level="high",
            compliance_requirements=["audit_trail", "rollback_capability"]
        )
        
        return capabilities
    
    async def analyze_automation_requirements(self, user_request: str) -> Dict[str, Any]:
        """Analyze user request to determine required capabilities."""
        try:
            prompt = f"""
            Analyze this automation request: "{user_request}"
            
            Determine the required capabilities from these categories:
            - web_automation: Navigation, clicking, form filling
            - data_extraction: Extracting data from web pages, tables, documents
            - form_handling: Complex form automation with validation
            - api_integration: REST API calls, authentication
            - file_processing: Excel, PDF, CSV file operations
            - email_automation: Email sending, receiving, processing
            - document_processing: Document parsing, OCR, text extraction
            - workflow_orchestration: Complex workflows, parallel execution
            - error_handling: Error recovery, retry logic
            - compliance: Data validation, security, audit trails
            
            Return as JSON with:
            - required_capabilities: List of capability names
            - complexity_level: simple/medium/complex/enterprise
            - estimated_duration: Time in seconds
            - risk_level: low/medium/high
            - compliance_requirements: List of compliance needs
            - technical_requirements: List of technical needs
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback analysis
                analysis = self._fallback_requirement_analysis(user_request)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze automation requirements: {e}")
            return self._fallback_requirement_analysis(user_request)
    
    def _fallback_requirement_analysis(self, user_request: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails."""
        request_lower = user_request.lower()
        
        capabilities = []
        complexity = "simple"
        duration = 30
        risk = "low"
        
        if any(word in request_lower for word in ["form", "fill", "input", "submit"]):
            capabilities.extend(["form_automation", "web_navigation"])
            complexity = "medium"
            duration = 45
            
        if any(word in request_lower for word in ["extract", "data", "table", "scrape"]):
            capabilities.extend(["data_extraction", "web_navigation"])
            complexity = "medium"
            duration = 40
            
        if any(word in request_lower for word in ["api", "rest", "json", "http"]):
            capabilities.extend(["api_integration"])
            complexity = "complex"
            duration = 25
            
        if any(word in request_lower for word in ["excel", "pdf", "file", "csv"]):
            capabilities.extend(["file_processing"])
            complexity = "medium"
            duration = 35
            
        if any(word in request_lower for word in ["email", "mail", "send"]):
            capabilities.extend(["email_automation"])
            complexity = "complex"
            duration = 30
            
        if any(word in request_lower for word in ["workflow", "parallel", "orchestrate"]):
            capabilities.extend(["workflow_orchestration"])
            complexity = "enterprise"
            duration = 90
            risk = "high"
            
        if not capabilities:
            capabilities = ["web_navigation"]
            
        return {
            "required_capabilities": capabilities,
            "complexity_level": complexity,
            "estimated_duration": duration,
            "risk_level": risk,
            "compliance_requirements": [],
            "technical_requirements": []
        }
    
    async def generate_automation_plan(self, user_request: str) -> Dict[str, Any]:
        """Generate comprehensive automation plan."""
        try:
            # Analyze requirements
            requirements = await self.analyze_automation_requirements(user_request)
            
            # Generate detailed plan
            plan = {
                "analysis": requirements,
                "steps": [],
                "capabilities_used": requirements["required_capabilities"],
                "estimated_completion_time": requirements["estimated_duration"],
                "risk_assessment": requirements["risk_level"],
                "compliance_checklist": requirements["compliance_requirements"],
                "validation": {"is_feasible": True, "warnings": [], "risks": [], "recommendations": []}
            }
            
            # Generate steps for each capability
            for capability_name in requirements["required_capabilities"]:
                capability = self.capabilities.get(capability_name)
                if capability:
                    steps = await self._generate_capability_steps(capability, user_request)
                    plan["steps"].extend(steps)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to generate automation plan: {e}")
            return self._generate_fallback_plan(user_request)
    
    async def _generate_capability_steps(self, capability: AutomationCapability, user_request: str) -> List[Dict[str, Any]]:
        """Generate steps for a specific capability."""
        steps = []
        
        if capability.category == AutomationCategory.WEB_AUTOMATION:
            steps.extend([
                {
                    "action": "navigate",
                    "description": f"Navigate to target page for {capability.name}",
                    "timeout": 30,
                    "retry_attempts": 3
                },
                {
                    "action": "wait",
                    "description": "Wait for page to load completely",
                    "duration": 5
                }
            ])
            
        elif capability.category == AutomationCategory.FORM_HANDLING:
            steps.extend([
                {
                    "action": "wait_for_element",
                    "description": "Wait for form elements to be visible",
                    "selector": "//form",
                    "timeout": 10
                },
                {
                    "action": "validate_form",
                    "description": "Validate form structure and required fields",
                    "validation_rules": capability.validation_rules
                }
            ])
            
        elif capability.category == AutomationCategory.DATA_EXTRACTION:
            steps.extend([
                {
                    "action": "identify_data_sources",
                    "description": "Identify data sources on the page",
                    "selectors": capability.selectors
                },
                {
                    "action": "extract_data",
                    "description": "Extract data using identified selectors",
                    "data_format": "structured"
                }
            ])
            
        elif capability.category == AutomationCategory.API_INTEGRATION:
            steps.extend([
                {
                    "action": "authenticate_api",
                    "description": "Authenticate with API service",
                    "auth_type": "bearer_token"
                },
                {
                    "action": "call_api",
                    "description": "Make API call with proper error handling",
                    "method": "POST",
                    "timeout": 30
                }
            ])
            
        return steps
    
    def _generate_fallback_plan(self, user_request: str) -> Dict[str, Any]:
        """Generate fallback automation plan."""
        return {
            "analysis": {
                "required_capabilities": ["web_navigation"],
                "complexity_level": "simple",
                "estimated_duration": 30,
                "risk_level": "low"
            },
            "steps": [
                {
                    "action": "navigate",
                    "description": "Navigate to target website",
                    "timeout": 30
                },
                {
                    "action": "wait",
                    "description": "Wait for page to load",
                    "duration": 5
                }
            ],
            "capabilities_used": ["web_navigation"],
            "estimated_completion_time": 30,
            "risk_assessment": "low",
            "compliance_checklist": [],
            "validation": {"is_feasible": True, "warnings": [], "risks": [], "recommendations": []}
        }
    
    async def validate_automation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate automation plan for feasibility and risks."""
        try:
            validation = {
                "is_feasible": True,
                "warnings": [],
                "risks": [],
                "recommendations": []
            }
            
            # Check capability dependencies
            for capability_name in plan.get("capabilities_used", []):
                capability = self.capabilities.get(capability_name)
                if capability:
                    # Check dependencies
                    for dependency in capability.dependencies:
                        if dependency not in plan.get("capabilities_used", []):
                            validation["warnings"].append(f"Missing dependency: {dependency}")
                    
                    # Check risk level
                    if capability.risk_level == "high":
                        validation["risks"].append(f"High risk capability: {capability_name}")
                    
                    # Check compliance requirements
                    for requirement in capability.compliance_requirements:
                        if requirement not in plan.get("compliance_checklist", []):
                            validation["recommendations"].append(f"Add compliance: {requirement}")
            
            # Check estimated duration
            if plan.get("estimated_completion_time", 0) > 300:  # 5 minutes
                validation["warnings"].append("Long execution time - consider optimization")
            
            # Check complexity
            if plan.get("analysis", {}).get("complexity_level") == "enterprise":
                validation["risks"].append("Enterprise complexity - requires careful testing")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Failed to validate automation plan: {e}")
            return {
                "is_feasible": False,
                "warnings": [f"Validation failed: {str(e)}"],
                "risks": ["Unknown risks due to validation failure"],
                "recommendations": ["Review and fix validation issues"]
            }
    
    async def get_capability(self, capability_name: str) -> Optional[AutomationCapability]:
        """Get a specific automation capability."""
        return self.capabilities.get(capability_name)
    
    async def list_capabilities(self) -> List[Dict[str, Any]]:
        """List all available automation capabilities."""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "category": cap.category.value,
                "complexity": cap.complexity.value,
                "examples": cap.examples,
                "estimated_duration": cap.estimated_duration,
                "risk_level": cap.risk_level
            }
            for cap in self.capabilities.values()
        ]
    
    async def execute_capability(self, capability_name: str, page: Page, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific automation capability."""
        try:
            capability = self.capabilities.get(capability_name)
            if not capability:
                return {"success": False, "error": f"Capability not found: {capability_name}"}
            
            self.logger.info(f"Executing capability: {capability_name}")
            
            if capability.category == AutomationCategory.WEB_AUTOMATION:
                return await self._execute_web_automation(page, context)
            elif capability.category == AutomationCategory.FORM_HANDLING:
                return await self._execute_form_automation(page, context)
            elif capability.category == AutomationCategory.DATA_EXTRACTION:
                return await self._execute_data_extraction(page, context)
            elif capability.category == AutomationCategory.API_INTEGRATION:
                return await self._execute_api_integration(context)
            elif capability.category == AutomationCategory.FILE_PROCESSING:
                return await self._execute_file_processing(context)
            else:
                return {"success": False, "error": f"Capability execution not implemented: {capability_name}"}
                
        except Exception as e:
            self.logger.error(f"Failed to execute capability {capability_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_web_automation(self, page: Page, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web automation capability."""
        try:
            # Navigate to target URL
            url = context.get("url", "")
            if url:
                await page.goto(url, wait_until="networkidle")
                await asyncio.sleep(2)
            
            # Execute web automation steps
            steps = context.get("steps", [])
            results = []
            
            for step in steps:
                result = await self._execute_web_step(page, step)
                results.append(result)
                
                if not result["success"]:
                    break
                    
                await asyncio.sleep(1)
            
            return {
                "success": all(r["success"] for r in results),
                "results": results,
                "capability": "web_automation"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_web_step(self, page: Page, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single web automation step."""
        try:
            action = step.get("action", "")
            selector = step.get("selector", "")
            
            if action == "click":
                element = await page.wait_for_selector(selector, timeout=10000)
                if element:
                    await element.click()
                    return {"success": True, "action": "click"}
                    
            elif action == "type":
                element = await page.wait_for_selector(selector, timeout=10000)
                if element:
                    text = step.get("text", "")
                    await element.fill(text)
                    return {"success": True, "action": "type", "text": text}
                    
            elif action == "wait":
                duration = step.get("duration", 2)
                await asyncio.sleep(duration)
                return {"success": True, "action": "wait", "duration": duration}
                
            return {"success": False, "error": f"Unknown action: {action}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_form_automation(self, page: Page, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute form automation capability."""
        try:
            form_data = context.get("form_data", {})
            results = []
            
            for field_name, field_value in form_data.items():
                # Find form field
                selector = f"//input[@name='{field_name}' or @id='{field_name}']"
                element = await page.wait_for_selector(selector, timeout=10000)
                
                if element:
                    await element.fill(str(field_value))
                    results.append({"field": field_name, "success": True})
                else:
                    results.append({"field": field_name, "success": False, "error": "Field not found"})
            
            # Submit form
            submit_selector = "//button[@type='submit']"
            submit_button = await page.wait_for_selector(submit_selector, timeout=10000)
            if submit_button:
                await submit_button.click()
                results.append({"action": "submit", "success": True})
            
            return {
                "success": all(r.get("success", False) for r in results),
                "results": results,
                "capability": "form_automation"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_data_extraction(self, page: Page, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction capability."""
        try:
            selectors = context.get("selectors", [])
            extracted_data = {}
            
            for selector in selectors:
                elements = await page.query_selector_all(selector)
                data = []
                
                for element in elements:
                    text = await element.text_content()
                    if text:
                        data.append(text.strip())
                
                extracted_data[selector] = data
            
            return {
                "success": True,
                "data": extracted_data,
                "capability": "data_extraction"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_api_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API integration capability."""
        try:
            url = context.get("url", "")
            method = context.get("method", "GET")
            headers = context.get("headers", {})
            data = context.get("data", {})
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        result = await response.json()
                else:
                    return {"success": False, "error": f"Unsupported method: {method}"}
            
            return {
                "success": True,
                "response": result,
                "capability": "api_integration"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_file_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file processing capability."""
        try:
            file_path = context.get("file_path", "")
            operation = context.get("operation", "read")
            
            if operation == "read_excel":
                df = pd.read_excel(file_path)
                data = df.to_dict("records")
                return {
                    "success": True,
                    "data": data,
                    "capability": "file_processing"
                }
            elif operation == "write_excel":
                data = context.get("data", [])
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)
                return {
                    "success": True,
                    "file_path": file_path,
                    "capability": "file_processing"
                }
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all capabilities."""
        return {
            "total_executions": len(self.execution_history),
            "success_rate": self._calculate_success_rate(),
            "average_duration": self._calculate_average_duration(),
            "error_patterns": self.error_patterns,
            "success_patterns": self.success_patterns,
            "capabilities_used": list(self.capabilities.keys())
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.execution_history:
            return 0.0
        
        successful = sum(1 for execution in self.execution_history if execution.get("success", False))
        return (successful / len(self.execution_history)) * 100
    
    def _calculate_average_duration(self) -> float:
        """Calculate average execution duration."""
        if not self.execution_history:
            return 0.0
        
        durations = [execution.get("duration", 0) for execution in self.execution_history]
        return sum(durations) / len(durations)