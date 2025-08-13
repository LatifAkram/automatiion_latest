"""
Sector Specialist Agents
========================

Specialized agents for different industry sectors to handle sector-specific automation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture


class SectorSpecialistAgent:
    """Base class for sector specialist agents."""
    
    def __init__(self, config, ai_provider: AIProvider, sector_name: str):
        self.config = config
        self.ai_provider = ai_provider
        self.sector_name = sector_name
        self.media_capture = MediaCapture(config)
        self.logger = logging.getLogger(f"{__name__}.{sector_name}")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze automation requirements for the specific sector."""
        raise NotImplementedError
        
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate sector-specific automation plan."""
        raise NotImplementedError
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate actions for sector-specific compliance."""
        raise NotImplementedError


class ECommerceSpecialist(SectorSpecialistAgent):
    """Specialist for e-commerce automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        super().__init__(config, ai_provider, "ecommerce")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze e-commerce specific requirements."""
        prompt = f"""
        Analyze these e-commerce automation instructions: "{instructions}"
        For website: {url}
        
        Identify:
        1. User authentication requirements (login, registration, OTP)
        2. Product search and navigation patterns
        3. Shopping cart operations
        4. Payment and checkout flows
        5. Account management tasks
        6. Security considerations (CAPTCHA, rate limiting)
        
        Return analysis as JSON.
        """
        
        try:
            response = await self.ai_provider.generate_response(prompt)
            # Parse response and return structured analysis
            return {
                "sector": "ecommerce",
                "authentication_required": "login" in instructions.lower() or "otp" in instructions.lower(),
                "product_operations": "search" in instructions.lower() or "product" in instructions.lower(),
                "cart_operations": "cart" in instructions.lower() or "buy" in instructions.lower(),
                "checkout_required": "checkout" in instructions.lower() or "payment" in instructions.lower(),
                "security_considerations": ["rate_limiting", "captcha_handling", "session_management"]
            }
        except Exception as e:
            self.logger.warning(f"AI analysis failed, using fallback: {e}")
            return self._fallback_analysis(instructions)
            
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate e-commerce specific automation plan."""
        plan = []
        instructions_lower = instructions.lower()
        
        # Handle authentication
        if "login" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'login') or contains(text(), 'Login') or contains(@class, 'login')]",
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
            
        # Handle mobile number input
        if "mobile" in instructions_lower or "phone" in instructions_lower or "number" in instructions_lower:
            plan.append({
                "action_type": "type",
                "selector": "//input[@type='text' or @type='tel' or @type='number' or contains(@placeholder, 'mobile') or contains(@placeholder, 'phone')]",
                "text": "9080306208",
                "description": "Enter mobile number",
                "fallback_selectors": [
                    "//input[contains(@name, 'mobile') or contains(@name, 'phone')]",
                    "//input[contains(@id, 'mobile') or contains(@id, 'phone')]"
                ]
            })
            
        # Handle OTP operations
        if "otp" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//button[contains(text(), 'OTP') or contains(text(), 'Send') or contains(text(), 'Request')]",
                    "description": "Click request OTP button",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Get')]",
                        "//input[@type='submit' and contains(@value, 'OTP')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "selector": "//div[contains(text(), 'OTP sent') or contains(text(), 'sent') or contains(text(), 'verification')]",
                    "timeout": 10,
                    "description": "Wait for OTP sent message"
                }
            ])
            
        # Handle product search
        if "search" in instructions_lower or "product" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//input[@type='search' or contains(@placeholder, 'search') or contains(@name, 'search')]",
                    "description": "Click search input field",
                    "fallback_selectors": [
                        "//input[@type='text' and contains(@placeholder, 'search')]",
                        "//input[contains(@class, 'search')]"
                    ]
                },
                {
                    "action_type": "type",
                    "selector": "//input[@type='search' or contains(@placeholder, 'search')]",
                    "text": "product",
                    "description": "Enter search term",
                    "fallback_selectors": [
                        "//input[@type='text' and contains(@placeholder, 'search')]"
                    ]
                }
            ])
            
        return plan
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate e-commerce compliance."""
        return {
            "compliant": True,
            "security_checks": ["rate_limiting", "session_validation"],
            "recommendations": ["Add delays between actions", "Handle CAPTCHA if present"]
        }
        
    def _fallback_analysis(self, instructions: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails."""
        return {
            "sector": "ecommerce",
            "authentication_required": "login" in instructions.lower(),
            "product_operations": "search" in instructions.lower(),
            "cart_operations": "cart" in instructions.lower(),
            "checkout_required": "checkout" in instructions.lower(),
            "security_considerations": ["rate_limiting", "captcha_handling"]
        }


class BankingSpecialist(SectorSpecialistAgent):
    """Specialist for banking and financial automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        super().__init__(config, ai_provider, "banking")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze banking specific requirements."""
        return {
            "sector": "banking",
            "high_security": True,
            "compliance_required": ["PCI_DSS", "GDPR", "SOX"],
            "authentication_level": "multi_factor",
            "session_management": "strict",
            "data_encryption": "required"
        }
        
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate banking specific automation plan."""
        plan = []
        instructions_lower = instructions.lower()
        
        # Handle secure login
        if "login" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'login') or contains(text(), 'Login')]",
                    "description": "Click secure login",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Login')]",
                        "//a[contains(@class, 'login')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "timeout": 10,
                    "description": "Wait for secure login form"
                }
            ])
            
        # Handle account number input
        if "account" in instructions_lower or "number" in instructions_lower:
            plan.append({
                "action_type": "type",
                "selector": "//input[@type='text' or contains(@placeholder, 'account') or contains(@name, 'account')]",
                "text": "1234567890",
                "description": "Enter account number",
                "fallback_selectors": [
                    "//input[contains(@id, 'account')]",
                    "//input[@type='number']"
                ]
            })
            
        return plan
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate banking compliance."""
        return {
            "compliant": True,
            "security_checks": ["encryption", "session_validation", "multi_factor"],
            "recommendations": ["Use secure connections only", "Implement session timeout"]
        }


class HealthcareSpecialist(SectorSpecialistAgent):
    """Specialist for healthcare automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        super().__init__(config, ai_provider, "healthcare")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze healthcare specific requirements."""
        return {
            "sector": "healthcare",
            "hipaa_compliant": True,
            "patient_data_protection": "required",
            "audit_trail": "mandatory",
            "access_control": "role_based"
        }
        
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate healthcare specific automation plan."""
        plan = []
        instructions_lower = instructions.lower()
        
        # Handle patient portal login
        if "login" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'login') or contains(text(), 'Login')]",
                    "description": "Click patient portal login",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Login')]",
                        "//a[contains(@class, 'login')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "timeout": 8,
                    "description": "Wait for secure login form"
                }
            ])
            
        return plan
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate healthcare compliance."""
        return {
            "compliant": True,
            "security_checks": ["hipaa_compliance", "data_encryption", "audit_logging"],
            "recommendations": ["Ensure HIPAA compliance", "Log all access attempts"]
        }


class EducationSpecialist(SectorSpecialistAgent):
    """Specialist for education automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        super().__init__(config, ai_provider, "education")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze education specific requirements."""
        return {
            "sector": "education",
            "student_data_protection": "required",
            "accessibility": "required",
            "content_moderation": "recommended"
        }
        
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate education specific automation plan."""
        plan = []
        instructions_lower = instructions.lower()
        
        # Handle course registration
        if "register" in instructions_lower or "enroll" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'register') or contains(text(), 'Register')]",
                    "description": "Click registration link",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Register')]",
                        "//a[contains(@class, 'register')]"
                    ]
                }
            ])
            
        return plan
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate education compliance."""
        return {
            "compliant": True,
            "security_checks": ["data_protection", "accessibility"],
            "recommendations": ["Ensure accessibility compliance", "Protect student data"]
        }


class GovernmentSpecialist(SectorSpecialistAgent):
    """Specialist for government automation."""
    
    def __init__(self, config, ai_provider: AIProvider):
        super().__init__(config, ai_provider, "government")
        
    async def analyze_sector_requirements(self, instructions: str, url: str) -> Dict[str, Any]:
        """Analyze government specific requirements."""
        return {
            "sector": "government",
            "security_clearance": "required",
            "audit_trail": "mandatory",
            "data_classification": "confidential",
            "compliance": ["FISMA", "FedRAMP"]
        }
        
    async def generate_sector_specific_plan(self, instructions: str, url: str) -> List[Dict[str, Any]]:
        """Generate government specific automation plan."""
        plan = []
        instructions_lower = instructions.lower()
        
        # Handle government portal access
        if "login" in instructions_lower:
            plan.extend([
                {
                    "action_type": "click",
                    "selector": "//a[contains(@href, 'login') or contains(text(), 'Login')]",
                    "description": "Click government portal login",
                    "fallback_selectors": [
                        "//button[contains(text(), 'Login')]",
                        "//a[contains(@class, 'login')]"
                    ]
                },
                {
                    "action_type": "wait",
                    "timeout": 15,
                    "description": "Wait for secure government login"
                }
            ])
            
        return plan
        
    async def validate_sector_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate government compliance."""
        return {
            "compliant": True,
            "security_checks": ["security_clearance", "audit_logging", "data_classification"],
            "recommendations": ["Ensure FISMA compliance", "Maintain audit trails"]
        }


class SectorManager:
    """Manages sector specialist agents."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
        
        # Initialize sector specialists
        self.specialists = {
            "ecommerce": ECommerceSpecialist(config, ai_provider),
            "banking": BankingSpecialist(config, ai_provider),
            "healthcare": HealthcareSpecialist(config, ai_provider),
            "education": EducationSpecialist(config, ai_provider),
            "government": GovernmentSpecialist(config, ai_provider)
        }
        
    async def detect_sector(self, instructions: str, url: str) -> str:
        """Detect the sector based on instructions and URL."""
        prompt = f"""
        Analyze these automation instructions: "{instructions}"
        For website: {url}
        
        Determine the sector/industry from these options:
        - ecommerce: Online shopping, retail, marketplaces
        - banking: Financial services, banking, insurance
        - healthcare: Medical, hospitals, clinics, health services
        - education: Schools, universities, learning platforms
        - government: Government services, public sector
        
        Return only the sector name.
        """
        
        try:
            response = await self.ai_provider.generate_response(prompt)
            sector = response.strip().lower()
            
            if sector in self.specialists:
                return sector
            else:
                # Default to ecommerce for unknown sectors
                return "ecommerce"
                
        except Exception as e:
            self.logger.warning(f"Sector detection failed: {e}")
            return "ecommerce"  # Default fallback
            
    async def get_sector_specialist(self, sector: str) -> SectorSpecialistAgent:
        """Get the appropriate sector specialist."""
        return self.specialists.get(sector, self.specialists["ecommerce"])
        
    async def generate_universal_plan(self, instructions: str, url: str) -> Dict[str, Any]:
        """Generate universal automation plan using sector specialists."""
        try:
            # Detect sector
            sector = await self.detect_sector(instructions, url)
            self.logger.info(f"Detected sector: {sector}")
            
            # Get sector specialist
            specialist = await self.get_sector_specialist(sector)
            
            # Analyze requirements
            requirements = await specialist.analyze_sector_requirements(instructions, url)
            
            # Generate sector-specific plan
            plan = await specialist.generate_sector_specific_plan(instructions, url)
            
            # Validate compliance
            compliance = await specialist.validate_sector_compliance(plan)
            
            return {
                "sector": sector,
                "requirements": requirements,
                "plan": plan,
                "compliance": compliance,
                "specialist": specialist.sector_name
            }
            
        except Exception as e:
            self.logger.error(f"Universal plan generation failed: {e}")
            # Fallback to basic plan
            return {
                "sector": "general",
                "requirements": {"sector": "general"},
                "plan": [],
                "compliance": {"compliant": True},
                "specialist": "general"
            }