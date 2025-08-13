"""
Automation API endpoints for intelligent automation with advanced capabilities
"""

import logging
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/automation", tags=["automation"])

class IntelligentAutomationRequest(BaseModel):
    """Request model for intelligent automation"""
    instructions: str
    url: str = "https://www.google.com"

class AutomationResponse(BaseModel):
    """Response model for automation"""
    status: str
    automation_id: str
    result: Dict[str, Any]
    message: str

@router.post("/intelligent", response_model=AutomationResponse)
async def intelligent_automation(request: IntelligentAutomationRequest):
    """Execute intelligent automation with advanced capabilities analysis"""
    try:
        logger.info(f"Intelligent automation request: {request.instructions}")
        
        # Get the orchestrator
        orchestrator = get_orchestrator()
        
        # Analyze automation requirements using advanced capabilities
        analysis = await orchestrator.analyze_automation_requirements(request.instructions)
        logger.info(f"Automation analysis: {analysis}")
        
        # Generate comprehensive automation plan
        plan = await orchestrator.generate_comprehensive_automation_plan(request.instructions)
        logger.info(f"Automation plan: {plan}")
        
        # Execute the automation based on the plan
        result = await orchestrator.execute_intelligent_automation(
            instructions=request.instructions,
            url=request.url,
            analysis=analysis,
            plan=plan
        )
        
        return AutomationResponse(
            status="success",
            automation_id=result.get("automation_id", str(uuid.uuid4())),
            result={
                "summary": result.get("summary", "Automation completed successfully"),
                "url": request.url,
                "automation_plan": plan.get("steps", []),
                "capabilities_used": [cap.name for cap in plan.get("capabilities_used", [])],
                "complexity_level": analysis.get("complexity_level", "simple"),
                "estimated_duration": analysis.get("estimated_duration", 300),
                "risk_level": analysis.get("risk_level", "low"),
                "compliance_requirements": analysis.get("compliance_requirements", []),
                "orchestrator_info": plan.get("orchestrator_info", {}),
                "validation": plan.get("validation", {}),
                "screenshots": result.get("screenshots", []),
                "progress": result.get("progress", 100)
            },
            message="Intelligent automation executed successfully with advanced capabilities analysis"
        )
        
    except Exception as e:
        logger.error(f"Intelligent automation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Intelligent automation failed: {str(e)}")

@router.get("/capabilities")
async def get_automation_capabilities():
    """Get all available automation capabilities"""
    try:
        orchestrator = get_orchestrator()
        capabilities = await orchestrator.advanced_capabilities.get_all_capabilities()
        
        return {
            "status": "success",
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "category": cap.category.value,
                    "complexity": cap.complexity,
                    "examples": cap.examples,
                    "selectors": cap.selectors,
                    "validation_rules": cap.validation_rules
                }
                for cap in capabilities
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.post("/analyze")
async def analyze_automation_request(instructions: str):
    """Analyze automation request and return detailed analysis"""
    try:
        orchestrator = get_orchestrator()
        analysis = await orchestrator.analyze_automation_requirements(instructions)
        
        return {
            "status": "success",
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Failed to analyze automation request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze automation request: {str(e)}")

@router.post("/plan")
async def generate_automation_plan(instructions: str):
    """Generate comprehensive automation plan"""
    try:
        orchestrator = get_orchestrator()
        plan = await orchestrator.generate_comprehensive_automation_plan(instructions)
        
        return {
            "status": "success",
            "plan": plan
        }
    except Exception as e:
        logger.error(f"Failed to generate automation plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate automation plan: {str(e)}")