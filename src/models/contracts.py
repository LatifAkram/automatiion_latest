"""
SUPER-OMEGA Hard Contracts
=========================

JSON Schemas for deterministic AI collaboration:
1. Step Contract - for workflow step definition
2. Tool/Agent Contract - for function calling
3. Evidence Contract - for audit/report & learning
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ActionType(str, Enum):
    """Supported action types for automation steps."""
    CLICK = "click"
    TYPE = "type"
    KEYPRESS = "keypress"
    HOVER = "hover"
    SCROLL = "scroll"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    VERIFY = "verify"


class TargetSelector(BaseModel):
    """Target element selector with multiple fallback strategies."""
    role: Optional[str] = Field(None, description="ARIA role")
    name: Optional[str] = Field(None, description="Accessible name")
    text: Optional[str] = Field(None, description="Text content")
    css: Optional[str] = Field(None, description="CSS selector")
    xpath: Optional[str] = Field(None, description="XPath selector")
    id: Optional[str] = Field(None, description="Element ID")
    class_name: Optional[str] = Field(None, description="CSS class name")
    tag_name: Optional[str] = Field(None, description="HTML tag name")
    attributes: Optional[Dict[str, str]] = Field(None, description="Element attributes")
    context: Optional[str] = Field(None, description="Contextual information")
    visual_template: Optional[str] = Field(None, description="Visual template path")
    semantic_embedding: Optional[List[float]] = Field(None, description="Semantic text embedding")


class Action(BaseModel):
    """Action definition for automation step."""
    type: ActionType = Field(..., description="Action type")
    target: Optional[TargetSelector] = Field(None, description="Target element selector")
    value: Optional[str] = Field(None, description="Input value for type actions")
    keys: Optional[List[str]] = Field(None, description="Keys for keypress actions")
    coordinates: Optional[Dict[str, int]] = Field(None, description="X,Y coordinates for click actions")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional action options")


class StepContract(BaseModel):
    """
    Step Contract (JSON Schema) - Core workflow step definition
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique step identifier")
    goal: str = Field(..., description="Step goal description")
    pre: List[str] = Field(default_factory=list, description="Preconditions that must be met")
    action: Action = Field(..., description="Action to perform")
    post: List[str] = Field(default_factory=list, description="Postconditions to verify")
    fallbacks: List[Action] = Field(default_factory=list, description="Fallback actions if primary fails")
    timeout_ms: int = Field(8000, description="Timeout in milliseconds")
    retries: int = Field(2, description="Number of retry attempts")
    evidence: List[str] = Field(default_factory=lambda: ["screenshot", "dom_diff", "event_log"], 
                               description="Evidence to capture")
    confidence_threshold: float = Field(0.8, description="Minimum confidence threshold")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class ToolInputSchema(BaseModel):
    """Input schema for tool/agent functions."""
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[str]] = Field(None, description="Allowed values")


class ToolOutputSchema(BaseModel):
    """Output schema for tool/agent functions."""
    type: str = Field(..., description="Return type")
    description: str = Field(..., description="Return description")
    properties: Optional[Dict[str, Any]] = Field(None, description="Object properties")


class ToolAgentContract(BaseModel):
    """
    Tool/Agent Contract - Function calling interface
    """
    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    input_schema: Dict[str, ToolInputSchema] = Field(..., description="Input parameters schema")
    output_schema: ToolOutputSchema = Field(..., description="Output schema")
    version: str = Field("1.0.0", description="Contract version")
    timeout_ms: int = Field(30000, description="Function timeout")
    retries: int = Field(3, description="Retry attempts")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceType(str, Enum):
    """Types of evidence that can be captured."""
    SCREENSHOT = "screenshot"
    DOM_SNAPSHOT = "dom_snapshot"
    DOM_DIFF = "dom_diff"
    EVENT_LOG = "event_log"
    NETWORK_LOG = "network_log"
    CONSOLE_LOG = "console_log"
    VIDEO = "video"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"


class FactSource(str, Enum):
    """Sources for real-time facts."""
    OFFICIAL = "official"
    PRIMARY = "primary"
    REPUTABLE = "reputable"
    SOCIAL = "social"
    API = "api"
    SCRAPE = "scrape"


class Fact(BaseModel):
    """Real-time fact with attribution."""
    value: Any = Field(..., description="Fact value")
    source: FactSource = Field(..., description="Fact source type")
    url: str = Field(..., description="Source URL")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    trust_score: float = Field(..., description="Trust score 0-1")
    verification_count: int = Field(1, description="Number of independent verifications")


class StepEvidence(BaseModel):
    """Evidence captured for a single step."""
    step_id: str = Field(..., description="Step identifier")
    type: EvidenceType = Field(..., description="Evidence type")
    data: Union[str, bytes, Dict[str, Any]] = Field(..., description="Evidence data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_path: Optional[str] = Field(None, description="File path if stored externally")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RunReport(BaseModel):
    """Complete run report with all evidence."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = Field(..., description="Overall run goal")
    status: str = Field(..., description="Run status")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    steps: List[StepContract] = Field(default_factory=list)
    evidence: List[StepEvidence] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    generated_code: Optional[Dict[str, str]] = Field(None, description="Generated automation code")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    error_log: Optional[List[str]] = Field(None, description="Error messages")


class EvidenceContract(BaseModel):
    """
    Evidence Contract - for audit/report & learning
    File structure: /runs/<id>/...
    """
    run_id: str = Field(..., description="Run identifier")
    report: RunReport = Field(..., description="Main report")
    step_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-step details")
    media_files: Dict[str, str] = Field(default_factory=dict, description="Media file paths")
    code_artifacts: Dict[str, str] = Field(default_factory=dict, description="Generated code files")
    facts_log: List[Fact] = Field(default_factory=list, description="All facts collected")
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: v.decode('utf-8') if isinstance(v, bytes) else str(v)
        }


# Contract validation functions
def validate_step_contract(step_data: Dict[str, Any]) -> StepContract:
    """Validate and create a step contract from raw data."""
    return StepContract(**step_data)


def validate_tool_contract(tool_data: Dict[str, Any]) -> ToolAgentContract:
    """Validate and create a tool/agent contract from raw data."""
    return ToolAgentContract(**tool_data)


def validate_evidence_contract(evidence_data: Dict[str, Any]) -> EvidenceContract:
    """Validate and create an evidence contract from raw data."""
    return EvidenceContract(**evidence_data)


# Example contracts for reference
EXAMPLE_STEP_CONTRACT = {
    "id": "uuid",
    "goal": "send_email",
    "pre": ["exists(role=button,name='Compose')", "visible('Compose')"],
    "action": {
        "type": "click",
        "target": {
            "role": "button",
            "name": "Compose"
        }
    },
    "post": ["dialog_open(name='New message')"],
    "fallbacks": [{"type": "keypress", "keys": ["c"]}],
    "timeout_ms": 8000,
    "retries": 2,
    "evidence": ["screenshot", "dom_diff", "event_log"]
}

EXAMPLE_TOOL_CONTRACT = {
    "name": "find_element",
    "description": "Find element using multiple selector strategies",
    "input_schema": {
        "role": {"type": "string", "description": "ARIA role", "required": False},
        "name": {"type": "string", "description": "Accessible name", "required": False},
        "context": {"type": "object", "description": "Context information", "required": False}
    },
    "output_schema": {
        "type": "object",
        "description": "Element location result",
        "properties": {
            "element_id": {"type": "string"},
            "confidence": {"type": "number"},
            "locators": {"type": "array"}
        }
    }
}