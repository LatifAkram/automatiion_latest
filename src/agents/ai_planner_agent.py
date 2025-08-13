"""
AI-1: Planner Agent (Brain)
===========================

Acts as the central brain that:
1. Analyzes task complexity
2. Activates parallel sub-agents
3. Coordinates search across multiple providers
4. Extracts URLs and requirements
5. Creates comprehensive execution plans
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from ..core.ai_provider import AIProvider
from ..utils.vector_store import VectorStore
from ..utils.audit import AuditLogger


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ULTRA_COMPLEX = "ultra_complex"


class SearchProvider(Enum):
    """Available search providers."""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    DOCUMENTATION = "documentation"
    ACADEMIC = "academic"
    NEWS = "news"
    REDDIT = "reddit"
    YOUTUBE = "youtube"


class SubAgentType(Enum):
    """Types of parallel sub-agents."""
    SEARCH_AGENT = "search_agent"
    URL_EXTRACTION_AGENT = "url_extraction_agent"
    DOM_ANALYSIS_AGENT = "dom_analysis_agent"
    COMPLEXITY_ANALYSIS_AGENT = "complexity_analysis_agent"
    REQUIREMENT_ANALYSIS_AGENT = "requirement_analysis_agent"


class AIPlannerAgent:
    """AI-1: Central brain that coordinates all automation tasks."""
    
    def __init__(self, config, ai_provider: AIProvider, vector_store: VectorStore, audit_logger: AuditLogger):
        self.config = config
        self.ai_provider = ai_provider
        self.vector_store = vector_store
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Sub-agents for parallel execution
        self.sub_agents = {}
        self.active_tasks = {}
        
    async def initialize(self):
        """Initialize the planner agent and all sub-agents."""
        self.logger.info("Initializing AI-1 Planner Agent (Brain)...")
        
        # Initialize sub-agents
        await self._initialize_sub_agents()
        
        self.logger.info("AI-1 Planner Agent initialized successfully")
    
    async def _initialize_sub_agents(self):
        """Initialize all parallel sub-agents."""
        # Search Agent for multiple providers
        self.sub_agents[SubAgentType.SEARCH_AGENT] = SearchSubAgent(
            self.config, self.ai_provider, self.audit_logger
        )
        
        # URL Extraction Agent
        self.sub_agents[SubAgentType.URL_EXTRACTION_AGENT] = URLExtractionSubAgent(
            self.config, self.ai_provider, self.audit_logger
        )
        
        # Complexity Analysis Agent
        self.sub_agents[SubAgentType.COMPLEXITY_ANALYSIS_AGENT] = ComplexityAnalysisSubAgent(
            self.config, self.ai_provider, self.audit_logger
        )
        
        # Requirement Analysis Agent
        self.sub_agents[SubAgentType.REQUIREMENT_ANALYSIS_AGENT] = RequirementAnalysisSubAgent(
            self.config, self.ai_provider, self.audit_logger
        )
        
        # Initialize all sub-agents
        for agent_type, agent in self.sub_agents.items():
            await agent.initialize()
            self.logger.info(f"Initialized sub-agent: {agent_type.value}")
    
    async def plan_automation_task(self, user_instructions: str) -> Dict[str, Any]:
        """
        Main planning method that coordinates all sub-agents.
        
        Args:
            user_instructions: User's automation request
            
        Returns:
            Comprehensive execution plan
        """
        try:
            self.logger.info(f"AI-1: Planning automation task: {user_instructions[:100]}...")
            
            # Step 1: Analyze task complexity in parallel
            complexity_task = asyncio.create_task(
                self.sub_agents[SubAgentType.COMPLEXITY_ANALYSIS_AGENT].analyze_complexity(user_instructions)
            )
            
            # Step 2: Extract URLs and requirements in parallel
            url_extraction_task = asyncio.create_task(
                self.sub_agents[SubAgentType.URL_EXTRACTION_AGENT].extract_urls_and_requirements(user_instructions)
            )
            
            # Step 3: Analyze requirements in parallel
            requirement_analysis_task = asyncio.create_task(
                self.sub_agents[SubAgentType.REQUIREMENT_ANALYSIS_AGENT].analyze_requirements(user_instructions)
            )
            
            # Wait for all parallel analyses to complete
            complexity_result, url_result, requirement_result = await asyncio.gather(
                complexity_task, url_extraction_task, requirement_analysis_task
            )
            
            # Step 4: Perform comprehensive search if needed
            search_results = {}
            if complexity_result["complexity"] in [TaskComplexity.COMPLEX, TaskComplexity.ULTRA_COMPLEX]:
                search_results = await self._perform_comprehensive_search(
                    user_instructions, url_result["urls"], requirement_result["requirements"]
                )
            
            # Step 5: Generate comprehensive execution plan
            execution_plan = await self._generate_execution_plan(
                user_instructions, complexity_result, url_result, requirement_result, search_results
            )
            
            # Step 6: Validate and optimize plan
            optimized_plan = await self._validate_and_optimize_plan(execution_plan)
            
            self.logger.info(f"AI-1: Generated comprehensive plan with {len(optimized_plan['steps'])} steps")
            
            return optimized_plan
            
        except Exception as e:
            self.logger.error(f"AI-1: Planning failed: {e}")
            return self._generate_fallback_plan(user_instructions)
    
    async def _perform_comprehensive_search(self, instructions: str, urls: List[str], requirements: List[str]) -> Dict[str, Any]:
        """Perform comprehensive search across multiple providers."""
        try:
            search_tasks = []
            
            # Search across all providers in parallel
            for provider in SearchProvider:
                search_task = asyncio.create_task(
                    self.sub_agents[SubAgentType.SEARCH_AGENT].search_provider(
                        provider, instructions, urls, requirements
                    )
                )
                search_tasks.append((provider, search_task))
            
            # Wait for all searches to complete
            search_results = {}
            for provider, task in search_tasks:
                try:
                    result = await task
                    search_results[provider.value] = result
                except Exception as e:
                    self.logger.warning(f"Search failed for {provider.value}: {e}")
                    search_results[provider.value] = {"error": str(e)}
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive search failed: {e}")
            return {}
    
    async def _generate_execution_plan(self, instructions: str, complexity_result: Dict, 
                                     url_result: Dict, requirement_result: Dict, 
                                     search_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive execution plan using AI."""
        try:
            prompt = f"""
            Generate a comprehensive automation execution plan based on:
            
            User Instructions: {instructions}
            Task Complexity: {complexity_result['complexity']}
            Extracted URLs: {url_result['urls']}
            Requirements: {requirement_result['requirements']}
            Search Results: {json.dumps(search_results, indent=2)}
            
            Create a detailed plan with:
            1. Pre-execution steps (URL validation, requirement verification)
            2. Main execution steps (web automation, data extraction, etc.)
            3. Post-execution steps (validation, reporting, cleanup)
            4. Error handling and recovery strategies
            5. Performance optimization recommendations
            
            Return as JSON with structure:
            {{
                "task_id": "unique_id",
                "complexity": "complexity_level",
                "estimated_duration": "duration_in_seconds",
                "risk_level": "low/medium/high",
                "required_capabilities": ["capability1", "capability2"],
                "pre_execution_steps": [...],
                "main_execution_steps": [...],
                "post_execution_steps": [...],
                "error_handling": {...},
                "optimization": {...},
                "search_context": {...}
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=60)
            
            try:
                plan = json.loads(response)
                plan["generated_at"] = datetime.utcnow().isoformat()
                plan["planner_version"] = "AI-1-v1.0"
                return plan
            except json.JSONDecodeError:
                return self._generate_fallback_plan(instructions)
                
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return self._generate_fallback_plan(instructions)
    
    async def _validate_and_optimize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and optimize the execution plan."""
        try:
            # Validate plan structure
            required_fields = ["main_execution_steps", "complexity", "estimated_duration"]
            for field in required_fields:
                if field not in plan:
                    plan[field] = self._get_default_value(field)
            
            # Optimize based on complexity
            if plan["complexity"] == TaskComplexity.ULTRA_COMPLEX:
                plan = await self._optimize_for_ultra_complex(plan)
            elif plan["complexity"] == TaskComplexity.COMPLEX:
                plan = await self._optimize_for_complex(plan)
            
            # Add validation metadata
            plan["validation"] = {
                "is_valid": True,
                "warnings": [],
                "optimizations_applied": True,
                "validated_at": datetime.utcnow().isoformat()
            }
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Plan validation failed: {e}")
            plan["validation"] = {
                "is_valid": False,
                "error": str(e),
                "validated_at": datetime.utcnow().isoformat()
            }
            return plan
    
    async def _optimize_for_ultra_complex(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for ultra-complex tasks."""
        # Add parallel execution capabilities
        plan["parallel_execution"] = True
        plan["sub_agent_coordination"] = True
        plan["advanced_error_recovery"] = True
        plan["performance_monitoring"] = True
        
        # Add advanced capabilities
        plan["required_capabilities"].extend([
            "workflow_orchestration",
            "parallel_processing",
            "advanced_error_handling",
            "performance_optimization"
        ])
        
        return plan
    
    async def _optimize_for_complex(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for complex tasks."""
        plan["parallel_execution"] = False
        plan["sub_agent_coordination"] = True
        plan["advanced_error_recovery"] = True
        
        return plan
    
    def _generate_fallback_plan(self, instructions: str) -> Dict[str, Any]:
        """Generate fallback plan when AI planning fails."""
        return {
            "task_id": f"fallback_{datetime.utcnow().timestamp()}",
            "complexity": TaskComplexity.MEDIUM.value,
            "estimated_duration": 300,
            "risk_level": "medium",
            "required_capabilities": ["web_automation", "basic_interaction"],
            "main_execution_steps": [
                {
                    "step": 1,
                    "action": "navigate",
                    "description": "Navigate to target website",
                    "timeout": 30
                },
                {
                    "step": 2,
                    "action": "wait",
                    "description": "Wait for page to load",
                    "duration": 5
                }
            ],
            "pre_execution_steps": [],
            "post_execution_steps": [],
            "error_handling": {"retry_attempts": 3, "fallback_strategy": "basic"},
            "optimization": {"enabled": False},
            "search_context": {},
            "generated_at": datetime.utcnow().isoformat(),
            "planner_version": "AI-1-fallback"
        }
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing plan fields."""
        defaults = {
            "main_execution_steps": [],
            "complexity": TaskComplexity.MEDIUM.value,
            "estimated_duration": 300
        }
        return defaults.get(field, None)


class SearchSubAgent:
    """Sub-agent for comprehensive search across multiple providers."""
    
    def __init__(self, config, ai_provider: AIProvider, audit_logger: AuditLogger):
        self.config = config
        self.ai_provider = ai_provider
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize search sub-agent."""
        self.logger.info("Initializing Search Sub-Agent...")
    
    async def search_provider(self, provider: SearchProvider, instructions: str, 
                            urls: List[str], requirements: List[str]) -> Dict[str, Any]:
        """Search a specific provider for relevant information."""
        try:
            self.logger.info(f"Searching {provider.value} for: {instructions[:50]}...")
            
            # This would integrate with actual search APIs
            # For now, return mock results
            return {
                "provider": provider.value,
                "query": instructions,
                "results": [
                    {
                        "title": f"Search result from {provider.value}",
                        "url": "https://example.com",
                        "snippet": f"Relevant information for: {instructions[:100]}",
                        "relevance_score": 0.8
                    }
                ],
                "total_results": 1,
                "search_time": 1.5
            }
            
        except Exception as e:
            self.logger.error(f"Search failed for {provider.value}: {e}")
            return {"error": str(e)}


class URLExtractionSubAgent:
    """Sub-agent for extracting URLs and requirements from instructions."""
    
    def __init__(self, config, ai_provider: AIProvider, audit_logger: AuditLogger):
        self.config = config
        self.ai_provider = ai_provider
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize URL extraction sub-agent."""
        self.logger.info("Initializing URL Extraction Sub-Agent...")
    
    async def extract_urls_and_requirements(self, instructions: str) -> Dict[str, Any]:
        """Extract URLs and requirements from user instructions."""
        try:
            prompt = f"""
            Extract URLs and requirements from this instruction: "{instructions}"
            
            Return as JSON:
            {{
                "urls": ["url1", "url2"],
                "requirements": ["req1", "req2"],
                "data_needed": ["data1", "data2"],
                "actions_required": ["action1", "action2"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                return self._extract_fallback(instructions)
                
        except Exception as e:
            self.logger.error(f"URL extraction failed: {e}")
            return self._extract_fallback(instructions)
    
    def _extract_fallback(self, instructions: str) -> Dict[str, Any]:
        """Fallback URL extraction using regex."""
        import re
        
        # Extract URLs using regex
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, instructions)
        
        return {
            "urls": urls,
            "requirements": [],
            "data_needed": [],
            "actions_required": []
        }


class ComplexityAnalysisSubAgent:
    """Sub-agent for analyzing task complexity."""
    
    def __init__(self, config, ai_provider: AIProvider, audit_logger: AuditLogger):
        self.config = config
        self.ai_provider = ai_provider
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize complexity analysis sub-agent."""
        self.logger.info("Initializing Complexity Analysis Sub-Agent...")
    
    async def analyze_complexity(self, instructions: str) -> Dict[str, Any]:
        """Analyze the complexity of the automation task."""
        try:
            prompt = f"""
            Analyze the complexity of this automation task: "{instructions}"
            
            Consider factors like:
            - Number of steps required
            - Type of interactions needed
            - Data processing requirements
            - Error handling complexity
            - Integration requirements
            
            Return as JSON:
            {{
                "complexity": "simple/medium/complex/ultra_complex",
                "reasoning": "explanation",
                "estimated_steps": 10,
                "risk_factors": ["factor1", "factor2"],
                "special_requirements": ["req1", "req2"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                return self._analyze_fallback(instructions)
                
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return self._analyze_fallback(instructions)
    
    def _analyze_fallback(self, instructions: str) -> Dict[str, Any]:
        """Fallback complexity analysis."""
        instructions_lower = instructions.lower()
        
        if any(word in instructions_lower for word in ["complex", "multiple", "parallel", "workflow"]):
            complexity = TaskComplexity.COMPLEX
        elif any(word in instructions_lower for word in ["simple", "basic", "click", "type"]):
            complexity = TaskComplexity.SIMPLE
        else:
            complexity = TaskComplexity.MEDIUM
        
        return {
            "complexity": complexity.value,
            "reasoning": "Fallback analysis based on keywords",
            "estimated_steps": 5,
            "risk_factors": [],
            "special_requirements": []
        }


class RequirementAnalysisSubAgent:
    """Sub-agent for analyzing requirements."""
    
    def __init__(self, config, ai_provider: AIProvider, audit_logger: AuditLogger):
        self.config = config
        self.ai_provider = ai_provider
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize requirement analysis sub-agent."""
        self.logger.info("Initializing Requirement Analysis Sub-Agent...")
    
    async def analyze_requirements(self, instructions: str) -> Dict[str, Any]:
        """Analyze requirements from user instructions."""
        try:
            prompt = f"""
            Analyze requirements from this instruction: "{instructions}"
            
            Identify:
            - Functional requirements
            - Technical requirements
            - Data requirements
            - Performance requirements
            - Security requirements
            
            Return as JSON:
            {{
                "requirements": ["req1", "req2"],
                "functional_requirements": ["func1", "func2"],
                "technical_requirements": ["tech1", "tech2"],
                "data_requirements": ["data1", "data2"],
                "performance_requirements": ["perf1", "perf2"],
                "security_requirements": ["sec1", "sec2"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                return self._analyze_fallback(instructions)
                
        except Exception as e:
            self.logger.error(f"Requirement analysis failed: {e}")
            return self._analyze_fallback(instructions)
    
    def _analyze_fallback(self, instructions: str) -> Dict[str, Any]:
        """Fallback requirement analysis."""
        return {
            "requirements": ["basic_automation"],
            "functional_requirements": ["web_interaction"],
            "technical_requirements": ["browser_automation"],
            "data_requirements": [],
            "performance_requirements": ["reasonable_speed"],
            "security_requirements": []
        }