"""
Advanced Multi-Agent Orchestrator
=================================

Coordinates multiple AI agents for ultra-complex automation tasks.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import with fallbacks for missing dependencies
try:
    from ..core.ai_provider import AIProvider
except ImportError:
    class AIProvider:
        def __init__(self, *args, **kwargs): pass
        def get_response(self, *args, **kwargs): return {"response": "fallback"}

try:
    from ..agents.planner import PlannerAgent
    from ..agents.executor import ExecutionAgent  
    from ..agents.conversational_ai import ConversationalAI
    from ..agents.parallel_executor import ParallelExecutor
    from ..agents.sector_specialists import SectorManager
except ImportError:
    class PlannerAgent:
        def __init__(self, *args, **kwargs): pass
        def plan(self, *args, **kwargs): return []
    class ExecutionAgent:
        def __init__(self, *args, **kwargs): pass
        def execute(self, *args, **kwargs): return True
    class ConversationalAI:
        def __init__(self, *args, **kwargs): pass
    class ParallelExecutor:
        def __init__(self, *args, **kwargs): pass
    class SectorManager:
        def __init__(self, *args, **kwargs): pass

try:
    from ..utils.media_capture import MediaCapture
    from ..utils.code_generator import CodeGenerator
    from ..utils.selector_drift import SelectorDriftDetector
except ImportError:
    class MediaCapture:
        def __init__(self, *args, **kwargs): pass
        def capture(self, *args, **kwargs): return None
    class CodeGenerator:
        def __init__(self, *args, **kwargs): pass
        def generate(self, *args, **kwargs): return ""
    class SelectorDriftDetector:
        def __init__(self, *args, **kwargs): pass
        def detect(self, *args, **kwargs): return False


class AdvancedOrchestrator:
    """Advanced orchestrator for ultra-complex automation tasks."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        # Initialize media capture with correct path
        if hasattr(config, 'database') and hasattr(config.database, 'media_path'):
            media_path = config.database.media_path
        else:
            media_path = 'data/media'
        self.media_capture = MediaCapture(media_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        from ..core.database import DatabaseManager
        from ..core.vector_store import VectorStore
        from ..core.audit import AuditLogger
        
        self.database = DatabaseManager(config.database)
        self.vector_store = VectorStore(config.database)
        self.audit_logger = AuditLogger(config)
        
        # Initialize all agents
        self.planner_agent = PlannerAgent(config.ai, self.vector_store, self.audit_logger)
        self.execution_agent = ExecutionAgent(config.automation, self.media_capture, SelectorDriftDetector(config.automation), self.audit_logger)
        self.conversational_ai = ConversationalAI(config, ai_provider)
        self.parallel_executor = ParallelExecutor(config, ai_provider)
        self.sector_manager = SectorManager(config, ai_provider)
        self.code_generator = CodeGenerator(config, ai_provider)
        
        # Task state management
        self.current_task = None
        self.task_history = []
        self.automation_state = {}
        
    async def execute_ultra_complex_automation(self, user_request: str) -> Dict[str, Any]:
        """Execute ultra-complex automation with multiple AI agents."""
        try:
            self.logger.info(f"Starting ultra-complex automation: {user_request}")
            
            # Step 1: AI-1 (Planner Agent) - Analyze and plan
            planning_result = await self._execute_planning_phase(user_request)
            
            # Step 2: Parallel sub-agents - Gather information
            parallel_results = await self._execute_parallel_phase(planning_result)
            
            # Step 3: AI-2 (DOM Analysis) - Analyze target websites
            dom_analysis = await self._execute_dom_analysis_phase(planning_result, parallel_results)
            
            # Step 4: AI-3 (Conversational) - Coordinate and reason
            coordination_result = await self._execute_coordination_phase(planning_result, parallel_results, dom_analysis)
            
            # Step 5: Execute automation
            execution_result = await self._execute_automation_phase(coordination_result)
            
            # Step 6: Generate code and report
            final_result = await self._generate_final_report(execution_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Ultra-complex automation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "user_request": user_request,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _execute_planning_phase(self, user_request: str) -> Dict[str, Any]:
        """Execute planning phase with AI-1 (Planner Agent)."""
        try:
            self.logger.info("Starting planning phase with AI-1")
            
            # Analyze task complexity and requirements
            complexity_analysis = await self.planner_agent.analyze_task_complexity(user_request)
            
            # Generate execution plan
            execution_plan = await self.planner_agent.generate_execution_plan(user_request, complexity_analysis)
            
            # Identify required sub-agents
            required_agents = await self.planner_agent.identify_required_agents(execution_plan)
            
            # Detect sector and get specialist
            sector = await self.sector_manager.detect_sector(user_request, "")
            sector_specialist = await self.sector_manager.get_sector_specialist(sector)
            
            return {
                "phase": "planning",
                "complexity_analysis": complexity_analysis,
                "execution_plan": execution_plan,
                "required_agents": required_agents,
                "sector": sector,
                "sector_specialist": sector_specialist.sector_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Planning phase failed: {e}")
            # Return a fallback planning result instead of raising
            return {
                "phase": "planning",
                "complexity_analysis": {
                    "complexity_level": "medium",
                    "estimated_steps": 5,
                    "data_sources": [],
                    "user_interactions": [],
                    "challenges": ["Planning failed"],
                    "success_probability": 0.7
                },
                "execution_plan": {
                    "execution_steps": [
                        {
                            "step": 1,
                            "action": "analyze_requirements",
                            "description": "Analyze user requirements",
                            "duration": 30
                        },
                        {
                            "step": 2,
                            "action": "execute_automation",
                            "description": "Execute automation task",
                            "duration": 120
                        }
                    ],
                    "estimated_duration": 150,
                    "success_criteria": ["task_completed"]
                },
                "required_agents": {
                    "web_search": True,
                    "dom_analysis": True,
                    "data_extraction": False,
                    "api_calls": False,
                    "file_processing": False
                },
                "sector": "general",
                "sector_specialist": "general",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _execute_parallel_phase(self, planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel phase with multiple sub-agents."""
        try:
            self.logger.info("Starting parallel execution phase")
            
            # Extract URLs from user request and planning
            urls = await self._extract_urls_from_planning(planning_result)
            
            # Create parallel tasks
            parallel_tasks = []
            
            # Web search tasks
            if planning_result.get("required_agents", {}).get("web_search", False):
                search_queries = await self._generate_search_queries(planning_result)
                for query in search_queries:
                    parallel_tasks.append({
                        "type": "web_search",
                        "query": query,
                        "providers": ["google", "bing", "duckduckgo", "github", "stackoverflow"]
                    })
            
            # DOM analysis tasks
            for url in urls:
                parallel_tasks.append({
                    "type": "dom_analysis",
                    "url": url
                })
            
            # Data extraction tasks
            if planning_result.get("required_agents", {}).get("data_extraction", False):
                parallel_tasks.append({
                    "type": "data_extraction",
                    "source": "web",
                    "selectors": planning_result.get("execution_plan", {}).get("selectors", {})
                })
            
            # Execute parallel tasks
            parallel_results = await self.parallel_executor.execute_parallel_tasks(parallel_tasks)
            
            return {
                "phase": "parallel_execution",
                "tasks_executed": len(parallel_tasks),
                "results": parallel_results,
                "urls_analyzed": urls,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Parallel phase failed: {e}")
            raise
            
    async def _execute_dom_analysis_phase(self, planning_result: Dict[str, Any], parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DOM analysis phase with AI-2."""
        try:
            self.logger.info("Starting DOM analysis phase with AI-2")
            
            # Extract URLs from parallel results
            urls = []
            for result in parallel_results.get("results", []):
                if result.get("result", {}).get("type") == "dom_analysis":
                    urls.append(result.get("result", {}).get("url"))
            
            # Perform detailed DOM analysis for each URL
            dom_analyses = []
            for url in urls:
                analysis = await self._perform_detailed_dom_analysis(url)
                dom_analyses.append(analysis)
            
            # Generate automation strategies
            automation_strategies = await self._generate_automation_strategies(dom_analyses, planning_result)
            
            return {
                "phase": "dom_analysis",
                "urls_analyzed": urls,
                "dom_analyses": dom_analyses,
                "automation_strategies": automation_strategies,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DOM analysis phase failed: {e}")
            # Return a fallback DOM analysis result
            return {
                "phase": "dom_analysis",
                "urls_analyzed": [],
                "dom_analyses": [],
                "automation_strategies": {
                    "strategy": "basic_automation",
                    "steps": ["navigate", "interact", "extract"]
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _execute_coordination_phase(self, planning_result: Dict[str, Any], parallel_results: Dict[str, Any], dom_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordination phase with AI-3 (Conversational Agent)."""
        try:
            self.logger.info("Starting coordination phase with AI-3")
            
            # Update conversational AI state
            await self.conversational_ai.update_automation_state({
                "planning_result": planning_result,
                "parallel_results": parallel_results,
                "dom_analysis": dom_analysis
            })
            
            # Generate coordinated execution plan
            coordinated_plan = await self._generate_coordinated_plan(planning_result, parallel_results, dom_analysis)
            
            # Validate plan with conversational AI
            validation_result = await self.conversational_ai.process_user_input(
                f"Validate this automation plan: {coordinated_plan}",
                {"phase": "coordination"}
            )
            
            return {
                "phase": "coordination",
                "coordinated_plan": coordinated_plan,
                "validation_result": validation_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Coordination phase failed: {e}")
            raise
            
    async def _execute_automation_phase(self, coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual automation."""
        try:
            self.logger.info("Starting automation execution phase")
            
            coordinated_plan = coordination_result.get("coordinated_plan", {})
            
            # Execute automation using execution agent
            execution_result = await self.execution_agent.execute_intelligent_automation(
                coordinated_plan.get("instructions", ""),
                coordinated_plan.get("url", "")
            )
            
            # Capture screenshots and videos
            media_captures = await self._capture_automation_media(execution_result)
            
            return {
                "phase": "automation_execution",
                "execution_result": execution_result,
                "media_captures": media_captures,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Automation phase failed: {e}")
            raise
            
    async def _generate_final_report(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report with code and documentation."""
        try:
            self.logger.info("Generating final report")
            
            # Generate code (Playwright/Selenium/Cypress)
            generated_code = await self.code_generator.generate_automation_code(execution_result)
            
            # Generate detailed report
            detailed_report = await self._generate_detailed_report(execution_result)
            
            # Update conversation summary
            conversation_summary = await self.conversational_ai.get_conversation_summary()
            
            return {
                "status": "completed",
                "execution_result": execution_result,
                "generated_code": generated_code,
                "detailed_report": detailed_report,
                "conversation_summary": conversation_summary,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Final report generation failed: {e}")
            raise
            
    async def _extract_urls_from_planning(self, planning_result: Dict[str, Any]) -> List[str]:
        """Extract URLs from planning result."""
        urls = []
        
        # Extract from user request
        user_request = planning_result.get("user_request", "")
        # Basic URL extraction - in production, use regex
        if "http" in user_request:
            words = user_request.split()
            for word in words:
                if word.startswith("http"):
                    urls.append(word)
                    
        return urls
        
    async def _generate_search_queries(self, planning_result: Dict[str, Any]) -> List[str]:
        """Generate search queries based on planning result."""
        queries = []
        
        complexity_analysis = planning_result.get("complexity_analysis", {})
        execution_plan = planning_result.get("execution_plan", {})
        
        # Generate queries based on task requirements
        if "login" in str(execution_plan).lower():
            queries.append("website login automation best practices")
            
        if "form" in str(execution_plan).lower():
            queries.append("web form automation techniques")
            
        if "data" in str(execution_plan).lower():
            queries.append("web scraping data extraction methods")
            
        return queries
        
    async def _perform_detailed_dom_analysis(self, url: str) -> Dict[str, Any]:
        """Perform detailed DOM analysis for a URL."""
        try:
            # Use AI to analyze DOM structure
            prompt = f"""
            Perform detailed DOM analysis for: {url}
            
            Analyze:
            1. Page structure and layout
            2. Interactive elements (buttons, links, forms)
            3. Input fields and their purposes
            4. Navigation patterns
            5. Potential automation targets
            6. Security measures (CAPTCHA, rate limiting)
            7. Accessibility features
            
            Return as structured JSON.
            """
            
            analysis = await self.ai_provider.generate_response(prompt)
            
            return {
                "url": url,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _generate_automation_strategies(self, dom_analyses: List[Dict[str, Any]], planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automation strategies based on DOM analyses."""
        try:
            strategies = []
            
            for analysis in dom_analyses:
                url = analysis.get("url", "")
                dom_analysis = analysis.get("analysis", "")
                
                prompt = f"""
                Generate automation strategy for: {url}
                
                DOM Analysis: {dom_analysis}
                Planning Result: {planning_result}
                
                Provide:
                1. Element selection strategies
                2. Wait conditions
                3. Error handling approaches
                4. Fallback mechanisms
                5. Performance optimizations
                
                Return as structured JSON.
                """
                
                strategy = await self.ai_provider.generate_response(prompt)
                strategies.append({
                    "url": url,
                    "strategy": strategy
                })
                
            return {
                "strategies": strategies,
                "total_strategies": len(strategies)
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _generate_coordinated_plan(self, planning_result: Dict[str, Any], parallel_results: Dict[str, Any], dom_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coordinated execution plan."""
        try:
            prompt = f"""
            Generate coordinated automation plan:
            
            Planning Result: {planning_result}
            Parallel Results: {parallel_results}
            DOM Analysis: {dom_analysis}
            
            Create a detailed, step-by-step execution plan that coordinates all findings.
            Include:
            1. Execution sequence
            2. Error handling
            3. Human handoff points
            4. Success criteria
            5. Rollback procedures
            
            Return as structured JSON.
            """
            
            coordinated_plan = await self.ai_provider.generate_response(prompt)
            
            return {
                "plan": coordinated_plan,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _capture_automation_media(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Capture screenshots and videos during automation."""
        try:
            media_captures = {
                "screenshots": [],
                "videos": [],
                "logs": []
            }
            
            # Capture screenshots
            if execution_result.get("screenshots"):
                media_captures["screenshots"] = execution_result["screenshots"]
                
            # Capture videos (if available)
            if execution_result.get("videos"):
                media_captures["videos"] = execution_result["videos"]
                
            # Capture logs
            media_captures["logs"] = execution_result.get("logs", [])
            
            return media_captures
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _generate_detailed_report(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed automation report."""
        try:
            prompt = f"""
            Generate detailed automation report:
            
            Execution Result: {execution_result}
            
            Include:
            1. Executive summary
            2. Technical details
            3. Performance metrics
            4. Issues encountered
            5. Recommendations
            6. Next steps
            
            Return as structured JSON.
            """
            
            report = await self.ai_provider.generate_response(prompt)
            
            return {
                "report": report,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def handle_human_handoff(self, user_input: str) -> Dict[str, Any]:
        """Handle human handoff during automation."""
        return await self.conversational_ai.process_user_input(user_input, {"handoff": True})
        
    async def resume_automation(self) -> Dict[str, Any]:
        """Resume automation after human handoff."""
        return await self.conversational_ai.resume_automation()
        
    async def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        return {
            "current_task": self.current_task,
            "automation_state": self.automation_state,
            "conversation_summary": await self.conversational_ai.get_conversation_summary()
        }