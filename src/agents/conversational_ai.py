"""
Advanced Conversational AI Agent
================================

Handles reasoning, follow-up questions, and human handoff during automation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture


class ConversationalAI:
    """Advanced conversational AI for automation reasoning and human interaction."""
    
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
        
        # Conversation context
        self.conversation_history = []
        self.current_task = None
        self.automation_state = {}
        self.human_handoff_required = False
        
    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input and generate intelligent response."""
        try:
            self.logger.info(f"Processing user input: {user_input[:100]}...")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Analyze input intent
            intent_analysis = await self._analyze_intent(user_input, context)
            
            # Generate response based on intent
            if intent_analysis["intent"] == "automation_request":
                response = await self._handle_automation_request(user_input, context)
            elif intent_analysis["intent"] == "follow_up_question":
                response = await self._handle_follow_up_question(user_input, context)
            elif intent_analysis["intent"] == "human_handoff":
                response = await self._handle_human_handoff(user_input, context)
            elif intent_analysis["intent"] == "clarification":
                response = await self._handle_clarification(user_input, context)
            else:
                response = await self._generate_general_response(user_input, context)
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response["message"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            return {
                "message": "I encountered an error processing your request. Please try again.",
                "type": "error",
                "error": str(e)
            }
            
    async def _analyze_intent(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user input intent using AI."""
        try:
            prompt = f"""
            Analyze this user input: "{user_input}"
            
            Context: {context or {}}
            
            Determine the intent from these categories:
            1. automation_request: User wants to perform automation
            2. follow_up_question: User is asking about previous automation
            3. human_handoff: User wants to take control or needs human intervention
            4. clarification: User needs more information or clarification
            5. general_chat: General conversation
            
            Return as JSON with intent and confidence score.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                intent_data = json.loads(response)
                return intent_data
            except:
                # Fallback intent detection
                input_lower = user_input.lower()
                if any(word in input_lower for word in ["automate", "click", "login", "fill", "submit"]):
                    return {"intent": "automation_request", "confidence": 0.8}
                elif any(word in input_lower for word in ["how", "what", "why", "when"]):
                    return {"intent": "follow_up_question", "confidence": 0.7}
                elif any(word in input_lower for word in ["help", "manual", "human"]):
                    return {"intent": "human_handoff", "confidence": 0.9}
                else:
                    return {"intent": "general_chat", "confidence": 0.6}
                    
        except Exception as e:
            self.logger.warning(f"Intent analysis failed: {e}")
            return {"intent": "general_chat", "confidence": 0.5}
            
    async def _handle_automation_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle automation requests."""
        try:
            # Analyze automation complexity
            complexity_analysis = await self._analyze_automation_complexity(user_input)
            
            if complexity_analysis["complexity"] == "simple":
                return await self._handle_simple_automation(user_input, context)
            elif complexity_analysis["complexity"] == "complex":
                return await self._handle_complex_automation(user_input, context)
            else:
                return await self._handle_ultra_complex_automation(user_input, context)
                
        except Exception as e:
            return {
                "message": f"I'm having trouble understanding your automation request. Could you please provide more details?",
                "type": "clarification_needed",
                "error": str(e)
            }
            
    async def _analyze_automation_complexity(self, user_input: str) -> Dict[str, Any]:
        """Analyze automation complexity using AI."""
        try:
            prompt = f"""
            Analyze the complexity of this automation request: "{user_input}"
            
            Categorize as:
            - simple: Basic form filling, clicking, navigation
            - complex: Multi-step workflows, data extraction, API calls
            - ultra_complex: Multi-site automation, complex decision making, parallel tasks
            
            Return as JSON with complexity and reasoning.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                complexity_data = json.loads(response)
                return complexity_data
            except:
                # Fallback complexity detection
                input_lower = user_input.lower()
                if any(word in input_lower for word in ["multiple", "parallel", "complex", "workflow"]):
                    return {"complexity": "ultra_complex", "reasoning": "Contains complex keywords"}
                elif any(word in input_lower for word in ["extract", "api", "data", "process"]):
                    return {"complexity": "complex", "reasoning": "Contains data processing keywords"}
                else:
                    return {"complexity": "simple", "reasoning": "Basic automation keywords"}
                    
        except Exception as e:
            return {"complexity": "simple", "reasoning": "Fallback classification"}
            
    async def _handle_simple_automation(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle simple automation requests."""
        return {
            "message": f"I'll help you with this automation: '{user_input}'. Let me start by analyzing the requirements and creating an execution plan.",
            "type": "automation_started",
            "complexity": "simple",
            "next_steps": [
                "Analyze requirements",
                "Generate automation plan",
                "Execute automation",
                "Provide results"
            ]
        }
        
    async def _handle_complex_automation(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle complex automation requests."""
        return {
            "message": f"This is a complex automation task: '{user_input}'. I'll need to coordinate multiple sub-agents and gather information from various sources. Let me start the parallel execution.",
            "type": "complex_automation_started",
            "complexity": "complex",
            "parallel_tasks": [
                "Web search for relevant information",
                "DOM analysis of target websites",
                "Data extraction planning",
                "Workflow orchestration"
            ]
        }
        
    async def _handle_ultra_complex_automation(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle ultra-complex automation requests."""
        return {
            "message": f"This is an ultra-complex automation task: '{user_input}'. I'll activate all available sub-agents and use advanced AI reasoning to coordinate the execution. This may require human intervention at certain points.",
            "type": "ultra_complex_automation_started",
            "complexity": "ultra_complex",
            "sub_agents": [
                "Planner Agent (AI-1)",
                "DOM Analysis Agent (AI-2)",
                "Conversational Agent (AI-3)",
                "Parallel Executor",
                "Sector Specialists"
            ],
            "human_handoff_points": [
                "Complex decision making",
                "Security verification",
                "Data validation",
                "Error recovery"
            ]
        }
        
    async def _handle_follow_up_question(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle follow-up questions about automation."""
        try:
            prompt = f"""
            User is asking a follow-up question: "{user_input}"
            
            Conversation history: {self.conversation_history[-5:] if self.conversation_history else []}
            Current automation state: {self.automation_state}
            
            Provide a helpful and contextual response. If the question is about automation progress, include current status.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            return {
                "message": response,
                "type": "follow_up_response",
                "context": self.automation_state
            }
            
        except Exception as e:
            return {
                "message": "I'm having trouble accessing the context of our conversation. Could you please provide more details about what you're asking?",
                "type": "clarification_needed"
            }
            
    async def _handle_human_handoff(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle human handoff requests."""
        self.human_handoff_required = True
        
        return {
            "message": "I understand you need human intervention. I'll pause the automation and wait for your input. You can take control of the browser and perform the required actions. When you're ready, just let me know and I'll resume the automation.",
            "type": "human_handoff",
            "automation_paused": True,
            "current_state": self.automation_state,
            "resume_instructions": "Say 'resume automation' when ready to continue"
        }
        
    async def _handle_clarification(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle clarification requests."""
        try:
            prompt = f"""
            User needs clarification: "{user_input}"
            
            Provide a clear and helpful response that addresses their question.
            If it's about automation, explain the process and what to expect.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            return {
                "message": response,
                "type": "clarification_response"
            }
            
        except Exception as e:
            return {
                "message": "I'm here to help! Could you please rephrase your question or provide more context?",
                "type": "clarification_response"
            }
            
    async def _generate_general_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate general conversational responses."""
        try:
            prompt = f"""
            User said: "{user_input}"
            
            Provide a helpful and conversational response. If they seem interested in automation, offer to help with automation tasks.
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            return {
                "message": response,
                "type": "general_response"
            }
            
        except Exception as e:
            return {
                "message": "Hello! I'm here to help you with automation tasks. What would you like to automate today?",
                "type": "general_response"
            }
            
    async def update_automation_state(self, state: Dict[str, Any]):
        """Update the current automation state."""
        self.automation_state.update(state)
        
    async def resume_automation(self) -> Dict[str, Any]:
        """Resume automation after human handoff."""
        if self.human_handoff_required:
            self.human_handoff_required = False
            return {
                "message": "Great! I'm resuming the automation. Let me continue from where we left off.",
                "type": "automation_resumed",
                "state": self.automation_state
            }
        else:
            return {
                "message": "No automation was paused. What would you like to automate?",
                "type": "no_automation_paused"
            }
            
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        try:
            if not self.conversation_history:
                return {"summary": "No conversation history available."}
                
            prompt = f"""
            Summarize this conversation:
            {self.conversation_history}
            
            Focus on:
            1. Main topics discussed
            2. Automation tasks requested
            3. Current status
            4. Key decisions made
            """
            
            summary = await self.ai_provider.generate_response(prompt)
            
            return {
                "summary": summary,
                "total_messages": len(self.conversation_history),
                "automation_state": self.automation_state,
                "human_handoff_required": self.human_handoff_required
            }
            
        except Exception as e:
            return {
                "summary": "Unable to generate summary due to error.",
                "error": str(e)
            }
            
    async def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.automation_state = {}
        self.human_handoff_required = False
        
        return {
            "message": "Conversation history cleared. Ready for new automation tasks!",
            "type": "conversation_cleared"
        }