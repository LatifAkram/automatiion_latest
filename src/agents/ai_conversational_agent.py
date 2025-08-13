"""
AI-3: Conversational Agent
==========================

Acts as conversational, follow-up, and reasoning agent.
Handles human handoff, reasoning, and conversational AI like Cursor AI.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture


class ConversationState(Enum):
    """Conversation states."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING_FOR_HUMAN = "waiting_for_human"
    HANDOFF_COMPLETE = "handoff_complete"
    FOLLOW_UP = "follow_up"


class HandoffReason(Enum):
    """Reasons for human handoff."""
    COMPLEX_DECISION = "complex_decision"
    ERROR_RECOVERY = "error_recovery"
    USER_REQUEST = "user_request"
    SECURITY_CONCERN = "security_concern"
    COMPLIANCE_ISSUE = "compliance_issue"
    TECHNICAL_LIMITATION = "technical_limitation"


class AIConversationalAgent:
    """AI-3: Conversational Agent for reasoning, follow-up, and human handoff."""
    
    def __init__(self, config, ai_provider: AIProvider, media_capture: MediaCapture):
        self.config = config
        self.ai_provider = ai_provider
        self.media_capture = media_capture
        self.logger = logging.getLogger(__name__)
        
        # Conversation management
        self.conversation_history = []
        self.current_state = ConversationState.IDLE
        self.active_session = None
        self.handoff_reasons = []
        
        # Reasoning and context
        self.context_memory = {}
        self.reasoning_chain = []
        self.follow_up_questions = []
        
        # Human handoff management
        self.handoff_sessions = {}
        self.human_input_queue = []
        
    async def initialize(self):
        """Initialize the conversational agent."""
        self.logger.info("Initializing AI-3 Conversational Agent...")
        self.logger.info("AI-3 Conversational Agent initialized successfully")
    
    async def process_conversation(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user conversation input with reasoning and follow-up capabilities.
        
        Args:
            user_input: User's message
            context: Conversation context
            
        Returns:
            Response with reasoning and potential handoff
        """
        try:
            self.logger.info(f"AI-3: Processing conversation: {user_input[:100]}...")
            
            # Step 1: Analyze user intent and context
            intent_analysis = await self._analyze_user_intent(user_input, context)
            
            # Step 2: Update conversation state
            await self._update_conversation_state(intent_analysis)
            
            # Step 3: Generate reasoning chain
            reasoning = await self._generate_reasoning_chain(user_input, context, intent_analysis)
            
            # Step 4: Check if human handoff is needed
            handoff_decision = await self._evaluate_handoff_need(user_input, context, reasoning)
            
            # Step 5: Generate response
            if handoff_decision["handoff_required"]:
                response = await self._initiate_human_handoff(handoff_decision, context)
            else:
                response = await self._generate_ai_response(user_input, context, reasoning)
            
            # Step 6: Generate follow-up questions
            follow_ups = await self._generate_follow_up_questions(user_input, context, response)
            
            # Step 7: Update conversation history
            await self._update_conversation_history(user_input, response, context)
            
            # Create comprehensive response
            conversation_response = {
                "response": response["message"],
                "reasoning": reasoning["chain"],
                "confidence": reasoning["confidence"],
                "follow_up_questions": follow_ups,
                "handoff_required": handoff_decision["handoff_required"],
                "handoff_reason": handoff_decision.get("reason"),
                "conversation_state": self.current_state.value,
                "context_updated": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"AI-3: Conversation processed - Handoff: {handoff_decision['handoff_required']}")
            return conversation_response
            
        except Exception as e:
            self.logger.error(f"AI-3: Conversation processing failed: {e}")
            return self._generate_error_response(str(e))
    
    async def _analyze_user_intent(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user intent using multiple AI providers."""
        try:
            prompt = f"""
            Analyze the user's intent and context:
            
            User Input: "{user_input}"
            Context: {json.dumps(context or {}, indent=2)}
            Conversation History: {len(self.conversation_history)} messages
            
            Determine:
            1. Primary intent (question, request, clarification, handoff, etc.)
            2. Emotional state (frustrated, satisfied, confused, etc.)
            3. Urgency level (low, medium, high)
            4. Complexity level (simple, medium, complex)
            5. Whether human handoff might be needed
            
            Return as JSON:
            {{
                "primary_intent": "intent_type",
                "emotional_state": "emotion",
                "urgency": "urgency_level",
                "complexity": "complexity_level",
                "handoff_likelihood": 0.0-1.0,
                "confidence": 0.0-1.0,
                "keywords": ["keyword1", "keyword2"],
                "suggested_actions": ["action1", "action2"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                intent = json.loads(response)
                intent["analyzed_at"] = datetime.utcnow().isoformat()
                return intent
            except json.JSONDecodeError:
                return self._analyze_intent_fallback(user_input)
                
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}")
            return self._analyze_intent_fallback(user_input)
    
    def _analyze_intent_fallback(self, user_input: str) -> Dict[str, Any]:
        """Fallback intent analysis."""
        input_lower = user_input.lower()
        
        # Simple keyword-based analysis
        if any(word in input_lower for word in ["help", "problem", "error", "issue"]):
            intent = "help_request"
            handoff_likelihood = 0.3
        elif any(word in input_lower for word in ["human", "person", "agent", "representative"]):
            intent = "handoff_request"
            handoff_likelihood = 0.8
        elif any(word in input_lower for word in ["question", "what", "how", "why"]):
            intent = "question"
            handoff_likelihood = 0.1
        else:
            intent = "general_conversation"
            handoff_likelihood = 0.1
        
        return {
            "primary_intent": intent,
            "emotional_state": "neutral",
            "urgency": "medium",
            "complexity": "medium",
            "handoff_likelihood": handoff_likelihood,
            "confidence": 0.7,
            "keywords": [],
            "suggested_actions": ["respond", "clarify"],
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    async def _update_conversation_state(self, intent_analysis: Dict[str, Any]):
        """Update conversation state based on intent analysis."""
        if intent_analysis["handoff_likelihood"] > 0.7:
            self.current_state = ConversationState.WAITING_FOR_HUMAN
        elif intent_analysis["primary_intent"] == "follow_up":
            self.current_state = ConversationState.FOLLOW_UP
        else:
            self.current_state = ConversationState.ACTIVE
    
    async def _generate_reasoning_chain(self, user_input: str, context: Dict[str, Any], 
                                      intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning chain for the conversation."""
        try:
            prompt = f"""
            Generate a reasoning chain for this conversation:
            
            User Input: "{user_input}"
            Intent: {intent_analysis['primary_intent']}
            Context: {json.dumps(context or {}, indent=2)}
            History: {len(self.conversation_history)} previous messages
            
            Provide step-by-step reasoning:
            1. What is the user asking/requesting?
            2. What context is relevant?
            3. What are the possible responses?
            4. What are the implications?
            5. What should be the next action?
            
            Return as JSON:
            {{
                "chain": [
                    "Step 1: Understanding the request",
                    "Step 2: Analyzing context",
                    "Step 3: Considering options",
                    "Step 4: Making decision"
                ],
                "confidence": 0.0-1.0,
                "key_insights": ["insight1", "insight2"],
                "recommended_action": "action_description"
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                reasoning = json.loads(response)
                reasoning["generated_at"] = datetime.utcnow().isoformat()
                
                # Store reasoning chain
                self.reasoning_chain.append(reasoning)
                
                return reasoning
            except json.JSONDecodeError:
                return self._generate_reasoning_fallback(user_input, intent_analysis)
                
        except Exception as e:
            self.logger.error(f"Reasoning generation failed: {e}")
            return self._generate_reasoning_fallback(user_input, intent_analysis)
    
    def _generate_reasoning_fallback(self, user_input: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback reasoning generation."""
        return {
            "chain": [
                f"Step 1: User intent is {intent_analysis['primary_intent']}",
                "Step 2: Analyzing user input for key information",
                "Step 3: Considering appropriate response",
                "Step 4: Preparing helpful response"
            ],
            "confidence": 0.6,
            "key_insights": ["User needs assistance", "Context is important"],
            "recommended_action": "Provide helpful response",
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _evaluate_handoff_need(self, user_input: str, context: Dict[str, Any], 
                                   reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if human handoff is needed."""
        try:
            prompt = f"""
            Evaluate if human handoff is needed:
            
            User Input: "{user_input}"
            Reasoning: {json.dumps(reasoning, indent=2)}
            Context: {json.dumps(context or {}, indent=2)}
            
            Consider factors:
            1. Complexity of the request
            2. User's emotional state
            3. Technical limitations
            4. Security concerns
            5. Compliance requirements
            6. User explicitly requesting human
            
            Return as JSON:
            {{
                "handoff_required": true/false,
                "reason": "reason_for_handoff",
                "urgency": "low/medium/high",
                "confidence": 0.0-1.0,
                "alternative_suggestions": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                handoff_decision = json.loads(response)
                handoff_decision["evaluated_at"] = datetime.utcnow().isoformat()
                return handoff_decision
            except json.JSONDecodeError:
                return self._evaluate_handoff_fallback(user_input)
                
        except Exception as e:
            self.logger.error(f"Handoff evaluation failed: {e}")
            return self._evaluate_handoff_fallback(user_input)
    
    def _evaluate_handoff_fallback(self, user_input: str) -> Dict[str, Any]:
        """Fallback handoff evaluation."""
        input_lower = user_input.lower()
        
        # Simple keyword-based evaluation
        handoff_keywords = ["human", "person", "agent", "representative", "speak to someone"]
        if any(keyword in input_lower for keyword in handoff_keywords):
            return {
                "handoff_required": True,
                "reason": HandoffReason.USER_REQUEST.value,
                "urgency": "medium",
                "confidence": 0.8,
                "alternative_suggestions": ["I can help you with that", "Let me try to assist"],
                "evaluated_at": datetime.utcnow().isoformat()
            }
        
        return {
            "handoff_required": False,
            "reason": None,
            "urgency": "low",
            "confidence": 0.7,
            "alternative_suggestions": [],
            "evaluated_at": datetime.utcnow().isoformat()
        }
    
    async def _initiate_human_handoff(self, handoff_decision: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate human handoff process."""
        try:
            self.logger.info(f"AI-3: Initiating human handoff - Reason: {handoff_decision['reason']}")
            
            # Create handoff session
            handoff_id = f"handoff_{datetime.utcnow().timestamp()}"
            handoff_session = {
                "id": handoff_id,
                "reason": handoff_decision["reason"],
                "urgency": handoff_decision["urgency"],
                "context": context,
                "created_at": datetime.utcnow().isoformat(),
                "status": "waiting_for_human"
            }
            
            self.handoff_sessions[handoff_id] = handoff_session
            
            # Generate handoff message
            handoff_message = await self._generate_handoff_message(handoff_decision, context)
            
            return {
                "message": handoff_message,
                "handoff_id": handoff_id,
                "handoff_session": handoff_session,
                "type": "handoff_initiated"
            }
            
        except Exception as e:
            self.logger.error(f"Handoff initiation failed: {e}")
            return {
                "message": "I understand you'd like to speak with a human. Let me connect you with someone who can help.",
                "handoff_id": None,
                "type": "handoff_fallback"
            }
    
    async def _generate_handoff_message(self, handoff_decision: Dict[str, Any], 
                                      context: Dict[str, Any]) -> str:
        """Generate appropriate handoff message."""
        reason = handoff_decision["reason"]
        urgency = handoff_decision["urgency"]
        
        if reason == HandoffReason.USER_REQUEST.value:
            return "I understand you'd like to speak with a human representative. I'm connecting you with someone who can assist you better."
        elif reason == HandoffReason.COMPLEX_DECISION.value:
            return "This is a complex matter that would benefit from human expertise. Let me connect you with a specialist."
        elif reason == HandoffReason.ERROR_RECOVERY.value:
            return "I'm experiencing some technical difficulties. Let me connect you with a human who can help resolve this."
        elif reason == HandoffReason.SECURITY_CONCERN.value:
            return "For security reasons, I need to connect you with a human representative to handle this request."
        else:
            return "I'm connecting you with a human representative who can better assist you with your request."
    
    async def _generate_ai_response(self, user_input: str, context: Dict[str, Any], 
                                  reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response using reasoning chain."""
        try:
            prompt = f"""
            Generate a conversational AI response:
            
            User Input: "{user_input}"
            Reasoning Chain: {json.dumps(reasoning['chain'], indent=2)}
            Context: {json.dumps(context or {}, indent=2)}
            Conversation History: {len(self.conversation_history)} messages
            
            Generate a helpful, conversational response that:
            1. Addresses the user's input directly
            2. Shows understanding of their request
            3. Provides useful information or assistance
            4. Maintains a friendly, professional tone
            5. Suggests next steps if appropriate
            
            Return as JSON:
            {{
                "message": "Your response message here",
                "tone": "friendly/professional/helpful",
                "suggested_actions": ["action1", "action2"],
                "confidence": 0.0-1.0
            }}
            """
            
            response = await self.ai_provider.generate_response(prompt, timeout=30)
            
            try:
                ai_response = json.loads(response)
                ai_response["type"] = "ai_response"
                ai_response["generated_at"] = datetime.utcnow().isoformat()
                return ai_response
            except json.JSONDecodeError:
                return self._generate_response_fallback(user_input)
                
        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            return self._generate_response_fallback(user_input)
    
    def _generate_response_fallback(self, user_input: str) -> Dict[str, Any]:
        """Fallback AI response generation."""
        return {
            "message": "I understand your request. Let me help you with that. Is there anything specific you'd like me to assist you with?",
            "tone": "helpful",
            "suggested_actions": ["clarify", "assist"],
            "confidence": 0.6,
            "type": "ai_response",
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_follow_up_questions(self, user_input: str, context: Dict[str, Any], 
                                          response: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions for better engagement."""
        try:
            prompt = f"""
            Generate 2-3 relevant follow-up questions:
            
            User Input: "{user_input}"
            AI Response: "{response.get('message', '')}"
            Context: {json.dumps(context or {}, indent=2)}
            
            Generate questions that:
            1. Help clarify the user's needs
            2. Provide additional assistance
            3. Guide the conversation forward
            4. Show proactive thinking
            
            Return as JSON array:
            ["Question 1?", "Question 2?", "Question 3?"]
            """
            
            response_text = await self.ai_provider.generate_response(prompt, timeout=20)
            
            try:
                questions = json.loads(response_text)
                if isinstance(questions, list):
                    return questions[:3]  # Limit to 3 questions
                else:
                    return self._generate_follow_up_fallback()
            except json.JSONDecodeError:
                return self._generate_follow_up_fallback()
                
        except Exception as e:
            self.logger.error(f"Follow-up generation failed: {e}")
            return self._generate_follow_up_fallback()
    
    def _generate_follow_up_fallback(self) -> List[str]:
        """Fallback follow-up questions."""
        return [
            "Is there anything else I can help you with?",
            "Would you like me to explain anything in more detail?",
            "Do you have any other questions about this?"
        ]
    
    async def _update_conversation_history(self, user_input: str, response: Dict[str, Any], 
                                         context: Dict[str, Any]):
        """Update conversation history."""
        conversation_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "ai_response": response.get("message", ""),
            "response_type": response.get("type", "unknown"),
            "context": context,
            "state": self.current_state.value
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    async def handle_human_input(self, handoff_id: str, human_input: str) -> Dict[str, Any]:
        """Handle input from human during handoff."""
        try:
            if handoff_id not in self.handoff_sessions:
                return {"error": "Invalid handoff session"}
            
            handoff_session = self.handoff_sessions[handoff_id]
            handoff_session["human_input"] = human_input
            handoff_session["handoff_completed_at"] = datetime.utcnow().isoformat()
            handoff_session["status"] = "handoff_complete"
            
            # Update conversation state
            self.current_state = ConversationState.HANDOFF_COMPLETE
            
            return {
                "message": "Thank you for the information. I'll continue assisting you with the automation.",
                "handoff_id": handoff_id,
                "status": "handoff_complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Human input handling failed: {e}")
            return {"error": str(e)}
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary and statistics."""
        return {
            "total_messages": len(self.conversation_history),
            "current_state": self.current_state.value,
            "active_handoffs": len([s for s in self.handoff_sessions.values() if s["status"] == "waiting_for_human"]),
            "completed_handoffs": len([s for s in self.handoff_sessions.values() if s["status"] == "handoff_complete"]),
            "reasoning_chains": len(self.reasoning_chain),
            "follow_up_questions": len(self.follow_up_questions),
            "session_start": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_activity": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "response": "I apologize, but I'm experiencing some technical difficulties. Please try again or let me know if you need assistance.",
            "reasoning": {
                "chain": ["Error occurred", "Generating fallback response"],
                "confidence": 0.3,
                "key_insights": ["Technical issue detected"],
                "recommended_action": "Retry or handoff"
            },
            "follow_up_questions": ["Would you like to try again?", "Should I connect you with a human?"],
            "handoff_required": False,
            "conversation_state": ConversationState.IDLE.value,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }