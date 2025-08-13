"""
AI-3: Conversational Agent (Reasoning & Context)
===============================================

The conversational agent that maintains context across sessions, provides reasoning,
and allows human takeover for tricky steps with seamless resumption.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from ..core.ai_provider import AIProvider
from ..core.vector_store import VectorStore
from ..core.audit import AuditLogger
from ..models.conversation import Conversation, Message, MessageType


class ConversationalAgent:
    """
    AI-3: Conversational Agent - Maintains context and provides intelligent reasoning.
    """
    
    def __init__(self, config: Any, vector_store: VectorStore, audit_logger: AuditLogger):
        self.config = config
        self.vector_store = vector_store
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # AI provider for conversational responses
        self.ai_provider = AIProvider(config)
        
        # Conversation management
        self.active_conversations: Dict[str, Conversation] = {}
        self.conversation_history: List[Conversation] = []
        
        # Context management
        self.global_context: Dict[str, Any] = {}
        self.session_context: Dict[str, Any] = {}
        
        # Reasoning patterns
        self.reasoning_templates = self._load_reasoning_templates()
        
    async def initialize(self):
        """Initialize the conversational agent."""
        await self.ai_provider.initialize()
        
        # Load conversation history from vector store
        await self._load_conversation_history()
        
        self.logger.info("Conversational Agent initialized")
        
    async def process_message(self, user_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user message and return a response with context."""
        try:
            # Add user_id to context
            if context is None:
                context = {}
            context["user_id"] = user_id
            
            # Get AI response
            response = await self.chat(message, context)
            
            # Log the conversation
            await self.audit_logger.log_conversation(
                session_id=context.get("session_id", "default"),
                message_type="user",
                message_content=message,
                user_id=user_id,
                workflow_id=context.get("workflow_id"),
                task_id=context.get("task_id")
            )
            
            await self.audit_logger.log_conversation(
                session_id=context.get("session_id", "default"),
                message_type="ai",
                message_content=response,
                user_id="ai_system",
                workflow_id=context.get("workflow_id"),
                task_id=context.get("task_id")
            )
            
            return {
                "response": response,
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error processing your message. Please try again.",
                "error": str(e),
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            
    def _load_reasoning_templates(self) -> Dict[str, Any]:
        """Load reasoning templates for different conversation types."""
        return {
            "workflow_explanation": {
                "template": "I'm executing this workflow because {reason}. The key steps are: {steps}. Expected outcome: {outcome}.",
                "variables": ["reason", "steps", "outcome"]
            },
            "decision_explanation": {
                "template": "I chose {choice} because {reasoning}. Alternative options were: {alternatives}. This decision optimizes for {optimization_goal}.",
                "variables": ["choice", "reasoning", "alternatives", "optimization_goal"]
            },
            "error_explanation": {
                "template": "The error occurred because {cause}. I'm taking these steps to resolve it: {resolution_steps}. This should prevent similar issues in the future.",
                "variables": ["cause", "resolution_steps"]
            },
            "progress_update": {
                "template": "Current progress: {progress_percentage}%. Completed: {completed_tasks}. Remaining: {remaining_tasks}. Estimated completion: {eta}.",
                "variables": ["progress_percentage", "completed_tasks", "remaining_tasks", "eta"]
            }
        }
        
    async def _load_conversation_history(self):
        """Load conversation history from vector store."""
        try:
            conversations = await self.vector_store.get_conversation_history()
            self.conversation_history = conversations
            self.logger.info(f"Loaded {len(conversations)} historical conversations")
        except Exception as e:
            self.logger.warning(f"Failed to load conversation history: {e}")
            
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None, 
                  performance_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with the conversational agent.
        
        Args:
            message: User message
            context: Additional context (workflow, task, etc.)
            performance_metrics: Current performance metrics
            
        Returns:
            AI response with reasoning and context
        """
        try:
            # Create or get conversation session
            session_id = context.get("session_id") if context else "default"
            conversation = await self._get_or_create_conversation(session_id)
            
            # Add user message
            user_message = Message(
                content=message,
                message_type=MessageType.USER,
                timestamp=datetime.utcnow(),
                context=context
            )
            conversation.add_message(user_message)
            
            # Check if this is a handoff request
            if self._is_handoff_request(message):
                return await self._handle_handoff_request(conversation, context)
            
            # Check if AI needs human intervention
            if self._needs_human_intervention(message, context):
                return await self._request_human_intervention(conversation, context)
            
            # Generate AI response with reasoning
            response = await self._generate_response(conversation, performance_metrics)
            
            # Add AI response
            ai_message = Message(
                content=response if isinstance(response, str) else response.get("content", str(response)),
                message_type=MessageType.AI,
                timestamp=datetime.utcnow(),
                context=response.get("context", {}) if isinstance(response, dict) else {}
            )
            conversation.add_message(ai_message)
            
            # Store conversation in vector store
            try:
                await self.vector_store.store_conversation(conversation.session_id, {
                    "messages": [msg.to_dict() for msg in conversation.messages],
                    "session_id": conversation.session_id,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                    "context": conversation.context
                })
            except Exception as e:
                self.logger.warning(f"Failed to store conversation: {e}")
                # Continue without storing - conversation still works
            
            # Log conversation
            await self.audit_logger.log_conversation(
                session_id=session_id,
                message_type="user",
                message_content=user_message.content,
                user_id=context.get("user_id", "unknown"),
                workflow_id=context.get("workflow_id")
            )
            
            await self.audit_logger.log_conversation(
                session_id=session_id,
                message_type="ai",
                message_content=ai_message.content,
                user_id="ai_system",
                workflow_id=context.get("workflow_id")
            )
            
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}", exc_info=True)
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _is_handoff_request(self, message: str) -> bool:
        """Check if message is a handoff request."""
        handoff_keywords = [
            "take control", "handoff", "human intervention", "manual step",
            "I need help", "can't proceed", "requires human input"
        ]
        return any(keyword.lower() in message.lower() for keyword in handoff_keywords)
    
    def _needs_human_intervention(self, message: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if AI needs human intervention."""
        intervention_triggers = [
            "captcha", "verification", "login required", "payment required",
            "form validation", "complex decision", "unexpected error",
            "security check", "two-factor", "phone verification"
        ]
        
        # Check message content
        if any(trigger.lower() in message.lower() for trigger in intervention_triggers):
            return True
        
        # Check context for automation issues
        if context and context.get("automation_status") == "blocked":
            return True
        
        return False
    
    async def _handle_handoff_request(self, conversation: Any, context: Optional[Dict[str, Any]]) -> str:
        """Handle handoff request from user."""
        handoff_id = f"handoff_{datetime.utcnow().timestamp()}"
        
        # Store handoff state
        context = context or {}
        context["handoff_id"] = handoff_id
        context["handoff_status"] = "requested"
        context["handoff_timestamp"] = datetime.utcnow().isoformat()
        
        return f"""🤝 **HUMAN INTERVENTION REQUESTED**

I've identified that this task requires human intervention. Here's what I need help with:

**Current Status**: {context.get('current_step', 'Unknown step')}
**Issue**: {context.get('issue_description', 'Requires human input')}
**Handoff ID**: {handoff_id}

**What you can do:**
1. **Take Control**: I'll pause the automation and let you handle this step
2. **Provide Input**: Give me the information I need to continue
3. **Skip Step**: If this step can be bypassed
4. **Modify Approach**: Suggest a different way to handle this

**To resume automation after you've handled the issue, simply say: "Resume automation"**

I'm ready to hand over control whenever you are! 🚀"""
    
    async def _request_human_intervention(self, conversation: Any, context: Optional[Dict[str, Any]]) -> str:
        """Request human intervention when AI encounters issues."""
        intervention_id = f"intervention_{datetime.utcnow().timestamp()}"
        
        # Store intervention state
        context = context or {}
        context["intervention_id"] = intervention_id
        context["intervention_status"] = "required"
        context["intervention_timestamp"] = datetime.utcnow().isoformat()
        
        return f"""⚠️ **HUMAN INTERVENTION REQUIRED**

I've encountered a situation that requires your expertise:

**Issue**: {context.get('issue_description', 'Complex decision or verification required')}
**Current Step**: {context.get('current_step', 'Unknown')}
**Intervention ID**: {intervention_id}

**Options:**
1. **Guide Me**: Provide specific instructions on how to proceed
2. **Take Over**: Handle this step manually and let me know when to resume
3. **Alternative Approach**: Suggest a different method
4. **Skip**: If this step can be safely bypassed

**I'll wait for your guidance before proceeding.** 

To continue after you've handled this, say: "Continue automation" 🎯"""
            
    async def _get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing conversation or create new one."""
        if session_id in self.active_conversations:
            return self.active_conversations[session_id]
            
        # Check if conversation exists in history
        for conv in self.conversation_history:
            if conv.session_id == session_id:
                self.active_conversations[session_id] = conv
                return conv
                
        # Create new conversation
        conversation = Conversation(
            session_id=session_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.active_conversations[session_id] = conversation
        return conversation
        
    async def _generate_response(self, conversation: Conversation, 
                               performance_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate AI response with reasoning and context."""
        
        # Build context for AI
        context = self._build_context(conversation, performance_metrics)
        
        # Create prompt with conversation history and context
        prompt = self._create_conversation_prompt(conversation, context)
        
        # Generate response using AI
        response_text = await self.ai_provider.generate_response(prompt)
        
        # Extract reasoning and context from response
        response_data = await self._parse_response(response_text, context)
        
        return response_data
        
    def _build_context(self, conversation: Conversation, 
                      performance_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive context for the conversation."""
        
        context = {
            "conversation_length": len(conversation.messages),
            "session_duration": (datetime.utcnow() - conversation.created_at).total_seconds(),
            "global_context": self.global_context.copy(),
            "session_context": self.session_context.copy(),
            "performance_metrics": performance_metrics or {},
            "conversation_topics": self._extract_topics(conversation),
            "user_preferences": self._extract_user_preferences(conversation)
        }
        
        return context
        
    def _create_conversation_prompt(self, conversation: Conversation, 
                                  context: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for the AI."""
        
        # Build conversation history
        history = ""
        for message in conversation.messages[-10:]:  # Last 10 messages for context
            role = "User" if message.message_type == MessageType.USER else "Assistant"
            history += f"{role}: {message.content}\n"
            
        # Build context summary
        context_summary = self._summarize_context(context)
        
        prompt = f"""
        You are an intelligent conversational agent for an automation platform. 
        You maintain context across sessions and provide clear reasoning for your responses.
        
        Current Context:
        {context_summary}
        
        Conversation History:
        {history}
        
        Instructions:
        1. Provide clear, helpful responses
        2. Explain your reasoning when making decisions
        3. Maintain context from previous messages
        4. If asked about automation, explain the process and reasoning
        5. If there are errors or issues, explain what happened and suggest solutions
        6. Be conversational but professional
        
        Please respond to the user's latest message with appropriate reasoning and context.
        """
        
        return prompt
        
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Summarize context for the AI prompt."""
        
        summary_parts = []
        
        if context.get("performance_metrics"):
            metrics = context["performance_metrics"]
            summary_parts.append(f"Performance: {len(metrics)} workflows tracked")
            
        if context.get("conversation_topics"):
            topics = context["conversation_topics"]
            summary_parts.append(f"Topics: {', '.join(topics[:3])}")
            
        if context.get("session_duration"):
            duration = context["session_duration"]
            summary_parts.append(f"Session duration: {duration:.0f}s")
            
        return "; ".join(summary_parts) if summary_parts else "No specific context"
        
    async def _parse_response(self, response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response and extract reasoning and context."""
        
        # Try to extract structured information from response
        try:
            # Look for reasoning patterns
            reasoning = self._extract_reasoning(response_text)
            
            # Extract any suggested actions
            actions = self._extract_suggested_actions(response_text)
            
            # Extract confidence level
            confidence = self._extract_confidence(response_text)
            
            return {
                "content": response_text,
                "reasoning": reasoning,
                "suggested_actions": actions,
                "confidence": confidence,
                "context": {
                    "reasoning_used": reasoning is not None,
                    "actions_suggested": len(actions) > 0,
                    "confidence_level": confidence
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")
            return {
                "content": response_text,
                "reasoning": None,
                "suggested_actions": [],
                "confidence": "medium",
                "context": {}
            }
            
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning from response text."""
        
        reasoning_indicators = [
            "because", "since", "as", "therefore", "thus", "hence",
            "the reason is", "this is because", "due to", "given that"
        ]
        
        for indicator in reasoning_indicators:
            if indicator in text.lower():
                # Extract the reasoning part
                parts = text.lower().split(indicator)
                if len(parts) > 1:
                    reasoning = parts[1].strip()
                    # Clean up the reasoning
                    reasoning = reasoning.split('.')[0] + '.'
                    return reasoning
                    
        return None
        
    def _extract_suggested_actions(self, text: str) -> List[str]:
        """Extract suggested actions from response text."""
        
        action_indicators = [
            "you should", "I recommend", "try", "consider", "suggest",
            "let's", "we can", "you could", "it would be good to"
        ]
        
        actions = []
        
        for indicator in action_indicators:
            if indicator in text.lower():
                # Extract action suggestions
                parts = text.lower().split(indicator)
                for part in parts[1:]:
                    action = part.split('.')[0].strip()
                    if action and len(action) > 10:  # Minimum meaningful length
                        actions.append(action)
                        
        return actions
        
    def _extract_confidence(self, text: str) -> str:
        """Extract confidence level from response text."""
        
        high_confidence_indicators = ["certain", "definitely", "absolutely", "sure", "confident"]
        low_confidence_indicators = ["maybe", "perhaps", "possibly", "uncertain", "not sure"]
        
        text_lower = text.lower()
        
        for indicator in high_confidence_indicators:
            if indicator in text_lower:
                return "high"
                
        for indicator in low_confidence_indicators:
            if indicator in text_lower:
                return "low"
                
        return "medium"
        
    def _extract_topics(self, conversation: Conversation) -> List[str]:
        """Extract main topics from conversation."""
        
        topics = set()
        
        for message in conversation.messages:
            content = message.content.lower()
            
            # Simple topic extraction based on keywords
            if any(word in content for word in ["workflow", "automation", "task"]):
                topics.add("automation")
            if any(word in content for word in ["error", "problem", "issue", "fail"]):
                topics.add("troubleshooting")
            if any(word in content for word in ["data", "extract", "process"]):
                topics.add("data_processing")
            if any(word in content for word in ["api", "call", "request"]):
                topics.add("api_integration")
            if any(word in content for word in ["web", "browser", "page"]):
                topics.add("web_automation")
                
        return list(topics)
        
    def _extract_user_preferences(self, conversation: Conversation) -> Dict[str, Any]:
        """Extract user preferences from conversation history."""
        
        preferences = {
            "detail_level": "medium",
            "technical_depth": "medium",
            "response_style": "conversational"
        }
        
        for message in conversation.messages:
            if message.message_type == MessageType.USER:
                content = message.content.lower()
                
                # Detect detail level preference
                if any(word in content for word in ["detailed", "specific", "step by step"]):
                    preferences["detail_level"] = "high"
                elif any(word in content for word in ["brief", "summary", "overview"]):
                    preferences["detail_level"] = "low"
                    
                # Detect technical depth preference
                if any(word in content for word in ["technical", "code", "implementation"]):
                    preferences["technical_depth"] = "high"
                elif any(word in content for word in ["simple", "explain", "layman"]):
                    preferences["technical_depth"] = "low"
                    
                # Detect response style preference
                if any(word in content for word in ["formal", "professional"]):
                    preferences["response_style"] = "formal"
                elif any(word in content for word in ["casual", "friendly"]):
                    preferences["response_style"] = "casual"
                    
        return preferences
        
    async def explain_workflow_decision(self, workflow_id: str, decision: str, 
                                      reasoning: str, alternatives: List[str]) -> str:
        """Explain a workflow decision with reasoning."""
        
        template = self.reasoning_templates["decision_explanation"]
        
        explanation = template["template"].format(
            choice=decision,
            reasoning=reasoning,
            alternatives=", ".join(alternatives),
            optimization_goal="efficiency and reliability"
        )
        
        # Store explanation in conversation context
        self.global_context[f"workflow_{workflow_id}_decision"] = {
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return explanation
        
    async def explain_error(self, error: str, cause: str, resolution_steps: List[str]) -> str:
        """Explain an error with cause and resolution steps."""
        
        template = self.reasoning_templates["error_explanation"]
        
        explanation = template["template"].format(
            cause=cause,
            resolution_steps=", ".join(resolution_steps)
        )
        
        return explanation
        
    async def provide_progress_update(self, workflow_id: str, progress: Dict[str, Any]) -> str:
        """Provide a progress update for a workflow."""
        
        template = self.reasoning_templates["progress_update"]
        
        update = template["template"].format(
            progress_percentage=progress.get("percentage", 0),
            completed_tasks=progress.get("completed", 0),
            remaining_tasks=progress.get("remaining", 0),
            eta=progress.get("eta", "unknown")
        )
        
        return update
        
    async def handle_human_takeover(self, workflow_id: str, step: str, 
                                  human_input: str) -> str:
        """Handle human takeover of a workflow step."""
        
        response = f"""
        I understand you want to take over the workflow at step: {step}
        
        Human input received: {human_input}
        
        I'll pause the automation and wait for your guidance. Once you've completed 
        the manual step, I can resume the workflow from where we left off.
        
        To resume, simply let me know when you're ready to continue.
        """
        
        # Store takeover context
        self.session_context["human_takeover"] = {
            "workflow_id": workflow_id,
            "step": step,
            "human_input": human_input,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    async def resume_after_takeover(self, workflow_id: str) -> str:
        """Resume workflow after human takeover."""
        
        takeover_context = self.session_context.get("human_takeover", {})
        
        if takeover_context.get("workflow_id") == workflow_id:
            response = f"""
            Perfect! I'm resuming the workflow from where we left off.
            
            Previous step: {takeover_context.get("step", "unknown")}
            Human input: {takeover_context.get("human_input", "none")}
            
            I'll continue with the automation from this point forward.
            """
            
            # Clear takeover context
            self.session_context.pop("human_takeover", None)
            
            return response
        else:
            return "I don't see any active human takeover for this workflow. The automation should continue normally."
            
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a conversation session."""
        
        conversation = self.active_conversations.get(session_id)
        if not conversation:
            return {"error": "Conversation not found"}
            
        return {
            "session_id": session_id,
            "message_count": len(conversation.messages),
            "duration": (conversation.updated_at - conversation.created_at).total_seconds(),
            "topics": self._extract_topics(conversation),
            "user_preferences": self._extract_user_preferences(conversation),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat()
        }
        
    async def shutdown(self):
        """Shutdown the conversational agent."""
        
        # Save all active conversations
        for session_id, conversation in self.active_conversations.items():
            try:
                await self.vector_store.store_conversation(conversation.session_id, {
                    "messages": [msg.to_dict() for msg in conversation.messages],
                    "session_id": conversation.session_id,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                    "context": conversation.context
                })
            except Exception as e:
                self.logger.warning(f"Failed to store conversation during shutdown: {e}")
                # Continue shutdown process
            
        # Shutdown AI provider
        await self.ai_provider.shutdown()
        
        self.logger.info("Conversational Agent shutdown complete")