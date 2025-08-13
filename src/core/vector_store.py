"""
Vector Store
===========

Vector database for semantic storage and learning capabilities using ChromaDB.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStore:
    """Vector database for semantic storage and learning using ChromaDB."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collections = {}
        self.db_path = Path(self.config.vector_db_path)
        
    async def initialize(self):
        """Initialize vector store."""
        try:
            # Ensure vector DB directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with simple settings
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize collections with simple embedding function
            self.collections = {}
            
            # Workflow plans collection
            self.collections["workflow_plans"] = self.client.get_or_create_collection(
                name="workflow_plans",
                metadata={"description": "Workflow planning and execution patterns"}
            )
            
            # Execution patterns collection
            self.collections["execution_patterns"] = self.client.get_or_create_collection(
                name="execution_patterns",
                metadata={"description": "Successful and failed execution patterns"}
            )
            
            # Conversations collection
            self.collections["conversations"] = self.client.get_or_create_collection(
                name="conversations",
                metadata={"description": "Conversation history and context"}
            )
            
            # Task templates collection
            self.collections["task_templates"] = self.client.get_or_create_collection(
                name="task_templates",
                metadata={"description": "Reusable task templates and patterns"}
            )
            
            # Failure patterns collection
            self.collections["failure_patterns"] = self.client.get_or_create_collection(
                name="failure_patterns",
                metadata={"description": "Failure patterns and resolution strategies"}
            )
            
            self.logger.info(f"Initialized {len(self.collections)} collections")
            self.logger.info(f"Vector store initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            # Continue without vector store
            self.client = None
            self.collections = {}
            
    async def store_plan(self, workflow_id: str, plan: Dict[str, Any]):
        """Store workflow plan in vector database."""
        try:
            if not self.client:
                self.logger.warning("Vector store not available, skipping plan storage")
                return
                
            # Create document content
            content = f"Workflow Plan: {plan.get('workflow_id', workflow_id)}\n"
            content += f"Domain: {plan.get('analysis', {}).get('domain', 'general')}\n"
            content += f"Tasks: {len(plan.get('tasks', []))}\n"
            content += f"Estimated Duration: {plan.get('estimated_duration', 0)}s\n"
            content += f"Success Probability: {plan.get('success_probability', 0.0)}\n"
            
            # Add task details
            for task in plan.get('tasks', []):
                content += f"- {task.get('name', 'Unknown')}: {task.get('type', 'general')}\n"
                
            # Store in collection
            self.collections["workflow_plans"].add(
                documents=[content],
                metadatas=[{
                    "workflow_id": workflow_id,
                    "domain": plan.get('analysis', {}).get('domain', 'general'),
                    "task_count": len(plan.get('tasks', [])),
                    "estimated_duration": plan.get('estimated_duration', 0),
                    "success_probability": plan.get('success_probability', 0.0),
                    "created_at": datetime.utcnow().isoformat()
                }],
                ids=[f"plan_{workflow_id}"]
            )
            
            self.logger.info(f"Stored plan for workflow {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store plan: {e}", exc_info=True)
            
    async def store_execution_pattern(self, workflow_id: str, execution_result: Union[Any, Dict[str, Any]]):
        """Store execution pattern for learning and optimization."""
        try:
            if not self.client:
                self.logger.warning("Vector store not available, skipping pattern storage")
                return
                
            # Handle both ExecutionResult objects and dictionaries
            if isinstance(execution_result, dict):
                success = execution_result.get("success", False)
                duration = execution_result.get("duration", 0)
                steps = execution_result.get("steps", [])
                errors = execution_result.get("errors", [])
            else:
                success = execution_result.success
                duration = execution_result.duration
                steps = execution_result.steps
                errors = execution_result.errors
                
            # Create document content
            content = f"Execution Pattern: {workflow_id}\n"
            content += f"Success: {success}\n"
            content += f"Duration: {duration}s\n"
            content += f"Steps: {len(steps)}\n"
            content += f"Errors: {len(errors)}\n"
            
            # Add step details
            for step in steps:
                if isinstance(step, dict):
                    task_name = step.get("task_name", "Unknown")
                    task_type = step.get("task_type", "Unknown")
                    step_success = step.get("success", False)
                else:
                    task_name = step.task_name
                    task_type = step.task_type
                    step_success = step.success
                content += f"- {task_name}: {task_type} ({'success' if step_success else 'failed'})\n"
                
            # Add error details
            for error in errors:
                content += f"Error: {error}\n"
                
            # Store in collection
            self.collections["execution_patterns"].add(
                documents=[content],
                metadatas=[{
                    "workflow_id": workflow_id,
                    "success": success,
                    "duration": duration,
                    "step_count": len(steps),
                    "error_count": len(errors),
                    "created_at": datetime.utcnow().isoformat()
                }],
                ids=[f"exec_{workflow_id}_{datetime.utcnow().timestamp()}"]
            )
            
            # Store failure pattern if execution failed
            if not success:
                await self._store_failure_pattern(workflow_id, execution_result)
                
            self.logger.info(f"Stored execution pattern for workflow {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store execution pattern: {e}", exc_info=True)
            
    async def _store_failure_pattern(self, workflow_id: str, execution_result: Any):
        """Store failure pattern for analysis."""
        try:
            content = f"Failure Pattern: {workflow_id}\n"
            content += f"Duration: {execution_result.duration}s\n"
            content += f"Steps: {len(execution_result.steps)}\n"
            
            # Add error details
            for error in execution_result.errors:
                content += f"Error: {error}\n"
                
            # Add failed step details
            for step in execution_result.steps:
                if not step.success:
                    content += f"Failed Step: {step.task_name} ({step.task_type}) - {step.error}\n"
                    
            # Store in failure patterns collection
            self.collections["failure_patterns"].add(
                documents=[content],
                metadatas=[{
                    "workflow_id": workflow_id,
                    "duration": execution_result.duration,
                    "step_count": len(execution_result.steps),
                    "error_count": len(execution_result.errors),
                    "failure_type": "execution_failure",
                    "created_at": datetime.utcnow().isoformat()
                }],
                ids=[f"failure_{workflow_id}_{datetime.utcnow().timestamp()}"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store failure pattern: {e}", exc_info=True)
            
    async def find_execution_patterns(self, domain: str, pattern_type: str = "general") -> List[Dict[str, Any]]:
        """Find relevant execution patterns."""
        try:
            if not self.client:
                return []
                
            # Create query content
            query_content = f"Execution Pattern Query\n"
            query_content += f"Domain: {domain}\n"
            query_content += f"Type: {pattern_type}\n"
            
            # Search for patterns
            results = self.collections["execution_patterns"].query(
                query_texts=[query_content],
                n_results=10
            )
            
            patterns = []
            for i, doc in enumerate(results['documents'][0]):
                patterns.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
                })
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find execution patterns: {e}", exc_info=True)
            return []
            
    async def find_similar_failures(self, execution_result: Any) -> List[Dict[str, Any]]:
        """Find similar failure patterns."""
        try:
            if not self.client:
                return []
                
            # Create query content
            query_content = f"Failure Pattern Query\n"
            query_content += f"Error Count: {len(execution_result.errors)}\n"
            
            for error in execution_result.errors:
                query_content += f"Error: {error}\n"
                
            # Search for similar patterns
            results = self.collections["failure_patterns"].query(
                query_texts=[query_content],
                n_results=5
            )
            
            similar_failures = []
            for i, doc in enumerate(results['documents'][0]):
                similar_failures.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
                })
                
            return similar_failures
            
        except Exception as e:
            self.logger.error(f"Failed to find similar failures: {e}", exc_info=True)
            return []
            
    async def store_conversation(self, conversation: Any):
        """Store conversation in vector database."""
        try:
            if not self.client:
                self.logger.warning("Vector store not available, skipping conversation storage")
                return
                
            # Create document content from conversation
            content = f"Conversation: {conversation.session_id}\n"
            content += f"Messages: {conversation.get_message_count()}\n"
            content += f"Duration: {(conversation.updated_at - conversation.created_at).total_seconds()}s\n"
            
            # Add recent messages
            for message in conversation.messages[-5:]:  # Last 5 messages
                content += f"{message.message_type.value.upper()}: {message.content[:100]}...\n"
                
            # Store in collection
            self.collections["conversations"].add(
                documents=[content],
                metadatas=[{
                    "session_id": conversation.session_id,
                    "message_count": conversation.get_message_count(),
                    "duration": (conversation.updated_at - conversation.created_at).total_seconds(),
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat()
                }],
                ids=[f"conv_{conversation.session_id}_{datetime.utcnow().timestamp()}"]
            )
            
            self.logger.info(f"Stored conversation for session {conversation.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}", exc_info=True)
            
    async def get_conversation_history(self) -> List[Any]:
        """Get conversation history from vector database."""
        try:
            if not self.client:
                return []
                
            # Get all conversations
            results = self.collections["conversations"].get()
            
            conversations = []
            for i, doc in enumerate(results['documents']):
                conversations.append({
                    "content": doc,
                    "metadata": results['metadatas'][i],
                    "id": results['ids'][i]
                })
                
            return conversations
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}", exc_info=True)
            return []
            
    async def find_similar_plans(self, workflow_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar workflow plans."""
        try:
            if not self.client:
                return []
                
            # Create query content
            query_content = f"Workflow Query: {workflow_request.get('name', 'Unknown')}\n"
            query_content += f"Domain: {workflow_request.get('domain', 'general')}\n"
            query_content += f"Description: {workflow_request.get('description', '')}\n"
            
            # Search for similar plans
            results = self.collections["workflow_plans"].query(
                query_texts=[query_content],
                n_results=3
            )
            
            similar_plans = []
            for i, doc in enumerate(results['documents'][0]):
                similar_plans.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
                })
                
            return similar_plans
            
        except Exception as e:
            self.logger.error(f"Failed to find similar plans: {e}", exc_info=True)
            return []
            
    async def store_task_template(self, template: Dict[str, Any]):
        """Store reusable task template."""
        try:
            if not self.client:
                self.logger.warning("Vector store not available, skipping template storage")
                return
                
            # Create document content
            content = f"Task Template: {template.get('name', 'Unknown')}\n"
            content += f"Type: {template.get('type', 'general')}\n"
            content += f"Description: {template.get('description', '')}\n"
            content += f"Parameters: {json.dumps(template.get('parameters', {}))}\n"
            
            # Store in collection
            self.collections["task_templates"].add(
                documents=[content],
                metadatas=[{
                    "name": template.get('name', 'Unknown'),
                    "type": template.get('type', 'general'),
                    "usage_count": template.get('usage_count', 0),
                    "success_rate": template.get('success_rate', 0.0),
                    "created_at": datetime.utcnow().isoformat()
                }],
                ids=[f"template_{template.get('name', 'unknown').replace(' ', '_')}"]
            )
            
            self.logger.info(f"Stored task template: {template.get('name', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to store task template: {e}", exc_info=True)
            
    async def find_task_templates(self, task_type: str, domain: str = "general") -> List[Dict[str, Any]]:
        """Find relevant task templates."""
        try:
            if not self.client:
                return []
                
            # Create query content
            query_content = f"Task Template Query\n"
            query_content += f"Type: {task_type}\n"
            query_content += f"Domain: {domain}\n"
            
            # Search for templates
            results = self.collections["task_templates"].query(
                query_texts=[query_content],
                n_results=5
            )
            
            templates = []
            for i, doc in enumerate(results['documents'][0]):
                templates.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
                })
                
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to find task templates: {e}", exc_info=True)
            return []
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if not self.client:
                return {"error": "Vector store not available"}
                
            stats = {}
            for collection_name, collection in self.collections.items():
                try:
                    count = collection.count()
                    stats[collection_name] = count
                except Exception as e:
                    stats[collection_name] = f"Error: {e}"
                    
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {"error": str(e)}
            
    async def shutdown(self):
        """Shutdown vector store."""
        try:
            if self.client:
                # ChromaDB handles cleanup automatically
                self.client = None
                self.collections = {}
                
            self.logger.info("Vector store shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during vector store shutdown: {e}", exc_info=True)