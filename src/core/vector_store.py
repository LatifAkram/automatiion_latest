"""
Vector Store Management
======================

Vector database management using ChromaDB for storing and retrieving
embeddings, plans, and patterns.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Vector store functionality will be limited.")


class VectorStore:
    """Vector database manager using ChromaDB."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.db_path = Path(config.vector_db_path)
        self.collections = {}
        
        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize vector store and collections."""
        try:
            if not CHROMADB_AVAILABLE:
                self.logger.warning("ChromaDB not available. Using mock vector store.")
                await self._initialize_mock_store()
                return
                
            # Initialize ChromaDB client with simple settings
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize collections
            await self._initialize_collections()
            
            self.logger.info(f"Vector store initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            # Fallback to mock store
            await self._initialize_mock_store()
            
    async def _initialize_collections(self):
        """Initialize ChromaDB collections."""
        try:
            # Define collection names and their metadata
            collection_configs = {
                "workflow_plans": {
                    "description": "Stores workflow execution plans and strategies"
                },
                "execution_patterns": {
                    "description": "Stores successful execution patterns and templates"
                },
                "conversations": {
                    "description": "Stores conversation history and context"
                },
                "task_templates": {
                    "description": "Stores reusable task templates and configurations"
                },
                "failure_patterns": {
                    "description": "Stores failure patterns for learning and improvement"
                }
            }
            
            # Create or get collections
            for collection_name, metadata in collection_configs.items():
                try:
                    collection = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata=metadata
                    )
                    self.collections[collection_name] = collection
                    self.logger.info(f"Collection '{collection_name}' initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize collection '{collection_name}': {e}")
                    # Create a mock collection
                    self.collections[collection_name] = MockCollection(collection_name)
                    
            self.logger.info(f"Initialized {len(self.collections)} collections")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collections: {e}", exc_info=True)
            # Create mock collections as fallback
            await self._initialize_mock_store()
            
    async def _initialize_mock_store(self):
        """Initialize mock vector store when ChromaDB is not available."""
        self.logger.info("Initializing mock vector store")
        
        # Create mock collections
        collection_names = [
            "workflow_plans",
            "execution_patterns", 
            "conversations",
            "task_templates",
            "failure_patterns"
        ]
        
        for name in collection_names:
            self.collections[name] = MockCollection(name)
            
        self.logger.info(f"Mock vector store initialized with {len(self.collections)} collections")
        
    async def store_plan(self, workflow_id: str, plan_data: Dict[str, Any]) -> bool:
        """Store workflow execution plan."""
        try:
            if "workflow_plans" not in self.collections:
                self.logger.error("workflow_plans collection not available")
                return False
                
            # Convert plan data to string for storage
            content = json.dumps(plan_data, default=str)
            
            # Add to collection
            self.collections["workflow_plans"].add(
                documents=[content],
                metadatas=[{
                    "workflow_id": workflow_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "plan_type": "execution_plan"
                }],
                ids=[f"plan_{workflow_id}"]
            )
            
            self.logger.info(f"Plan stored for workflow: {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store plan: {e}", exc_info=True)
            return False
            
    async def store_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]) -> bool:
        """Store execution pattern."""
        try:
            collection_name = "execution_patterns"
            if collection_name not in self.collections:
                self.logger.error(f"{collection_name} collection not available")
                return False
                
            content = json.dumps(pattern_data, default=str)
            pattern_id = f"pattern_{pattern_type}_{datetime.utcnow().timestamp()}"
            
            self.collections[collection_name].add(
                documents=[content],
                metadatas=[{
                    "pattern_type": pattern_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "success_rate": pattern_data.get("success_rate", 0.0)
                }],
                ids=[pattern_id]
            )
            
            self.logger.info(f"Pattern stored: {pattern_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store pattern: {e}", exc_info=True)
            return False
            
    async def store_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> bool:
        """Store conversation data."""
        try:
            if "conversations" not in self.collections:
                self.logger.error("conversations collection not available")
                return False
                
            content = json.dumps(conversation_data, default=str)
            
            self.collections["conversations"].add(
                documents=[content],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "user_id": conversation_data.get("user_id", "unknown")
                }],
                ids=[f"conv_{conversation_id}"]
            )
            
            self.logger.info(f"Conversation stored: {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}", exc_info=True)
            return False
            
    async def store_task_template(self, template_name: str, template_data: Dict[str, Any]) -> bool:
        """Store task template."""
        try:
            if "task_templates" not in self.collections:
                self.logger.error("task_templates collection not available")
                return False
                
            content = json.dumps(template_data, default=str)
            
            self.collections["task_templates"].add(
                documents=[content],
                metadatas=[{
                    "template_name": template_name,
                    "created_at": datetime.utcnow().isoformat(),
                    "template_type": template_data.get("type", "generic")
                }],
                ids=[f"template_{template_name}"]
            )
            
            self.logger.info(f"Task template stored: {template_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store task template: {e}", exc_info=True)
            return False
            
    async def store_failure_pattern(self, failure_type: str, failure_data: Dict[str, Any]) -> bool:
        """Store failure pattern for learning."""
        try:
            if "failure_patterns" not in self.collections:
                self.logger.error("failure_patterns collection not available")
                return False
                
            content = json.dumps(failure_data, default=str)
            failure_id = f"failure_{failure_type}_{datetime.utcnow().timestamp()}"
            
            self.collections["failure_patterns"].add(
                documents=[content],
                metadatas=[{
                    "failure_type": failure_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "severity": failure_data.get("severity", "medium")
                }],
                ids=[failure_id]
            )
            
            self.logger.info(f"Failure pattern stored: {failure_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store failure pattern: {e}", exc_info=True)
            return False
            
    async def search_plans(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for workflow plans."""
        try:
            if "workflow_plans" not in self.collections:
                return []
                
            results = self.collections["workflow_plans"].query(
                query_texts=[query],
                n_results=n_results
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            self.logger.error(f"Failed to search plans: {e}", exc_info=True)
            return []
            
    async def search_patterns(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for execution patterns."""
        try:
            if "execution_patterns" not in self.collections:
                return []
                
            results = self.collections["execution_patterns"].query(
                query_texts=[query],
                n_results=n_results
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            self.logger.error(f"Failed to search patterns: {e}", exc_info=True)
            return []
            
    async def search_conversations(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history."""
        try:
            if "conversations" not in self.collections:
                return []
                
            results = self.collections["conversations"].query(
                query_texts=[query],
                n_results=n_results
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            self.logger.error(f"Failed to search conversations: {e}", exc_info=True)
            return []
            
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format search results for consistent output."""
        formatted_results = []
        
        try:
            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])
            ids = results.get("ids", [[]])
            
            for i in range(len(documents[0])):
                try:
                    doc_data = json.loads(documents[0][i]) if documents[0][i] else {}
                    formatted_results.append({
                        "id": ids[0][i] if ids and ids[0] else None,
                        "content": doc_data,
                        "metadata": metadatas[0][i] if metadatas and metadatas[0] else {},
                        "score": results.get("distances", [[]])[0][i] if results.get("distances") else None
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to format result {i}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to format search results: {e}")
            
        return formatted_results
        
    async def get_statistics(self) -> Dict[str, int]:
        """Get vector store statistics."""
        try:
            stats = {}
            
            for collection_name, collection in self.collections.items():
                try:
                    if hasattr(collection, 'count'):
                        stats[collection_name] = collection.count()
                    else:
                        stats[collection_name] = 0
                except Exception as e:
                    self.logger.warning(f"Failed to get count for {collection_name}: {e}")
                    stats[collection_name] = 0
                    
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {}
            
    async def cleanup(self):
        """Cleanup vector store resources."""
        try:
            if self.client:
                # ChromaDB cleanup if needed
                pass
            self.logger.info("Vector store cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup vector store: {e}", exc_info=True)


class MockCollection:
    """Mock collection for when ChromaDB is not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.ids = []
        
    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Mock add method."""
        for doc, metadata, doc_id in zip(documents, metadatas, ids):
            self.documents.append(doc)
            self.metadatas.append(metadata)
            self.ids.append(doc_id)
            
    def query(self, query_texts: List[str], n_results: int = 5):
        """Mock query method."""
        # Return mock results
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
            "ids": [self.ids[:n_results]],
            "distances": [[0.1] * min(n_results, len(self.documents))]
        }
        
    def count(self) -> int:
        """Mock count method."""
        return len(self.documents)