#!/usr/bin/env python3
"""
Test script for the Autonomous Multi-Agent Automation Platform
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.core.config import Config
from src.core.database import DatabaseManager
from src.core.vector_store import VectorStore
from src.core.audit import AuditLogger
from src.core.ai_provider import AIProvider
from src.utils.logger import setup_logging


async def test_platform_initialization():
    """Test the platform initialization."""
    print("Testing Autonomous Multi-Agent Automation Platform...")
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load configuration
        print("1. Loading configuration...")
        config = Config()
        print(f"   ✓ Configuration loaded successfully")
        print(f"   - Environment: {config.environment}")
        print(f"   - Log level: {config.log_level}")
        print(f"   - Data path: {config.data_path}")
        
        # Test database initialization
        print("\n2. Testing database initialization...")
        database = DatabaseManager(config.database)
        await database.initialize()
        print(f"   ✓ Database initialized successfully")
        print(f"   - Database path: {database.db_path}")
        
        # Test vector store initialization
        print("\n3. Testing vector store initialization...")
        vector_store = VectorStore(config.database)
        await vector_store.initialize()
        print(f"   ✓ Vector store initialized successfully")
        print(f"   - Vector DB path: {config.database.vector_db_path}")
        
        # Test audit logger initialization
        print("\n4. Testing audit logger initialization...")
        audit_logger = AuditLogger(config)
        await audit_logger.initialize()
        print(f"   ✓ Audit logger initialized successfully")
        print(f"   - Audit DB path: {audit_logger.db_path}")
        
        # Test AI provider initialization
        print("\n5. Testing AI provider initialization...")
        ai_provider = AIProvider(config.ai)
        await ai_provider.initialize()
        print(f"   ✓ AI provider initialized successfully")
        
        # Get available providers
        available_providers = [p for p, config in ai_provider.providers.items() if config['available']]
        print(f"   - Available AI providers: {available_providers}")
        
        # Test basic AI functionality
        if available_providers:
            print("\n6. Testing AI functionality...")
            try:
                response = await ai_provider.generate_response(
                    "Hello! This is a test message. Please respond with 'Test successful'.",
                    max_tokens=50
                )
                print(f"   ✓ AI response received: {response[:100]}...")
            except Exception as e:
                print(f"   ⚠ AI test failed (this is expected if no API keys are configured): {e}")
        
        # Test database operations
        print("\n7. Testing database operations...")
        
        # Test saving and retrieving performance metrics
        test_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "avg_duration": 0.0
        }
        
        await database.save_performance_metrics("test_metrics", test_metrics)
        retrieved_metrics = await database.get_performance_metrics()
        print(f"   ✓ Database operations successful")
        print(f"   - Retrieved metrics: {len(retrieved_metrics)} entries")
        
        # Test vector store operations
        print("\n8. Testing vector store operations...")
        
        # Test storing and retrieving a simple document
        test_document = {
            "workflow_id": "test_workflow",
            "content": "This is a test workflow for platform verification",
            "domain": "testing",
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        await vector_store.store_plan("test_workflow", test_document)
        stats = await vector_store.get_statistics()
        print(f"   ✓ Vector store operations successful")
        print(f"   - Vector store statistics: {stats}")
        
        # Test audit logging
        print("\n9. Testing audit logging...")
        
        await audit_logger.log_system_activity(
            "platform_test",
            {"test_type": "initialization", "status": "success"}
        )
        
        audit_stats = await audit_logger.get_audit_statistics()
        print(f"   ✓ Audit logging successful")
        print(f"   - Audit statistics: {audit_stats}")
        
        # Cleanup
        print("\n10. Cleaning up...")
        await database.shutdown()
        await vector_store.shutdown()
        await audit_logger.shutdown()
        await ai_provider.shutdown()
        print(f"   ✓ Cleanup completed successfully")
        
        print("\n🎉 Platform initialization test completed successfully!")
        print("\nThe Autonomous Multi-Agent Automation Platform is ready to use.")
        print("\nNext steps:")
        print("1. Configure API keys in .env file for AI providers")
        print("2. Run 'python main.py' to start the platform")
        print("3. Access the API at http://localhost:8000")
        print("4. View API documentation at http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Platform initialization test failed: {e}")
        logging.error(f"Platform test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_platform_initialization())
    sys.exit(0 if success else 1)