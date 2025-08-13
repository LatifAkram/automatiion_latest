#!/usr/bin/env python3
"""
Autonomous Multi-Agent Automation Platform
==========================================

A comprehensive automation platform that executes ultra-complex workflows
across all domains using AI-powered agents for planning, execution, and reasoning.

Core Components:
- AI-1: Planner Agent (Brain)
- AI-2: Execution Agents (Automation)
- AI-3: Conversational Agent (Reasoning & Context)
- Vector DB for learning and self-healing
- Enterprise compliance and governance
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.core.orchestrator import MultiAgentOrchestrator
from src.core.config import Config
from src.utils.logger import setup_logging
from src.api.server import start_api_server


async def main():
    """Main entry point for the automation platform."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")
        
        # Initialize the multi-agent orchestrator
        orchestrator = MultiAgentOrchestrator(config)
        await orchestrator.initialize()
        logger.info("Multi-agent orchestrator initialized")
        
        # Start the API server
        logger.info("Starting API server...")
        server_thread = start_api_server(orchestrator)
        
        # Keep the main thread alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            if 'orchestrator' in locals():
                await orchestrator.shutdown()
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        if 'orchestrator' in locals():
            await orchestrator.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if 'orchestrator' in locals():
            await orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
