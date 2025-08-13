#!/usr/bin/env python3
"""
Server Startup Script
====================

Separate script to start the API server without event loop conflicts.
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
        start_api_server(orchestrator)
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())