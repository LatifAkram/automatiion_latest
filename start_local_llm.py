#!/usr/bin/env python3
"""
Local LLM Server Starter
========================

This script starts a local LLM server for the automation platform.
"""

import asyncio
import aiohttp
from aiohttp import web
import json
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_BASE = os.getenv("LOCAL_LLM_BASE", "http://localhost:1234/v1")
MODEL_NAME = "deepseek-coder-v2-lite-instruct"

class LocalLLMServer:
    def __init__(self):
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_post('/v1/chat/completions', self.chat_completions)
        self.app.router.add_get('/health', self.health_check)
        
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "model": MODEL_NAME,
            "endpoint": "/v1/chat/completions"
        })
        
    async def chat_completions(self, request):
        """Handle chat completions requests."""
        try:
            # Parse request
            data = await request.json()
            model = data.get("model", MODEL_NAME)
            messages = data.get("messages", [])
            temperature = data.get("temperature", 0.7)
            max_tokens = data.get("max_tokens", 1000)
            stream = data.get("stream", False)
            
            logger.info(f"Received chat completion request: model={model}, messages={len(messages)}")
            
            # Forward to actual local LLM server (no mock)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{LOCAL_BASE}/chat/completions", json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
                    return web.json_response(payload)
            
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            return web.json_response({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }, status=500)
            
async def main():
    """Start the local LLM server."""
    server = LocalLLMServer()
    
    # Start the server
    runner = web.AppRunner(server.app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 1234)
    await site.start()
    
    logger.info("🚀 Local LLM Server started on http://localhost:1234")
    logger.info("📋 Available endpoints:")
    logger.info("   - POST /v1/chat/completions")
    logger.info("   - GET /health")
    logger.info("")
    logger.info("💡 Test with: curl -X POST http://localhost:1234/v1/chat/completions \\")
    logger.info("   -H 'Content-Type: application/json' \\")
    logger.info("   -d '{\"model\": \"deepseek-coder-v2-lite-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
    
    try:
        # Keep the server running
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())