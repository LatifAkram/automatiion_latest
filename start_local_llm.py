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
            
            # For now, return a mock response since we don't have the actual LLM running
            # In a real implementation, this would call the actual local LLM
            response_content = self._generate_mock_response(messages, temperature)
            
            response_data = {
                "id": f"chatcmpl-{int(asyncio.get_event_loop().time())}",
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(str(messages)) + len(response_content.split())
                }
            }
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            return web.json_response({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }, status=500)
            
    def _generate_mock_response(self, messages: list, temperature: float) -> str:
        """Generate a mock response for testing."""
        if not messages:
            return "Hello! I'm your local AI assistant. How can I help you today?"
            
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
                
        if not last_user_message:
            return "I'm here to help! What would you like to know?"
            
        # Generate contextual response based on the message
        message_lower = last_user_message.lower()
        
        if "book" in message_lower and "ticket" in message_lower:
            return """🤖 **AUTOMATION REQUEST DETECTED**

I understand you want to automate ticket booking! Let me break down what I can help you with:

**🎯 Available Automation Types:**
• **Web Automation**: Form filling, data extraction, navigation
• **Ticket Booking**: Flight, train, bus ticket searches with real websites
• **E-commerce**: Product searches, price comparisons, reviews
• **Banking**: Account monitoring, transaction analysis
• **Healthcare**: Appointment scheduling, data collection
• **Entertainment**: Content aggregation, streaming analysis

**🔧 How I Can Help:**
1. **Analyze your request** and create a comprehensive workflow
2. **Execute automation** with real-time progress updates
3. **Provide detailed reasoning** for each action taken
4. **Handle errors** and suggest intelligent solutions
5. **Generate detailed reports** with screenshots and data
6. **Real-time monitoring** of automation progress

**💡 Example Requests:**
• "Book a flight from Delhi to Mumbai for Friday 6 AM"
• "Search for the best laptop deals under ₹50,000"
• "Fill out this form with my details"
• "Extract product prices from Amazon"
• "Monitor stock prices for Apple and Google"

**🚀 What would you like to automate?** Please provide specific details about your task, and I'll create a comprehensive automation workflow with real-time progress tracking!"""
        
        elif "search" in message_lower:
            return """🔍 **SEARCH REQUEST DETECTED**

I can help you search across multiple sources for comprehensive information!

**🌐 Search Capabilities:**
• **Web Search**: Google, Bing, DuckDuckGo with real-time results
• **Live Data Scraping**: Direct website access for current information
• **Multi-source Aggregation**: Results from multiple sites simultaneously
• **Structured Results**: Organized, relevant information with source attribution
• **Real-time Updates**: Latest information as it becomes available

**📊 What I Can Search:**
• **Product Information**: Prices, reviews, specifications, availability
• **Travel Data**: Flights, hotels, destinations, prices, availability
• **Financial Data**: Stock prices, market trends, company information
• **News & Updates**: Latest information, trends, developments
• **Technical Data**: Documentation, tutorials, code examples
• **Academic Research**: Papers, studies, scholarly articles

**🔧 Search Process:**
1. **Analyze your query** for optimal search strategy
2. **Search multiple sources** simultaneously for comprehensive coverage
3. **Aggregate and filter** results for relevance and accuracy
4. **Provide structured response** with sources and confidence scores
5. **Include relevant links** and detailed snippets
6. **Real-time validation** of information accuracy

**💡 Example Searches:**
• "Best smartphones 2024 under ₹30,000 with camera comparison"
• "Flight prices Delhi to Bangalore next week with multiple airlines"
• "Latest news about AI developments and market impact"
• "How to automate web scraping with Python and Playwright"
• "Stock market analysis for tech companies this quarter"

**🚀 What would you like me to search for?** I'll provide comprehensive, real-time results from multiple sources with detailed analysis!"""
        
        elif "help" in message_lower or "what can you do" in message_lower:
            return """🆘 **COMPREHENSIVE HELP & GUIDANCE CENTER**

Welcome to our advanced AI-powered automation platform! Here's everything you need to know:

**🎯 Platform Overview:**
This is a **world-class AI automation platform** that can handle ultra-complex tasks across multiple domains using real web automation, intelligent search, multi-agent coordination, and advanced reasoning capabilities.

**🤖 Key Features:**
• **Multi-Agent System**: 5 specialized AI agents working in perfect coordination
• **Real Web Automation**: Live browser automation with Playwright for actual website interaction
• **Intelligent Search**: Multi-source search with real-time data aggregation
• **Conversational AI**: Natural language interaction with reasoning and context
• **Progress Tracking**: Real-time updates and detailed status monitoring
• **Export System**: Generate comprehensive reports in multiple formats
• **Human-AI Handoff**: Seamless transition between AI and human control

**🔧 How to Use:**
1. **Describe your task** in natural language with specific details
2. **I'll analyze** and create a comprehensive workflow with reasoning
3. **Execute automation** with real-time progress updates and detailed explanations
4. **Provide results** with comprehensive reports, screenshots, and analysis

**💡 Need specific help?** Just ask me about any particular feature or task, and I'll provide detailed guidance!"""
        
        else:
            return f"I understand you said: '{last_user_message}'. I'm your local AI assistant and I'm here to help you with automation tasks, web searches, data analysis, and more. What specific task would you like to work on?"

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