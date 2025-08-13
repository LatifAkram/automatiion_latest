"""
AI Provider
==========

Handles communication with different AI providers (OpenAI, Anthropic, Google Gemini, Local LLM)
with intelligent fallback mechanisms and provider selection.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import aiohttp
import hashlib

# AI Provider imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class AIProvider:
    """
    Unified AI provider that handles multiple AI services with fallback mechanisms.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Provider configurations
        self.providers = {
            "openai": {
                "available": openai is not None and config.openai_api_key,
                "config": config,
                "client": None
            },
            "anthropic": {
                "available": anthropic is not None and config.anthropic_api_key,
                "config": config,
                "client": None
            },
            "google": {
                "available": genai is not None and config.google_api_key,
                "config": config,
                "client": None
            },
            "local": {
                "available": True,  # Always available as fallback
                "config": config,
                "client": None
            }
        }
        
        # Provider priority order
        self.provider_priority = ["openai", "anthropic", "google", "local"]
        
        # Performance tracking
        self.provider_performance = {}
        self.fallback_count = 0
        
        # Cache for responses
        self.response_cache = {}
        
    async def initialize(self):
        """Initialize AI providers."""
        self.logger.info("Initializing AI providers...")
        
        # Initialize provider tracking
        self.providers = {
            "openai": {"available": False, "client": None},
            "anthropic": {"available": False, "client": None},
            "google": {"available": False, "client": None},
            "local": {"available": False, "client": None}
        }
        
        # Initialize client attributes
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.local_llm_client = None
        
        # Initialize OpenAI
        if self.config.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)
                self.providers["openai"]["available"] = True
                self.providers["openai"]["client"] = self.openai_client
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                
        # Initialize Anthropic
        if self.config.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
                self.providers["anthropic"]["available"] = True
                self.providers["anthropic"]["client"] = self.anthropic_client
                self.logger.info("Anthropic client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")
                
        # Initialize Google
        if self.config.google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.google_api_key)
                # Try Gemini 2.0 Flash Exp first, fallback to gemini-pro if not available
                try:
                    self.google_client = genai.GenerativeModel(self.config.google_model)
                    # Test the model
                    test_response = await asyncio.to_thread(
                        self.google_client.generate_content,
                        "test",
                        generation_config=genai.types.GenerationConfig(max_output_tokens=10)
                    )
                    self.logger.info(f"Google Gemini {self.config.google_model} client initialized successfully")
                except Exception as model_error:
                    self.logger.warning(f"Gemini 2.0 Flash Exp not available, falling back to gemini-pro: {model_error}")
                    self.google_client = genai.GenerativeModel('gemini-pro')
                    self.logger.info("Google Gemini Pro client initialized as fallback")
                
                self.providers["google"]["available"] = True
                self.providers["google"]["client"] = self.google_client
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google: {e}")
                
        # Initialize Local LLM
        try:
            await self._test_local_llm()
            self.providers["local"]["available"] = True
            self.providers["local"]["client"] = self.local_llm_client
            self.logger.info("Local LLM client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Local LLM: {e}")
            
        available_providers = [name for name, info in self.providers.items() if info["available"]]
        self.logger.info(f"AI providers initialized. Available: {available_providers}")
        
    async def _test_local_llm(self):
        """Test local LLM connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.local_llm_url}/v1/chat/completions",
                    json={
                        "model": self.config.local_llm_model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": "test"}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 10,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Local LLM returned status {response.status}")
                    
                    # Store the session for future use
                    self.local_llm_client = session
        except Exception as e:
            raise Exception(f"Local LLM connection failed: {e}")
            
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a response using available AI providers with intelligent fallback."""
        try:
            # Try available providers in order with fallback
            providers_to_try = []
            
            if self.openai_client:
                providers_to_try.append(("OpenAI", self._call_openai))
            if self.anthropic_client:
                providers_to_try.append(("Anthropic", self._call_anthropic))
            if self.google_client:
                providers_to_try.append(("Google Gemini", self._call_google))
            if self.local_llm_client:
                providers_to_try.append(("Local LLM", self._call_local_llm))
            
            # Try each provider in order
            for provider_name, provider_func in providers_to_try:
                try:
                    self.logger.info(f"Attempting to use {provider_name} for response generation")
                    response = await provider_func(prompt, max_tokens, temperature)
                    self.logger.info(f"Successfully generated response using {provider_name}")
                    return response
                except Exception as e:
                    self.logger.warning(f"{provider_name} failed: {e}, trying next provider...")
                    continue
            
            # If all providers failed, use fallback
            self.logger.warning("All AI providers failed, using fallback response")
            return self._generate_fallback_response(prompt)
                
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            return self._generate_fallback_response(prompt)
            
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when no AI providers are available."""
        try:
            # Enhanced keyword-based response generation for automation workflows
            prompt_lower = prompt.lower()
            
            # Automation workflow creation
            if any(word in prompt_lower for word in ["create", "build", "make", "set up"]) and any(word in prompt_lower for word in ["workflow", "automation", "process"]):
                return "I'd be happy to help you create an automation workflow! I can assist with various types of automation including web scraping, form filling, data extraction, and more. What specific task would you like to automate? Please provide details about the target website, actions needed, and expected outcomes."
            
            # Workflow status and monitoring
            elif any(word in prompt_lower for word in ["status", "progress", "monitor", "track"]) and any(word in prompt_lower for word in ["workflow", "automation", "process"]):
                return "I can help you monitor your automation workflows! The system provides real-time status updates, execution metrics, and performance analytics. Would you like me to show you the current workflow status or help you set up monitoring for a specific automation?"
            
            # Error handling and troubleshooting
            elif any(word in prompt_lower for word in ["error", "failed", "issue", "problem", "troubleshoot", "fix"]):
                return "I understand you're experiencing an issue with your automation. I can help you troubleshoot and resolve it! The platform includes advanced error detection, self-healing capabilities, and detailed logging. Could you share the specific error message or describe what's happening?"
            
            # Search and data gathering
            elif any(word in prompt_lower for word in ["search", "find", "gather", "collect", "data", "information"]):
                return "I can help you with search and data gathering tasks! The platform integrates with multiple search engines and can extract information from websites. What type of data are you looking for? I can help you set up automated data collection workflows."
            
            # Export and reporting
            elif any(word in prompt_lower for word in ["export", "report", "download", "save", "generate"]):
                return "I can help you export and generate reports! The platform supports multiple export formats including Excel, PDF, and JSON. You can also include screenshots and detailed execution logs. What type of report or export do you need?"
            
            # General help and assistance
            elif any(word in prompt_lower for word in ["help", "assist", "support", "guide", "how"]):
                return "I'm here to help you with all aspects of automation! I can assist with workflow creation, execution monitoring, error resolution, data extraction, and more. The platform is designed to handle complex automation tasks across various domains. What would you like to work on?"
            
            # Greeting and introduction
            elif any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
                return "Hello! I'm your AI automation assistant. I can help you create, manage, and optimize automation workflows for various tasks including web automation, data extraction, form filling, and more. What would you like to automate today?"
            
            # Default response for automation context
            else:
                return "I understand you're working with automation workflows. I can help you create, monitor, and optimize automation tasks. The platform supports web automation, data extraction, form filling, and more. What specific automation task would you like to work on? For enhanced AI capabilities, you can configure API keys for OpenAI, Anthropic, Google, or local LLM services."
                
        except Exception as e:
            return "I apologize, but I'm currently experiencing technical difficulties. I can still help you with automation workflows using the platform's built-in capabilities. What would you like to automate?"
        
    def _generate_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate cache key for response caching."""
        content = f"{prompt}:{max_tokens}:{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
        
    async def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI API."""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
            
    async def _call_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Anthropic API."""
        try:
            response = await asyncio.to_thread(
                self.providers["anthropic"]["client"].messages.create,
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
            
    async def _call_google(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Google Gemini API."""
        try:
            # Use the already initialized client
            response = await asyncio.to_thread(
                self.google_client.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Google Gemini API error: {e}")
            
    async def _call_local_llm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call local LLM API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.local_llm_url}/v1/chat/completions",
                    json={
                        "model": self.config.local_llm_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Local LLM returned status {response.status}")
                        
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
        except Exception as e:
            raise Exception(f"Local LLM API error: {e}")
            
    def _track_performance(self, provider: str, duration: float, success: bool):
        """Track provider performance metrics."""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration": 0,
                "avg_duration": 0
            }
            
        metrics = self.provider_performance[provider]
        metrics["total_calls"] += 1
        
        if success:
            metrics["successful_calls"] += 1
            metrics["total_duration"] += duration
            metrics["avg_duration"] = metrics["total_duration"] / metrics["successful_calls"]
        else:
            metrics["failed_calls"] += 1
            
    async def generate_structured_response(self, prompt: str, response_format: Dict[str, Any], 
                                        max_tokens: Optional[int] = None, 
                                        temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate structured response in specified format.
        
        Args:
            prompt: Input prompt
            response_format: Expected response format
            max_tokens: Maximum tokens to generate
            temperature: Response creativity (lower for structured responses)
            
        Returns:
            Structured response as dictionary
        """
        # Add format instructions to prompt
        format_instructions = f"""
        Please respond in the following JSON format:
        {json.dumps(response_format, indent=2)}
        
        Ensure your response is valid JSON and matches this structure exactly.
        """
        
        full_prompt = f"{prompt}\n\n{format_instructions}"
        
        # Generate response
        response_text = await self.generate_response(
            full_prompt, max_tokens, temperature
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
                
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            # Return fallback structured response
            return self._create_fallback_response(response_format, response_text)
            
    def _create_fallback_response(self, response_format: Dict[str, Any], 
                                response_text: str) -> Dict[str, Any]:
        """Create fallback response when JSON parsing fails."""
        fallback = {}
        
        for key, value_type in response_format.items():
            if isinstance(value_type, dict):
                fallback[key] = self._create_fallback_response(value_type, response_text)
            elif value_type == "string":
                fallback[key] = response_text[:100] if response_text else ""
            elif value_type == "number":
                fallback[key] = 0
            elif value_type == "boolean":
                fallback[key] = False
            elif value_type == "array":
                fallback[key] = []
            else:
                fallback[key] = None
                
        return fallback
        
    async def generate_embeddings(self, text: str, provider: Optional[str] = None) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text
            provider: Specific provider to use (optional)
            
        Returns:
            Embedding vector
        """
        # For now, use OpenAI embeddings as default
        if provider is None:
            provider = "openai"
            
        if provider == "openai" and self.providers["openai"]["available"]:
            try:
                response = await asyncio.to_thread(
                    openai.Embedding.create,
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                raise Exception(f"OpenAI embedding error: {e}")
        else:
            # Fallback to simple hash-based embedding
            return self._create_simple_embedding(text)
            
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create simple hash-based embedding as fallback."""
        import hashlib
        
        # Create hash of text
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to 1536-dimensional vector (same as OpenAI embeddings)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 1536:
                break
            embedding.append(float(int(hash_hex[i:i+2], 16)) / 255.0)
            
        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.append(0.0)
            
        return embedding[:1536]
        
    def get_best_provider(self) -> str:
        """Get the best performing provider based on metrics."""
        if not self.provider_performance:
            return self.provider_priority[0]
            
        # Calculate success rate and average duration for each provider
        provider_scores = {}
        
        for provider, metrics in self.provider_performance.items():
            if metrics["total_calls"] == 0:
                continue
                
            success_rate = metrics["successful_calls"] / metrics["total_calls"]
            avg_duration = metrics["avg_duration"]
            
            # Score based on success rate and speed (lower duration is better)
            score = success_rate * (1.0 / (1.0 + avg_duration))
            provider_scores[provider] = score
            
        if not provider_scores:
            return self.provider_priority[0]
            
        return max(provider_scores, key=provider_scores.get)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all providers."""
        return {
            "providers": self.provider_performance,
            "fallback_count": self.fallback_count,
            "best_provider": self.get_best_provider(),
            "available_providers": [p for p, config in self.providers.items() if config["available"]],
            "cache_size": len(self.response_cache)
        }
        
    async def shutdown(self):
        """Shutdown AI providers."""
        self.logger.info("Shutting down AI providers...")
        
        # Close any open connections
        for provider_name, provider_config in self.providers.items():
            if provider_config["client"]:
                # Most providers don't need explicit cleanup
                provider_config["client"] = None
                
        self.logger.info("AI providers shutdown complete")