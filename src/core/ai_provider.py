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
                    self.config.local_llm_url,
                    json={"prompt": "test", "max_tokens": 10},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Local LLM returned status {response.status}")
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
            # Simple keyword-based response generation
            prompt_lower = prompt.lower()
            
            if "workflow" in prompt_lower and "status" in prompt_lower:
                return "I can see you're asking about workflow status. The workflow is currently in the planning phase. I'll provide more detailed updates once the execution begins."
            elif "error" in prompt_lower or "failed" in prompt_lower:
                return "I understand there's an issue. Let me analyze the error and suggest a solution. Could you provide more details about what specific error you're encountering?"
            elif "help" in prompt_lower or "assist" in prompt_lower:
                return "I'm here to help! I can assist with workflow planning, execution monitoring, error resolution, and general automation guidance. What specific area do you need help with?"
            elif "optimize" in prompt_lower or "improve" in prompt_lower:
                return "I can help optimize your automation workflows. Based on the current patterns, I recommend focusing on error handling and performance monitoring for better results."
            else:
                return "I understand your request. I'm currently operating in fallback mode due to AI provider configuration. For full AI capabilities, please configure API keys for OpenAI, Anthropic, Google, or local LLM services."
                
        except Exception as e:
            return "I apologize, but I'm currently unable to provide a detailed response. Please check your AI provider configuration."
        
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
                    self.config.local_llm_url,
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "model": self.config.local_llm_model
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Local LLM returned status {response.status}")
                        
                    data = await response.json()
                    return data.get("response", data.get("text", ""))
                    
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