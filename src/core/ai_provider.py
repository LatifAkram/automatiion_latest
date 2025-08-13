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
        
    async def initialize(self):
        """Initialize AI providers."""
        self.logger.info("Initializing AI providers...")
        
        # Initialize OpenAI
        if self.providers["openai"]["available"]:
            try:
                openai.api_key = self.config.openai_api_key
                self.providers["openai"]["client"] = openai
                self.logger.info("OpenAI provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                self.providers["openai"]["available"] = False
                
        # Initialize Anthropic
        if self.providers["anthropic"]["available"]:
            try:
                self.providers["anthropic"]["client"] = anthropic.Anthropic(
                    api_key=self.config.anthropic_api_key
                )
                self.logger.info("Anthropic provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")
                self.providers["anthropic"]["available"] = False
                
        # Initialize Google Gemini
        if self.providers["google"]["available"]:
            try:
                genai.configure(api_key=self.config.google_api_key)
                self.providers["google"]["client"] = genai
                self.logger.info("Google Gemini provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google Gemini: {e}")
                self.providers["google"]["available"] = False
                
        # Initialize Local LLM
        if self.providers["local"]["available"]:
            try:
                # Test local LLM connection
                await self._test_local_llm()
                self.logger.info("Local LLM provider initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Local LLM: {e}")
                self.providers["local"]["available"] = False
                
        self.logger.info(f"AI providers initialized. Available: {[p for p, config in self.providers.items() if config['available']]}")
        
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
            
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None, 
                              temperature: float = 0.7, provider: Optional[str] = None) -> str:
        """
        Generate response using the best available AI provider.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Response creativity (0.0 to 1.0)
            provider: Specific provider to use (optional)
            
        Returns:
            Generated response text
        """
        if max_tokens is None:
            max_tokens = self.config.openai_max_tokens
            
        # Use specified provider or determine best available
        if provider and self.providers[provider]["available"]:
            providers_to_try = [provider]
        else:
            providers_to_try = [p for p in self.provider_priority if self.providers[p]["available"]]
            
        if not providers_to_try:
            raise Exception("No AI providers available")
            
        # Try providers in order with fallback
        last_error = None
        
        for provider_name in providers_to_try:
            try:
                start_time = asyncio.get_event_loop().time()
                
                if provider_name == "openai":
                    response = await self._call_openai(prompt, max_tokens, temperature)
                elif provider_name == "anthropic":
                    response = await self._call_anthropic(prompt, max_tokens, temperature)
                elif provider_name == "google":
                    response = await self._call_google(prompt, max_tokens, temperature)
                elif provider_name == "local":
                    response = await self._call_local_llm(prompt, max_tokens, temperature)
                else:
                    continue
                    
                # Track performance
                duration = asyncio.get_event_loop().time() - start_time
                self._track_performance(provider_name, duration, True)
                
                self.logger.info(f"Generated response using {provider_name}")
                return response
                
            except Exception as e:
                last_error = e
                self._track_performance(provider_name, 0, False)
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                continue
                
        # All providers failed
        self.fallback_count += 1
        raise Exception(f"All AI providers failed. Last error: {last_error}")
        
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
            model = genai.GenerativeModel(self.config.google_model)
            response = await asyncio.to_thread(
                model.generate_content,
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
            "available_providers": [p for p, config in self.providers.items() if config["available"]]
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