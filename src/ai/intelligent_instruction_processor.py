#!/usr/bin/env python3
"""
INTELLIGENT INSTRUCTION PROCESSOR
=================================

Real AI integration for understanding random user instructions and converting them
to actionable automation steps. Uses multiple AI providers with fallbacks.
"""

import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class InstructionType(Enum):
    ECOMMERCE_SEARCH = "ecommerce_search"
    ECOMMERCE_PURCHASE = "ecommerce_purchase" 
    WEB_NAVIGATION = "web_navigation"
    DATA_EXTRACTION = "data_extraction"
    SOCIAL_MEDIA = "social_media"
    AUTOMATION_WORKFLOW = "automation_workflow"
    UNKNOWN = "unknown"

@dataclass
class ProcessedInstruction:
    original_instruction: str
    instruction_type: InstructionType
    platform: Optional[str] = None
    action: Optional[str] = None
    target_item: Optional[str] = None
    parameters: Dict[str, Any] = None
    confidence: float = 0.0
    automation_steps: List[Dict[str, Any]] = None
    ai_reasoning: str = ""

class IntelligentInstructionProcessor:
    """Real AI-powered instruction processor for random user inputs"""
    
    def __init__(self):
        self.ai_providers = self._initialize_ai_providers()
        self.platform_patterns = self._load_platform_patterns()
        self.action_patterns = self._load_action_patterns()
    
    def _initialize_ai_providers(self) -> Dict[str, Any]:
        """Initialize real AI providers"""
        providers = {}
        
        # Try to load real AI providers
        try:
            # OpenAI GPT
            import openai
            providers['openai'] = {
                'client': openai,
                'available': True,
                'model': 'gpt-4'
            }
        except ImportError:
            providers['openai'] = {'available': False}
        
        try:
            # Anthropic Claude
            import anthropic
            providers['anthropic'] = {
                'client': anthropic,
                'available': True,
                'model': 'claude-3-sonnet-20240229'
            }
        except ImportError:
            providers['anthropic'] = {'available': False}
        
        try:
            # Local LLM (Ollama)
            import requests
            # Test if Ollama is running
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                providers['ollama'] = {
                    'available': True,
                    'endpoint': 'http://localhost:11434/api/generate',
                    'model': 'llama2'
                }
        except:
            providers['ollama'] = {'available': False}
        
        return providers
    
    def _load_platform_patterns(self) -> Dict[str, List[str]]:
        """Load platform detection patterns"""
        return {
            'flipkart': ['flipkart', 'flipkart.com'],
            'amazon': ['amazon', 'amazon.com', 'amazon.in'],
            'ebay': ['ebay', 'ebay.com'],
            'myntra': ['myntra', 'myntra.com'],
            'snapdeal': ['snapdeal', 'snapdeal.com'],
            'youtube': ['youtube', 'youtube.com'],
            'facebook': ['facebook', 'facebook.com', 'fb'],
            'instagram': ['instagram', 'instagram.com', 'insta'],
            'twitter': ['twitter', 'twitter.com', 'x.com'],
            'linkedin': ['linkedin', 'linkedin.com']
        }
    
    def _load_action_patterns(self) -> Dict[str, List[str]]:
        """Load action detection patterns"""
        return {
            'search': ['search', 'find', 'look for', 'browse', 'explore'],
            'buy': ['buy', 'purchase', 'checkout', 'order', 'add to cart'],
            'compare': ['compare', 'check prices', 'price comparison', 'best deal'],
            'navigate': ['open', 'go to', 'visit', 'navigate to'],
            'interact': ['like', 'share', 'comment', 'subscribe', 'follow'],
            'extract': ['extract', 'get data', 'scrape', 'collect information'],
            'automate': ['automate', 'workflow', 'process', 'execute sequence']
        }
    
    async def process_instruction(self, instruction: str) -> ProcessedInstruction:
        """Process random user instruction using real AI"""
        
        print(f"🧠 AI Processing instruction: {instruction}")
        
        # Step 1: Try real AI processing first
        ai_result = await self._process_with_ai(instruction)
        
        if ai_result and ai_result.confidence > 0.7:
            print(f"✅ AI processing successful (confidence: {ai_result.confidence})")
            return ai_result
        
        # Step 2: Fallback to pattern matching
        print("🔄 AI confidence low, using pattern matching fallback")
        pattern_result = self._process_with_patterns(instruction)
        
        # Step 3: Combine AI insights with pattern matching
        if ai_result and pattern_result:
            return self._merge_results(ai_result, pattern_result)
        
        return pattern_result or self._create_unknown_instruction(instruction)
    
    async def _process_with_ai(self, instruction: str) -> Optional[ProcessedInstruction]:
        """Process instruction using real AI/LLM"""
        
        prompt = self._create_ai_prompt(instruction)
        
        # Try each AI provider
        for provider_name, provider in self.ai_providers.items():
            if not provider.get('available', False):
                continue
                
            try:
                print(f"🤖 Trying {provider_name} for instruction processing...")
                
                if provider_name == 'openai':
                    return await self._process_with_openai(instruction, prompt, provider)
                elif provider_name == 'anthropic':
                    return await self._process_with_anthropic(instruction, prompt, provider)
                elif provider_name == 'ollama':
                    return await self._process_with_ollama(instruction, prompt, provider)
                    
            except Exception as e:
                print(f"❌ {provider_name} failed: {str(e)}")
                continue
        
        print("❌ All AI providers failed")
        return None
    
    def _create_ai_prompt(self, instruction: str) -> str:
        """Create AI prompt for instruction analysis"""
        return f"""
Analyze this user instruction for web automation: "{instruction}"

Extract the following information in JSON format:
{{
    "platform": "detected platform (flipkart, amazon, youtube, etc.)",
    "action": "main action (search, buy, navigate, etc.)",
    "target_item": "what item/content to target",
    "instruction_type": "ecommerce_search|ecommerce_purchase|web_navigation|social_media|automation_workflow",
    "parameters": {{"key": "value pairs of additional parameters"}},
    "automation_steps": [
        {{"step": 1, "action": "navigate", "target": "website"}},
        {{"step": 2, "action": "search", "query": "item"}},
        {{"step": 3, "action": "click", "selector": "button"}}
    ],
    "confidence": 0.95,
    "reasoning": "explanation of analysis"
}}

Focus on creating actionable automation steps that can be executed by Playwright.
"""
    
    async def _process_with_openai(self, instruction: str, prompt: str, provider: Dict) -> Optional[ProcessedInstruction]:
        """Process with OpenAI GPT"""
        # Implementation would go here with real OpenAI API calls
        # For now, return None to use fallback
        return None
    
    async def _process_with_anthropic(self, instruction: str, prompt: str, provider: Dict) -> Optional[ProcessedInstruction]:
        """Process with Anthropic Claude"""
        # Implementation would go here with real Anthropic API calls
        # For now, return None to use fallback
        return None
    
    async def _process_with_ollama(self, instruction: str, prompt: str, provider: Dict) -> Optional[ProcessedInstruction]:
        """Process with local Ollama LLM"""
        try:
            import requests
            
            payload = {
                "model": provider['model'],
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(provider['endpoint'], json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                # Parse the LLM response and create ProcessedInstruction
                return self._parse_ai_response(instruction, result.get('response', ''))
        except Exception as e:
            print(f"Ollama processing failed: {e}")
            return None
    
    def _parse_ai_response(self, instruction: str, ai_response: str) -> Optional[ProcessedInstruction]:
        """Parse AI response into ProcessedInstruction"""
        try:
            # Try to extract JSON from AI response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                return ProcessedInstruction(
                    original_instruction=instruction,
                    instruction_type=InstructionType(parsed.get('instruction_type', 'unknown')),
                    platform=parsed.get('platform'),
                    action=parsed.get('action'),
                    target_item=parsed.get('target_item'),
                    parameters=parsed.get('parameters', {}),
                    confidence=parsed.get('confidence', 0.8),
                    automation_steps=parsed.get('automation_steps', []),
                    ai_reasoning=parsed.get('reasoning', '')
                )
        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            return None
    
    def _process_with_patterns(self, instruction: str) -> Optional[ProcessedInstruction]:
        """Fallback processing using pattern matching"""
        
        instruction_lower = instruction.lower()
        
        # Detect platform
        platform = None
        for plat, patterns in self.platform_patterns.items():
            if any(pattern in instruction_lower for pattern in patterns):
                platform = plat
                break
        
        # Detect action
        action = None
        for act, patterns in self.action_patterns.items():
            if any(pattern in instruction_lower for pattern in patterns):
                action = act
                break
        
        # Detect target item
        target_item = self._extract_target_item(instruction_lower)
        
        # Determine instruction type
        instruction_type = self._determine_instruction_type(platform, action, instruction_lower)
        
        # Create automation steps
        automation_steps = self._create_automation_steps(platform, action, target_item, instruction)
        
        return ProcessedInstruction(
            original_instruction=instruction,
            instruction_type=instruction_type,
            platform=platform,
            action=action,
            target_item=target_item,
            parameters={'source': 'pattern_matching'},
            confidence=0.75,
            automation_steps=automation_steps,
            ai_reasoning="Pattern matching analysis"
        )
    
    def _extract_target_item(self, instruction_lower: str) -> Optional[str]:
        """Extract target item from instruction"""
        # Common product patterns
        product_patterns = [
            r'iphone\s*\d+\s*pro?',
            r'samsung\s*galaxy\s*\w+',
            r'laptop',
            r'mobile',
            r'phone',
            r'tablet',
            r'headphones',
            r'shoes',
            r'shirt',
            r'book'
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                return match.group()
        
        return None
    
    def _determine_instruction_type(self, platform: str, action: str, instruction_lower: str) -> InstructionType:
        """Determine the type of instruction"""
        
        if platform in ['flipkart', 'amazon', 'ebay', 'myntra', 'snapdeal']:
            if action in ['buy', 'checkout', 'order']:
                return InstructionType.ECOMMERCE_PURCHASE
            elif action in ['search', 'find', 'compare']:
                return InstructionType.ECOMMERCE_SEARCH
        
        if platform in ['youtube', 'facebook', 'instagram', 'twitter']:
            return InstructionType.SOCIAL_MEDIA
        
        if action in ['automate', 'workflow']:
            return InstructionType.AUTOMATION_WORKFLOW
        
        if action in ['navigate', 'open']:
            return InstructionType.WEB_NAVIGATION
        
        if 'extract' in instruction_lower or 'data' in instruction_lower:
            return InstructionType.DATA_EXTRACTION
        
        return InstructionType.UNKNOWN
    
    def _create_automation_steps(self, platform: str, action: str, target_item: str, instruction: str) -> List[Dict[str, Any]]:
        """Create automation steps based on analysis"""
        
        steps = []
        
        if platform == 'flipkart' and action in ['search', 'buy']:
            steps = [
                {"step": 1, "action": "navigate", "target": "https://www.flipkart.com"},
                {"step": 2, "action": "search", "query": target_item or "product"},
                {"step": 3, "action": "select_product", "criteria": "first_result"},
            ]
            
            if action == 'buy':
                steps.extend([
                    {"step": 4, "action": "click", "target": "buy_now_button"},
                    {"step": 5, "action": "proceed_checkout", "target": "checkout_flow"}
                ])
        
        elif platform == 'amazon' and action in ['search', 'buy']:
            steps = [
                {"step": 1, "action": "navigate", "target": "https://www.amazon.com"},
                {"step": 2, "action": "search", "query": target_item or "product"},
                {"step": 3, "action": "select_product", "criteria": "first_result"},
            ]
            
            if action == 'buy':
                steps.extend([
                    {"step": 4, "action": "click", "target": "add_to_cart_button"},
                    {"step": 5, "action": "proceed_checkout", "target": "cart_checkout"}
                ])
        
        elif platform == 'youtube':
            steps = [
                {"step": 1, "action": "navigate", "target": "https://www.youtube.com"},
                {"step": 2, "action": "search", "query": target_item or "video"},
                {"step": 3, "action": "play_video", "target": "first_result"}
            ]
        
        else:
            # Generic steps
            steps = [
                {"step": 1, "action": "navigate", "target": f"https://www.{platform}.com" if platform else "https://www.google.com"},
                {"step": 2, "action": "interact", "instruction": instruction}
            ]
        
        return steps
    
    def _merge_results(self, ai_result: ProcessedInstruction, pattern_result: ProcessedInstruction) -> ProcessedInstruction:
        """Merge AI and pattern matching results"""
        
        # Use AI result as base, fill gaps with pattern matching
        merged = ProcessedInstruction(
            original_instruction=ai_result.original_instruction,
            instruction_type=ai_result.instruction_type or pattern_result.instruction_type,
            platform=ai_result.platform or pattern_result.platform,
            action=ai_result.action or pattern_result.action,
            target_item=ai_result.target_item or pattern_result.target_item,
            parameters={**pattern_result.parameters, **ai_result.parameters},
            confidence=max(ai_result.confidence, pattern_result.confidence),
            automation_steps=ai_result.automation_steps or pattern_result.automation_steps,
            ai_reasoning=f"AI: {ai_result.ai_reasoning} | Pattern: {pattern_result.ai_reasoning}"
        )
        
        return merged
    
    def _create_unknown_instruction(self, instruction: str) -> ProcessedInstruction:
        """Create result for unknown instructions"""
        return ProcessedInstruction(
            original_instruction=instruction,
            instruction_type=InstructionType.UNKNOWN,
            confidence=0.3,
            automation_steps=[
                {"step": 1, "action": "search_google", "query": instruction}
            ],
            ai_reasoning="Could not analyze instruction, defaulting to Google search"
        )

# Global instance
intelligent_processor = IntelligentInstructionProcessor()