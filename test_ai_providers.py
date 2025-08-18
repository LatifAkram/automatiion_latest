#!/usr/bin/env python3
"""
Test AI Providers Integration
============================

Test the specific Gemini and Local LLM endpoints you provided.
"""

import asyncio
import aiohttp
import json

async def test_gemini_api():
    """Test your specific Gemini API endpoint"""
    
    print("🧪 Testing Gemini API...")
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyBb-AFGtxM2biSnESY85nyk-fdR74O153c"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Analyze this automation instruction: 'open flipkart and checkout iphone 14 pro with least price'. Provide platform, action, and automation steps in JSON format."
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        print("✅ Gemini API - WORKING")
                        print(f"   📝 Response: {content[:200]}...")
                        return True
                else:
                    error_text = await response.text()
                    print(f"❌ Gemini API - HTTP {response.status}: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Gemini API - FAILED: {e}")
        return False

async def test_local_llm():
    """Test your specific Local LLM endpoint"""
    
    print("\n🧪 Testing Local LLM (Vision)...")
    
    try:
        payload = {
            "model": "qwen2-vl-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are an expert automation assistant with vision capabilities."},
                {"role": "user", "content": "Analyze this automation instruction: 'open flipkart and checkout iphone 14 pro with least price'. Provide platform, action, and automation steps."}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        print("✅ Local LLM Vision - WORKING")
                        print(f"   📝 Response: {content[:200]}...")
                        return True
                else:
                    error_text = await response.text()
                    print(f"❌ Local LLM - HTTP {response.status}: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Local LLM - FAILED: {e}")
        return False

async def test_ai_integration():
    """Test AI integration in the system"""
    
    print("\n🧪 Testing AI Integration in System...")
    
    try:
        import sys
        from pathlib import Path
        
        # Add paths
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir / 'src' / 'core'))
        
        # Import AI provider
        from ai_provider import AIProvider
        
        # Create a minimal config
        class Config:
            def __init__(self):
                self.openai_api_key = None
                self.anthropic_api_key = None
                self.google_api_key = None
                self.local_llm_url = "http://localhost:1234"
                self.local_llm_model = "qwen2-vl-7b-instruct"
        
        config = Config()
        ai_provider = AIProvider(config)
        
        # Test response generation
        response = await ai_provider.generate_response(
            "Analyze this instruction: 'open flipkart and buy iphone 14 pro'. What platform and actions are needed?"
        )
        
        print("✅ AI Provider Integration - WORKING")
        print(f"   📝 Response: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ AI Provider Integration - FAILED: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 TESTING AI PROVIDERS INTEGRATION")
    print("=" * 60)
    
    gemini_works = await test_gemini_api()
    local_llm_works = await test_local_llm()
    integration_works = await test_ai_integration()
    
    print(f"\n📊 RESULTS:")
    print(f"   Gemini API: {'✅ WORKING' if gemini_works else '❌ FAILED'}")
    print(f"   Local LLM Vision: {'✅ WORKING' if local_llm_works else '❌ FAILED'}")
    print(f"   AI Integration: {'✅ WORKING' if integration_works else '❌ FAILED'}")
    
    if gemini_works or local_llm_works:
        print(f"\n🎯 AI PROVIDERS READY FOR REAL-TIME DATA PROCESSING")
    else:
        print(f"\n⚠️ AI PROVIDERS NEED SETUP")

if __name__ == "__main__":
    asyncio.run(main())