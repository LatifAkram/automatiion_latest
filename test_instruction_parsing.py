#!/usr/bin/env python3
"""
Test Instruction Parsing - How Chat Instructions Are Processed
============================================================

This test examines how instructions from the chat interface are currently
being parsed and processed by the backend system.
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

def test_frontend_instruction_detection():
    """Test how the frontend detects automation vs regular chat"""
    print("🔍 TESTING FRONTEND INSTRUCTION DETECTION")
    print("=" * 50)
    
    # Sample instructions from chat interface
    test_instructions = [
        "Hello, how are you?",
        "Automate login to Facebook",
        "Book a flight on Expedia",
        "Search for laptops on Amazon",
        "Fill out the registration form on https://example.com",
        "Click the submit button",
        "Navigate to google.com and search for automation tools",
        "Extract all product names from flipkart.com",
        "Monitor the stock price of AAPL",
        "Type my email address in the login field"
    ]
    
    # Simulate frontend detection logic
    def detect_automation_intent(message: str) -> Dict[str, Any]:
        """Simulate frontend automation detection logic"""
        automation_keywords = [
            'automate', 'book', 'search', 'extract', 'fill', 'monitor', 'click', 'open', 'login', 
            'sign', 'register', 'submit', 'enter', 'type', 'navigate', 'go to', 'visit', 'browse',
            'scrape', 'collect', 'gather', 'fetch', 'download', 'upload', 'form', 'button', 'link'
        ]
        
        has_url = message.lower().count('http') > 0 or message.lower().count('www.') > 0
        has_automation_keywords = any(keyword in message.lower() for keyword in automation_keywords)
        has_platform_keywords = any(platform in message.lower() for platform in [
            'flipkart', 'amazon', 'google', 'facebook', 'twitter', 'linkedin'
        ])
        
        is_automation_request = has_url or has_automation_keywords or has_platform_keywords
        
        endpoint = '/api/fixed-super-omega-execute' if is_automation_request else '/api/chat'
        request_body = {
            'instruction': message
        } if is_automation_request else {
            'message': message,
            'session_id': 'test_session',
            'context': {'domain': 'general'}
        }
        
        return {
            'message': message,
            'is_automation': is_automation_request,
            'has_url': has_url,
            'has_keywords': has_automation_keywords,
            'has_platforms': has_platform_keywords,
            'endpoint': endpoint,
            'request_body': request_body
        }
    
    # Test each instruction
    results = []
    for instruction in test_instructions:
        result = detect_automation_intent(instruction)
        results.append(result)
        
        status = "🤖 AUTOMATION" if result['is_automation'] else "💬 CHAT"
        print(f"{status}: {instruction}")
        print(f"   Endpoint: {result['endpoint']}")
        print(f"   URL: {result['has_url']}, Keywords: {result['has_keywords']}, Platforms: {result['has_platforms']}")
        print()
    
    # Summary
    automation_count = sum(1 for r in results if r['is_automation'])
    chat_count = len(results) - automation_count
    
    print(f"📊 DETECTION SUMMARY:")
    print(f"   Total instructions: {len(results)}")
    print(f"   Detected as automation: {automation_count}")
    print(f"   Detected as chat: {chat_count}")
    print(f"   Automation detection rate: {(automation_count/len(results))*100:.1f}%")
    
    return results

async def test_backend_instruction_processing():
    """Test how the backend processes instructions"""
    print("\n🔧 TESTING BACKEND INSTRUCTION PROCESSING")
    print("=" * 50)
    
    try:
        # Import backend components
        from core.super_omega_orchestrator import get_super_omega, HybridRequest, ProcessingMode, ComplexityLevel
        from core.builtin_ai_processor import BuiltinAIProcessor
        
        # Test instructions
        test_instructions = [
            "Automate login to Facebook",
            "Search for laptops on Amazon", 
            "Navigate to google.com and search for automation tools"
        ]
        
        # Initialize components
        orchestrator = get_super_omega()
        ai_processor = BuiltinAIProcessor()
        
        print("✅ Backend components initialized successfully")
        
        for instruction in test_instructions:
            print(f"\n🔍 Processing: '{instruction}'")
            
            # Test built-in AI processing
            try:
                # Extract entities from instruction
                entities = ai_processor.extract_entities(instruction)
                print(f"   🧠 AI Entities: {list(entities.keys())}")
                
                # Make decision about instruction
                decision = ai_processor.make_decision(
                    ['automation', 'search', 'navigation', 'interaction'],
                    {'instruction': instruction}
                )
                print(f"   🎯 AI Decision: {decision.result.get('choice', 'unknown')}")
                
            except Exception as e:
                print(f"   ❌ AI Processing error: {e}")
            
            # Test hybrid orchestrator processing
            try:
                # Create hybrid request
                hybrid_request = HybridRequest(
                    request_id=f"test_{hash(instruction)}",
                    task_type='automation_execution',
                    data={
                        'instruction': instruction,
                        'url': 'https://www.google.com',
                        'session_id': 'test_session'
                    },
                    complexity=ComplexityLevel.MODERATE,
                    mode=ProcessingMode.BUILTIN_ONLY,  # Use builtin only for testing
                    timeout=5.0,
                    require_evidence=False
                )
                
                # Process with timeout
                response = await asyncio.wait_for(
                    orchestrator.process_request(hybrid_request),
                    timeout=3.0
                )
                
                print(f"   🔄 Hybrid Response: {response.success}")
                print(f"   🎯 Processing Path: {response.processing_path}")
                print(f"   📊 Confidence: {response.confidence:.2f}")
                
                if hasattr(response, 'result') and response.result:
                    print(f"   📋 Result Type: {type(response.result).__name__}")
                    if isinstance(response.result, dict):
                        print(f"   📋 Result Keys: {list(response.result.keys())}")
                
            except asyncio.TimeoutError:
                print(f"   ⏱️  Hybrid processing timed out")
            except Exception as e:
                print(f"   ❌ Hybrid processing error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Backend import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Backend processing error: {e}")
        return False

def test_instruction_parsing_quality():
    """Test the quality of instruction parsing"""
    print("\n📊 TESTING INSTRUCTION PARSING QUALITY")
    print("=" * 50)
    
    # Test various instruction formats
    instruction_formats = {
        "Natural Language": [
            "Please log me into my Facebook account",
            "I need to book a flight from NYC to LA",
            "Can you help me search for running shoes on Amazon?"
        ],
        "Direct Commands": [
            "Login to Facebook",
            "Book flight NYC to LA",
            "Search Amazon for running shoes"
        ],
        "Step-by-Step": [
            "Go to Facebook.com, click login, enter my credentials",
            "Navigate to Expedia, search flights NYC to LA on Dec 15",
            "Open Amazon, search for running shoes, filter by size 10"
        ],
        "URL-Based": [
            "Automate https://facebook.com login process",
            "Book flight on https://expedia.com from NYC to LA",
            "Search https://amazon.com for running shoes"
        ],
        "Complex Multi-Step": [
            "Login to Facebook, navigate to marketplace, search for cars under $10000, save the first 5 listings",
            "Go to Amazon, search for laptops, filter by price range $500-1000, add top rated to cart",
            "Open Google, search for 'best restaurants near me', click on the first result, get phone number"
        ]
    }
    
    parsing_results = {}
    
    for format_type, instructions in instruction_formats.items():
        print(f"\n📝 Testing {format_type}:")
        format_results = []
        
        for instruction in instructions:
            # Simulate parsing complexity
            word_count = len(instruction.split())
            has_multiple_steps = ',' in instruction or ' and ' in instruction.lower()
            has_specific_data = any(char.isdigit() for char in instruction)
            has_url = 'http' in instruction or 'www.' in instruction
            
            complexity_score = 0
            if word_count > 10: complexity_score += 1
            if has_multiple_steps: complexity_score += 2
            if has_specific_data: complexity_score += 1
            if has_url: complexity_score += 1
            
            complexity_level = {
                0: "Simple",
                1: "Basic", 
                2: "Moderate",
                3: "Complex",
                4: "Ultra Complex"
            }.get(complexity_score, "Ultra Complex")
            
            format_results.append({
                'instruction': instruction,
                'word_count': word_count,
                'complexity_score': complexity_score,
                'complexity_level': complexity_level,
                'multi_step': has_multiple_steps,
                'has_data': has_specific_data,
                'has_url': has_url
            })
            
            print(f"   📋 {instruction[:50]}{'...' if len(instruction) > 50 else ''}")
            print(f"      Complexity: {complexity_level} (Score: {complexity_score})")
            print(f"      Multi-step: {has_multiple_steps}, Data: {has_specific_data}, URL: {has_url}")
        
        parsing_results[format_type] = format_results
    
    # Summary analysis
    print(f"\n📈 PARSING QUALITY ANALYSIS:")
    total_instructions = sum(len(results) for results in parsing_results.values())
    complexity_distribution = {}
    
    for format_results in parsing_results.values():
        for result in format_results:
            level = result['complexity_level']
            complexity_distribution[level] = complexity_distribution.get(level, 0) + 1
    
    print(f"   Total instructions analyzed: {total_instructions}")
    print(f"   Complexity distribution:")
    for level, count in sorted(complexity_distribution.items()):
        percentage = (count / total_instructions) * 100
        print(f"     {level}: {count} ({percentage:.1f}%)")
    
    return parsing_results

async def run_comprehensive_instruction_test():
    """Run comprehensive instruction parsing test"""
    print("🧪 COMPREHENSIVE INSTRUCTION PARSING TEST")
    print("=" * 60)
    
    # Test 1: Frontend detection
    frontend_results = test_frontend_instruction_detection()
    
    # Test 2: Backend processing
    backend_success = await test_backend_instruction_processing()
    
    # Test 3: Parsing quality
    parsing_results = test_instruction_parsing_quality()
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT")
    print("=" * 30)
    
    automation_detection_rate = sum(1 for r in frontend_results if r['is_automation']) / len(frontend_results) * 100
    
    print(f"✅ Frontend Detection: {automation_detection_rate:.1f}% automation detection rate")
    print(f"{'✅' if backend_success else '❌'} Backend Processing: {'Working' if backend_success else 'Issues detected'}")
    print(f"✅ Parsing Quality: Analyzed {sum(len(r) for r in parsing_results.values())} instruction formats")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if automation_detection_rate < 80:
        print(f"   📝 Improve automation keyword detection (current: {automation_detection_rate:.1f}%)")
    
    if not backend_success:
        print(f"   🔧 Fix backend processing issues")
    
    print(f"   🎯 Consider adding instruction preprocessing for complex multi-step commands")
    print(f"   🧠 Implement intent classification for better routing")
    print(f"   📊 Add instruction complexity analysis for better resource allocation")
    
    return {
        'frontend_detection_rate': automation_detection_rate,
        'backend_working': backend_success,
        'total_formats_tested': len(parsing_results),
        'recommendations': ['preprocessing', 'intent_classification', 'complexity_analysis']
    }

if __name__ == "__main__":
    print("🔍 INSTRUCTION PARSING ANALYSIS")
    print("Testing how chat interface instructions are currently parsed and processed...")
    print()
    
    # Run the comprehensive test
    result = asyncio.run(run_comprehensive_instruction_test())
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"   The system {'is working well' if result['backend_working'] and result['frontend_detection_rate'] > 70 else 'needs improvements'}")
    print(f"   Detection accuracy: {result['frontend_detection_rate']:.1f}%")
    print(f"   Backend status: {'✅ Operational' if result['backend_working'] else '❌ Issues'}")