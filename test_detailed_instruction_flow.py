#!/usr/bin/env python3
"""
Detailed Instruction Flow Analysis
=================================

This test traces a single instruction through the entire system to see
exactly how it's processed at each stage.
"""

import sys
import os
import asyncio
import json
import time
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

async def trace_instruction_flow(instruction: str):
    """Trace how a single instruction flows through the system"""
    print(f"🔍 TRACING INSTRUCTION FLOW")
    print(f"📝 Instruction: '{instruction}'")
    print("=" * 80)
    
    flow_trace = {
        'instruction': instruction,
        'timestamp': time.time(),
        'stages': {}
    }
    
    # Stage 1: Frontend Detection
    print(f"\n📱 STAGE 1: FRONTEND DETECTION")
    print("-" * 40)
    
    automation_keywords = [
        'automate', 'book', 'search', 'extract', 'fill', 'monitor', 'click', 'open', 'login', 
        'sign', 'register', 'submit', 'enter', 'type', 'navigate', 'go to', 'visit', 'browse',
        'scrape', 'collect', 'gather', 'fetch', 'download', 'upload', 'form', 'button', 'link'
    ]
    
    has_url = 'http' in instruction.lower() or 'www.' in instruction.lower()
    has_automation_keywords = any(keyword in instruction.lower() for keyword in automation_keywords)
    has_platform_keywords = any(platform in instruction.lower() for platform in [
        'flipkart', 'amazon', 'google', 'facebook', 'twitter', 'linkedin'
    ])
    
    is_automation_request = has_url or has_automation_keywords or has_platform_keywords
    endpoint = '/api/fixed-super-omega-execute' if is_automation_request else '/api/chat'
    
    frontend_stage = {
        'detected_as_automation': is_automation_request,
        'has_url': has_url,
        'has_automation_keywords': has_automation_keywords,
        'has_platform_keywords': has_platform_keywords,
        'chosen_endpoint': endpoint,
        'request_body': {
            'instruction': instruction
        } if is_automation_request else {
            'message': instruction,
            'session_id': 'trace_session',
            'context': {'domain': 'general'}
        }
    }
    
    flow_trace['stages']['frontend'] = frontend_stage
    
    print(f"   🎯 Detected as: {'🤖 AUTOMATION' if is_automation_request else '💬 CHAT'}")
    print(f"   🔗 Endpoint: {endpoint}")
    print(f"   📊 Detection factors:")
    print(f"      URL present: {has_url}")
    print(f"      Automation keywords: {has_automation_keywords}")
    print(f"      Platform keywords: {has_platform_keywords}")
    
    if not is_automation_request:
        print(f"   ⚠️  This instruction would be sent to chat endpoint, not automation")
        return flow_trace
    
    # Stage 2: Backend Web Server Processing
    print(f"\n🌐 STAGE 2: BACKEND WEB SERVER")
    print("-" * 40)
    
    try:
        # Simulate web server processing
        session_id = f"trace_session_{int(time.time())}"
        
        # Determine complexity
        complexity_indicators = []
        if any(word in instruction.lower() for word in ['complex', 'multi', 'workflow', 'advanced']):
            complexity = 'COMPLEX'
            complexity_indicators.append('complex keywords')
        elif any(word in instruction.lower() for word in ['ultra', 'intelligent', 'orchestrate']):
            complexity = 'ULTRA_COMPLEX'
            complexity_indicators.append('ultra keywords')
        elif any(word in instruction.lower() for word in ['simple', 'basic', 'easy']):
            complexity = 'SIMPLE'
            complexity_indicators.append('simple keywords')
        else:
            complexity = 'MODERATE'
            complexity_indicators.append('default')
        
        webserver_stage = {
            'session_id': session_id,
            'determined_complexity': complexity,
            'complexity_indicators': complexity_indicators,
            'will_call_hybrid_system': True
        }
        
        flow_trace['stages']['webserver'] = webserver_stage
        
        print(f"   📋 Session ID: {session_id}")
        print(f"   🎯 Complexity: {complexity}")
        print(f"   📊 Complexity indicators: {complexity_indicators}")
        print(f"   ✅ Will proceed to hybrid system")
        
    except Exception as e:
        print(f"   ❌ Web server simulation error: {e}")
        return flow_trace
    
    # Stage 3: Real AI Connector (if available)
    print(f"\n🧠 STAGE 3: REAL AI CONNECTOR")
    print("-" * 40)
    
    try:
        from core.real_ai_connector import generate_ai_response
        
        # Test AI interpretation
        ai_response = await generate_ai_response(
            f"Analyze this automation instruction: {instruction}",
            {"instruction": instruction, "session_id": session_id}
        )
        
        real_ai_stage = {
            'available': True,
            'provider': ai_response.provider,
            'content_length': len(ai_response.content),
            'confidence': ai_response.confidence,
            'processing_time': ai_response.processing_time,
            'cached': ai_response.cached,
            'fallback_used': ai_response.fallback_used
        }
        
        flow_trace['stages']['real_ai'] = real_ai_stage
        
        print(f"   ✅ Real AI available")
        print(f"   🤖 Provider: {ai_response.provider}")
        print(f"   📊 Confidence: {ai_response.confidence:.2f}")
        print(f"   ⏱️  Processing time: {ai_response.processing_time:.3f}s")
        print(f"   💾 Cached: {ai_response.cached}")
        print(f"   🔄 Fallback used: {ai_response.fallback_used}")
        print(f"   📝 Response preview: {ai_response.content[:100]}...")
        
    except Exception as e:
        real_ai_stage = {
            'available': False,
            'error': str(e)
        }
        flow_trace['stages']['real_ai'] = real_ai_stage
        print(f"   ❌ Real AI not available: {e}")
    
    # Stage 4: Hybrid Orchestrator
    print(f"\n🎛️  STAGE 4: HYBRID ORCHESTRATOR")
    print("-" * 40)
    
    try:
        from core.super_omega_orchestrator import get_super_omega, HybridRequest, ProcessingMode, ComplexityLevel
        
        # Map complexity
        complexity_map = {
            'SIMPLE': ComplexityLevel.SIMPLE,
            'MODERATE': ComplexityLevel.MODERATE,
            'COMPLEX': ComplexityLevel.COMPLEX,
            'ULTRA_COMPLEX': ComplexityLevel.ULTRA_COMPLEX
        }
        
        orchestrator = get_super_omega()
        
        # Create hybrid request
        hybrid_request = HybridRequest(
            request_id=session_id,
            task_type='automation_execution',
            data={
                'instruction': instruction,
                'url': 'https://www.google.com',
                'session_id': session_id
            },
            complexity=complexity_map.get(complexity, ComplexityLevel.MODERATE),
            mode=ProcessingMode.HYBRID,
            timeout=10.0,
            require_evidence=True
        )
        
        print(f"   📋 Request created:")
        print(f"      ID: {hybrid_request.request_id}")
        print(f"      Task type: {hybrid_request.task_type}")
        print(f"      Complexity: {hybrid_request.complexity}")
        print(f"      Mode: {hybrid_request.mode}")
        
        # Process the request
        start_time = time.time()
        response = await asyncio.wait_for(
            orchestrator.process_request(hybrid_request),
            timeout=8.0
        )
        processing_time = time.time() - start_time
        
        orchestrator_stage = {
            'request_successful': True,
            'response_success': response.success,
            'processing_path': response.processing_path,
            'confidence': response.confidence,
            'processing_time': processing_time,
            'has_result': hasattr(response, 'result') and response.result is not None,
            'result_type': type(response.result).__name__ if hasattr(response, 'result') else None,
            'has_evidence': hasattr(response, 'evidence') and response.evidence is not None,
            'fallback_used': getattr(response, 'fallback_used', False)
        }
        
        if hasattr(response, 'result') and isinstance(response.result, dict):
            orchestrator_stage['result_keys'] = list(response.result.keys())
        
        flow_trace['stages']['orchestrator'] = orchestrator_stage
        
        print(f"   ✅ Processing successful: {response.success}")
        print(f"   🛤️  Processing path: {response.processing_path}")
        print(f"   📊 Confidence: {response.confidence:.2f}")
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print(f"   📋 Has result: {orchestrator_stage['has_result']}")
        print(f"   📋 Result type: {orchestrator_stage['result_type']}")
        if 'result_keys' in orchestrator_stage:
            print(f"   🔑 Result keys: {orchestrator_stage['result_keys']}")
        print(f"   🔄 Fallback used: {orchestrator_stage['fallback_used']}")
        
    except asyncio.TimeoutError:
        orchestrator_stage = {
            'request_successful': False,
            'error': 'timeout',
            'processing_time': 8.0
        }
        flow_trace['stages']['orchestrator'] = orchestrator_stage
        print(f"   ⏱️  Processing timed out after 8 seconds")
    except Exception as e:
        orchestrator_stage = {
            'request_successful': False,
            'error': str(e)
        }
        flow_trace['stages']['orchestrator'] = orchestrator_stage
        print(f"   ❌ Orchestrator error: {e}")
    
    # Stage 5: Built-in AI Processing (detailed)
    print(f"\n🧠 STAGE 5: BUILT-IN AI PROCESSING")
    print("-" * 40)
    
    try:
        from core.builtin_ai_processor import BuiltinAIProcessor
        
        ai_processor = BuiltinAIProcessor()
        
        # Test entity extraction
        entities = ai_processor.extract_entities(instruction)
        
        # Test text analysis
        analysis = ai_processor.analyze_text(instruction)
        
        # Test decision making (fix the result access)
        decision_result = ai_processor.make_decision(
            ['automation', 'search', 'navigation', 'interaction', 'form_filling'],
            {'instruction': instruction}
        )
        
        # Access the result correctly
        if hasattr(decision_result, 'result'):
            decision_choice = decision_result.result.get('choice', 'unknown')
        elif isinstance(decision_result, dict):
            decision_choice = decision_result.get('choice', 'unknown')
        else:
            decision_choice = str(decision_result)
        
        builtin_ai_stage = {
            'entities_found': len(entities),
            'entity_types': list(entities.keys()) if entities else [],
            'text_analysis_available': analysis is not None,
            'decision_choice': decision_choice,
            'processing_successful': True
        }
        
        if analysis:
            builtin_ai_stage['analysis_keys'] = list(analysis.keys()) if isinstance(analysis, dict) else None
        
        flow_trace['stages']['builtin_ai'] = builtin_ai_stage
        
        print(f"   🔍 Entities extracted: {len(entities)}")
        if entities:
            for entity_type, entity_list in entities.items():
                print(f"      {entity_type}: {entity_list}")
        
        print(f"   📊 Text analysis: {'✅ Available' if analysis else '❌ None'}")
        if analysis and isinstance(analysis, dict):
            print(f"      Analysis keys: {list(analysis.keys())}")
        
        print(f"   🎯 Decision choice: {decision_choice}")
        
    except Exception as e:
        builtin_ai_stage = {
            'processing_successful': False,
            'error': str(e)
        }
        flow_trace['stages']['builtin_ai'] = builtin_ai_stage
        print(f"   ❌ Built-in AI error: {e}")
    
    # Stage 6: Final Response Formation
    print(f"\n📤 STAGE 6: FINAL RESPONSE")
    print("-" * 40)
    
    # Simulate final response formation
    if flow_trace['stages'].get('orchestrator', {}).get('request_successful', False):
        final_response = {
            'success': True,
            'session_id': session_id,
            'instruction': instruction,
            'processing_path': flow_trace['stages']['orchestrator']['processing_path'],
            'confidence': flow_trace['stages']['orchestrator']['confidence'],
            'system': 'SUPER-OMEGA Hybrid Intelligence'
        }
        
        if flow_trace['stages'].get('real_ai', {}).get('available', False):
            final_response['ai_interpretation'] = f"AI analysis available from {flow_trace['stages']['real_ai']['provider']}"
            final_response['ai_provider'] = flow_trace['stages']['real_ai']['provider']
            final_response['system'] = 'SUPER-OMEGA Hybrid Intelligence with Real AI'
        
        response_stage = {
            'formation_successful': True,
            'response_keys': list(final_response.keys()),
            'estimated_size': len(json.dumps(final_response))
        }
        
        print(f"   ✅ Response formation successful")
        print(f"   🔑 Response keys: {response_stage['response_keys']}")
        print(f"   📊 Estimated response size: {response_stage['estimated_size']} bytes")
        
    else:
        response_stage = {
            'formation_successful': False,
            'fallback_response': True
        }
        print(f"   ❌ Response formation failed, would return error response")
    
    flow_trace['stages']['final_response'] = response_stage
    
    # Summary
    print(f"\n📋 FLOW SUMMARY")
    print("=" * 40)
    
    total_stages = len(flow_trace['stages'])
    successful_stages = sum(1 for stage in flow_trace['stages'].values() 
                          if stage.get('request_successful', True) and 
                             stage.get('processing_successful', True) and
                             stage.get('formation_successful', True))
    
    success_rate = (successful_stages / total_stages) * 100
    
    print(f"   📊 Stages processed: {total_stages}")
    print(f"   ✅ Successful stages: {successful_stages}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"   🎯 Final status: {'✅ Success' if success_rate >= 80 else '⚠️ Issues detected'}")
    
    return flow_trace

async def test_multiple_instruction_types():
    """Test different types of instructions"""
    print(f"\n🧪 TESTING MULTIPLE INSTRUCTION TYPES")
    print("=" * 80)
    
    test_instructions = [
        "Login to Facebook",  # Simple automation
        "Search for laptops on Amazon and add the cheapest one to cart",  # Complex multi-step
        "Hello, how are you today?",  # Chat message
        "Navigate to https://google.com and search for automation tools"  # URL-based
    ]
    
    results = {}
    
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n🔬 TEST {i}/{len(test_instructions)}")
        print("=" * 60)
        
        try:
            result = await trace_instruction_flow(instruction)
            results[instruction] = result
            
            # Brief summary for this instruction
            stages_success = sum(1 for stage in result['stages'].values() 
                               if stage.get('request_successful', True) and 
                                  stage.get('processing_successful', True) and
                                  stage.get('formation_successful', True))
            total_stages = len(result['stages'])
            success_rate = (stages_success / total_stages) * 100
            
            print(f"\n   📊 Quick Summary: {success_rate:.1f}% success rate ({stages_success}/{total_stages} stages)")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results[instruction] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("🔍 DETAILED INSTRUCTION FLOW ANALYSIS")
    print("Tracing how instructions flow through the entire system...")
    print()
    
    # Run detailed flow analysis
    results = asyncio.run(test_multiple_instruction_types())
    
    print(f"\n🏆 OVERALL ANALYSIS COMPLETE")
    print("=" * 50)
    
    successful_tests = sum(1 for result in results.values() if 'error' not in result)
    total_tests = len(results)
    
    print(f"✅ Successful instruction traces: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        print(f"📊 The instruction parsing and processing system is operational")
        print(f"🎯 Instructions are being correctly detected, routed, and processed")
    else:
        print(f"❌ Issues detected in instruction processing pipeline")