#!/usr/bin/env python3
"""
TEST CORRECTED THREE ARCHITECTURE FLOW
=====================================

Test the corrected flow:
Frontend → Backend → Autonomous Orchestrator → Intent Analysis → Task Scheduling → Multi-Architecture Execution → Result Aggregation
"""

import requests
import json
import time

def test_three_architecture_flow():
    """Test the corrected three architecture flow"""
    
    print("🧪 TESTING CORRECTED THREE ARCHITECTURE FLOW")
    print("=" * 60)
    
    # Test different types of instructions to verify proper flow
    test_cases = [
        {
            'name': 'Simple Navigation (Should use Built-in Foundation)',
            'instruction': 'open youtube.com',
            'expected_primary_arch': 'builtin_foundation'
        },
        {
            'name': 'Intelligent YouTube Engagement (Should use AI Swarm)',
            'instruction': 'go to youtube and like the first video and subscribe to the channel',
            'expected_primary_arch': 'ai_swarm'
        },
        {
            'name': 'Complex Automation (Should use Autonomous Layer)',
            'instruction': 'automate a multi-step workflow to process data',
            'expected_primary_arch': 'autonomous_layer'
        }
    ]
    
    backend_url = "http://localhost:8888"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test_case['name']}")
        print(f"   📝 Instruction: {test_case['instruction']}")
        
        try:
            # Send request to backend
            response = requests.post(
                f"{backend_url}/api/execute",
                json={
                    'instruction': test_case['instruction'],
                    'priority': 'NORMAL'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if it went through the proper flow
                print(f"   ✅ Response received")
                print(f"   🏗️ Architecture used: {result.get('architecture_used', 'unknown')}")
                
                # Check for flow trace
                if 'flow_trace' in result:
                    flow_trace = result['flow_trace']
                    print(f"   🔄 Flow trace available:")
                    print(f"      🧠 Step 1 - Orchestrator: {flow_trace.get('step_1_orchestrator', {}).get('orchestrator_decision', 'unknown')}")
                    print(f"      🤖 Step 2 - Intent Analysis: {flow_trace.get('step_2_intent_analysis', {}).get('primary_intent', 'unknown')}")
                    print(f"      📅 Step 3 - Task Scheduling: {flow_trace.get('step_3_task_scheduling', {}).get('scheduler_status', 'unknown')}")
                    print(f"      ⚡ Step 4 - Execution: {len(flow_trace.get('step_4_execution', {}).get('architectures_attempted', []))} architectures attempted")
                    print(f"      📊 Step 5 - Aggregation: {flow_trace.get('step_5_aggregation', {}).get('overall_success', 'unknown')}")
                    
                    # Verify proper three architecture flow
                    if result.get('architecture_used') == 'three_architecture_orchestrated':
                        print(f"   ✅ PROPER THREE ARCHITECTURE FLOW CONFIRMED")
                    else:
                        print(f"   ❌ STILL USING SINGLE ARCHITECTURE: {result.get('architecture_used')}")
                else:
                    print(f"   ❌ NO FLOW TRACE - NOT USING THREE ARCHITECTURE FLOW")
                
                # Show execution details
                if 'result' in result and isinstance(result['result'], dict):
                    final_result = result['result']
                    if 'architectures_used' in final_result:
                        print(f"   🏗️ Architectures used: {final_result['architectures_used']}")
                    if 'quality_score' in final_result:
                        print(f"   📊 Quality score: {final_result['quality_score']}")
                
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed - backend not running on {backend_url}")
            return False
        except Exception as e:
            print(f"   💥 Test failed: {str(e)}")
            return False
        
        time.sleep(1)  # Brief pause between tests
    
    print(f"\n✅ THREE ARCHITECTURE FLOW TESTING COMPLETED")
    return True

if __name__ == "__main__":
    test_three_architecture_flow()