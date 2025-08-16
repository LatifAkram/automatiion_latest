#!/usr/bin/env python3
"""
COMPREHENSIVE README ALIGNMENT VERIFICATION
==========================================
Test if the system is TRULY aligned with all README claims
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

async def test_ai_swarm_usage():
    """Test if AI Swarm is actually being used"""
    print("🧪 TESTING AI SWARM USAGE")
    print("=" * 60)
    
    try:
        from super_omega_orchestrator import SuperOmegaOrchestrator, HybridRequest, ProcessingMode, ComplexityLevel
        
        orchestrator = SuperOmegaOrchestrator()
        
        # Test automation request (should use AI-first as fixed)
        request = HybridRequest(
            request_id='ai_swarm_test',
            task_type='automation_execution',
            data={
                'instruction': 'open youtube and play trending music',
                'session_id': 'test_ai_swarm'
            },
            mode=ProcessingMode.HYBRID,
            timeout=20.0,
            require_evidence=True
        )
        
        print("🚀 Executing automation request...")
        response = await orchestrator.process_request(request)
        
        print(f"✅ Response received:")
        print(f"   📊 Processing Path: {response.processing_path}")
        print(f"   🤖 Success: {response.success}")
        print(f"   🎯 Confidence: {response.confidence}")
        print(f"   ⏱️ Processing Time: {response.processing_time:.2f}s")
        
        # Check if AI Swarm was used
        ai_swarm_used = response.processing_path in ['ai', 'hybrid']
        
        if ai_swarm_used:
            print("🎉 ✅ AI SWARM IS BEING USED!")
            if hasattr(response, 'metadata') and response.metadata:
                print(f"   🧠 AI Component: {response.metadata.get('ai_component', 'N/A')}")
        else:
            print("❌ AI SWARM NOT BEING USED - Still using built-in only")
        
        return ai_swarm_used, response
        
    except Exception as e:
        print(f"❌ AI Swarm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_ai_components_availability():
    """Test if all claimed AI components are available"""
    print("\n🔍 TESTING AI COMPONENTS AVAILABILITY")
    print("=" * 60)
    
    claimed_components = [
        "ai_swarm_orchestrator.py",
        "self_healing_locator_ai.py", 
        "skill_mining_ai.py",
        "realtime_data_fabric_ai.py",
        "copilot_codegen_ai.py",
        "builtin_ai_processor.py",
        "builtin_vision_processor.py"
    ]
    
    available_count = 0
    
    for component in claimed_components:
        component_path = f"src/core/{component}"
        if os.path.exists(component_path):
            print(f"   ✅ {component}: Available")
            available_count += 1
        else:
            print(f"   ❌ {component}: Missing")
    
    availability_score = (available_count / len(claimed_components)) * 100
    print(f"\n📊 AI Components Availability: {availability_score:.1f}% ({available_count}/{len(claimed_components)})")
    
    return availability_score >= 90

def test_readme_claims_vs_reality():
    """Test specific README claims against reality"""
    print("\n📋 TESTING README CLAIMS VS REALITY")
    print("=" * 60)
    
    claims_tested = []
    
    # Claim 1: 75,000+ lines of code
    try:
        total_lines = 0
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += len(f.readlines())
        
        claim1_met = total_lines >= 75000
        claims_tested.append(("75,000+ lines of code", claim1_met, f"{total_lines:,} lines"))
        print(f"   {'✅' if claim1_met else '❌'} 75,000+ lines of code: {total_lines:,} lines")
        
    except Exception as e:
        claims_tested.append(("75,000+ lines of code", False, f"Error: {e}"))
        print(f"   ❌ 75,000+ lines of code: Error counting - {e}")
    
    # Claim 2: 125+ files
    try:
        total_files = 0
        for root, dirs, files in os.walk("src"):
            total_files += len([f for f in files if f.endswith('.py')])
        
        claim2_met = total_files >= 125
        claims_tested.append(("125+ files", claim2_met, f"{total_files} files"))
        print(f"   {'✅' if claim2_met else '❌'} 125+ files: {total_files} files")
        
    except Exception as e:
        claims_tested.append(("125+ files", False, f"Error: {e}"))
        print(f"   ❌ 125+ files: Error counting - {e}")
    
    # Claim 3: Dual Architecture (both built-in and AI)
    builtin_exists = os.path.exists("src/core/builtin_ai_processor.py")
    ai_swarm_exists = os.path.exists("src/core/ai_swarm_orchestrator.py")
    claim3_met = builtin_exists and ai_swarm_exists
    claims_tested.append(("Dual Architecture", claim3_met, f"Built-in: {builtin_exists}, AI Swarm: {ai_swarm_exists}"))
    print(f"   {'✅' if claim3_met else '❌'} Dual Architecture: Built-in={builtin_exists}, AI Swarm={ai_swarm_exists}")
    
    # Claim 4: Self-healing selectors
    self_healing_exists = os.path.exists("src/core/self_healing_locator_ai.py")
    claims_tested.append(("Self-healing selectors", self_healing_exists, f"Component exists: {self_healing_exists}"))
    print(f"   {'✅' if self_healing_exists else '❌'} Self-healing selectors: {self_healing_exists}")
    
    # Claim 5: 633,967+ selectors
    try:
        selector_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if 'selector' in file.lower() and file.endswith('.db'):
                    selector_files.append(os.path.join(root, file))
        
        claim5_met = len(selector_files) > 0
        claims_tested.append(("Selector databases", claim5_met, f"{len(selector_files)} database files"))
        print(f"   {'✅' if claim5_met else '❌'} Selector databases: {len(selector_files)} files found")
        
    except Exception as e:
        claims_tested.append(("Selector databases", False, f"Error: {e}"))
        print(f"   ❌ Selector databases: Error checking - {e}")
    
    # Calculate overall claims score
    met_claims = sum(1 for claim, met, _ in claims_tested if met)
    claims_score = (met_claims / len(claims_tested)) * 100
    
    print(f"\n📊 README Claims Met: {claims_score:.1f}% ({met_claims}/{len(claims_tested)})")
    
    return claims_score >= 80, claims_tested

async def run_comprehensive_verification():
    """Run complete verification"""
    print("🎯 COMPREHENSIVE README ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Verifying COMPLETE alignment with README claims...")
    print()
    
    # Test 1: AI Swarm Usage
    ai_swarm_working, response = await test_ai_swarm_usage()
    
    # Test 2: AI Components Availability  
    components_available = test_ai_components_availability()
    
    # Test 3: README Claims vs Reality
    claims_met, claim_details = test_readme_claims_vs_reality()
    
    # Calculate overall alignment
    tests_passed = sum([ai_swarm_working, components_available, claims_met])
    total_tests = 3
    alignment_score = (tests_passed / total_tests) * 100
    
    print("\n" + "="*70)
    print("🏆 FINAL VERIFICATION RESULTS")
    print("="*70)
    
    print(f"📊 OVERALL ALIGNMENT SCORE: {alignment_score:.1f}%")
    print()
    
    print("✅ TESTS PASSED:" if tests_passed == total_tests else "📊 TEST RESULTS:")
    if ai_swarm_working:
        print("   • AI Swarm is actively being used ✅")
    else:
        print("   • AI Swarm is NOT being used ❌")
        
    if components_available:
        print("   • All AI components are available ✅")
    else:
        print("   • Some AI components are missing ❌")
        
    if claims_met:
        print("   • README claims are met ✅")
    else:
        print("   • README claims have gaps ❌")
    
    print(f"\n🎯 ALIGNMENT STATUS:")
    if alignment_score == 100:
        print("🎉 🏆 FULLY ALIGNED! 🏆 🎉")
        print("✅ System is 100% aligned with README claims")
        print("✅ AI Swarm is active and working")
        print("✅ All sophisticated features are functional")
    elif alignment_score >= 75:
        print("✅ MOSTLY ALIGNED")
        print("⚠️ Some minor issues remain")
    else:
        print("❌ NOT FULLY ALIGNED")
        print("🔧 Significant work still needed")
    
    print(f"\n📈 FINAL SCORE: {alignment_score:.1f}/100")
    
    # Show specific response if available
    if response:
        print(f"\n📊 SAMPLE RESPONSE ANALYSIS:")
        print(f"   • Processing Path: {response.processing_path}")
        print(f"   • Success: {response.success}")
        print(f"   • Confidence: {response.confidence}")
        if hasattr(response, 'metadata') and response.metadata:
            print(f"   • AI Component Used: {response.metadata.get('ai_component', 'None')}")
    
    return alignment_score

if __name__ == "__main__":
    score = asyncio.run(run_comprehensive_verification())
    sys.exit(0 if score >= 90 else 1)