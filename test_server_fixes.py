#!/usr/bin/env python3
"""
Quick test to verify server fixes
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'core'))

def test_enhanced_parser_import():
    """Test enhanced parser import"""
    print("🧪 Testing enhanced parser import...")
    try:
        from enhanced_instruction_parser import parse_instruction_enhanced
        print("✅ Enhanced parser import successful")
        
        # Quick test
        result = parse_instruction_enhanced("Login to Facebook")
        print(f"✅ Enhanced parser working: {result.instruction_type.value}")
        return True
    except Exception as e:
        print(f"❌ Enhanced parser import failed: {e}")
        return False

def test_builtin_performance_monitor():
    """Test builtin performance monitor import"""
    print("\n🧪 Testing builtin performance monitor import...")
    try:
        from builtin_performance_monitor import get_system_metrics
        print("✅ Performance monitor import successful")
        
        # Quick test
        metrics = get_system_metrics()
        print(f"✅ Performance monitor working: {type(metrics).__name__}")
        return True
    except Exception as e:
        print(f"❌ Performance monitor import failed: {e}")
        return False

def test_super_omega_imports():
    """Test super omega orchestrator imports"""
    print("\n🧪 Testing super omega imports...")
    try:
        from super_omega_orchestrator import get_super_omega
        print("✅ Super omega import successful")
        
        # Quick test
        orchestrator = get_super_omega()
        print(f"✅ Super omega working: {type(orchestrator).__name__}")
        return True
    except Exception as e:
        print(f"❌ Super omega import failed: {e}")
        return False

def test_web_server_imports():
    """Test web server imports"""
    print("\n🧪 Testing web server imports...")
    try:
        from ui.builtin_web_server import BuiltinWebServer
        print("✅ Web server import successful")
        return True
    except Exception as e:
        print(f"❌ Web server import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TESTING SERVER FIXES")
    print("=" * 30)
    
    results = []
    results.append(test_enhanced_parser_import())
    results.append(test_builtin_performance_monitor())
    results.append(test_super_omega_imports())
    results.append(test_web_server_imports())
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 RESULTS: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All fixes working! Server should start properly now.")
    else:
        print("⚠️  Some issues remain. Check the errors above.")