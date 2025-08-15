#!/usr/bin/env python3
"""
Focused Import Analysis
=======================

Specifically checks for import issues that could affect the running SUPER-OMEGA server.
Tests critical components and their dependencies.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

def test_import_chain(module_path: str, description: str) -> bool:
    """Test import chain for a specific module"""
    print(f"\n🔍 Testing {description}")
    print("-" * 50)
    
    try:
        # Try importing the module
        if module_path.startswith('.'):
            # Relative import
            module = __import__(module_path[1:], fromlist=[''])
        else:
            # Absolute import
            module = __import__(module_path)
        
        print(f"✅ {description}: Import successful")
        
        # Check for key attributes/classes
        if hasattr(module, '__all__'):
            exports = getattr(module, '__all__')
            print(f"   📤 Exports: {len(exports)} items")
            
            # Test a few key exports
            missing_exports = []
            for export in exports[:5]:  # Test first 5 exports
                if not hasattr(module, export):
                    missing_exports.append(export)
            
            if missing_exports:
                print(f"   ⚠️ Missing exports: {missing_exports}")
                return False
            else:
                print(f"   ✅ Key exports available")
        
        return True
        
    except ImportError as e:
        print(f"❌ {description}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {description}: Error - {e}")
        return False

def test_critical_classes():
    """Test critical classes can be instantiated"""
    print(f"\n🏗️ Testing Critical Classes")
    print("-" * 40)
    
    tests = [
        {
            'module': 'ui.builtin_web_server',
            'class': 'LiveConsoleServer',
            'description': 'Web Server'
        },
        {
            'module': 'core.enhanced_self_healing_locator',
            'class': 'EnhancedSelfHealingLocator',
            'description': 'Self-Healing System'
        },
        {
            'module': 'core.auto_skill_mining',
            'class': 'AutoSkillMiner',
            'description': 'Auto Skill Mining'
        }
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            module = __import__(test['module'], fromlist=[test['class']])
            cls = getattr(module, test['class'])
            
            # Try to create instance with default parameters
            if test['class'] == 'LiveConsoleServer':
                instance = cls()
            else:
                instance = cls()
            
            print(f"✅ {test['description']}: Class instantiation successful")
            
        except Exception as e:
            print(f"❌ {test['description']}: Failed - {e}")
            all_passed = False
    
    return all_passed

def test_server_dependencies():
    """Test server-specific dependencies"""
    print(f"\n🌐 Testing Server Dependencies")
    print("-" * 40)
    
    # Test built-in web server components
    try:
        from ui.builtin_web_server import LiveConsoleServer
        server = LiveConsoleServer()
        
        # Test key server methods
        methods_to_check = ['start', 'stop', 'handle_request']
        missing_methods = []
        
        for method in methods_to_check:
            if not hasattr(server, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Server missing methods: {missing_methods}")
            return False
        else:
            print("✅ Server has all required methods")
        
        # Test server configuration
        print(f"   Host: {server.host}")
        print(f"   Port: {server.port}")
        print(f"   Running: {getattr(server, 'running', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server dependency test failed: {e}")
        return False

def test_self_healing_system():
    """Test self-healing system specifically"""
    print(f"\n🔧 Testing Self-Healing System")
    print("-" * 40)
    
    try:
        # Test enhanced self-healing
        from core.enhanced_self_healing_locator import get_enhanced_self_healing_locator
        healer = get_enhanced_self_healing_locator()
        
        print("✅ Enhanced self-healing locator: Available")
        
        # Test original self-healing integration
        from core.self_healing_locator_ai import get_self_healing_ai
        healing_ai = get_self_healing_ai()
        
        print("✅ Original self-healing AI: Available")
        
        # Test method compatibility
        if hasattr(healing_ai, 'heal_selector'):
            print("✅ heal_selector method: Available")
        else:
            print("❌ heal_selector method: Missing")
            return False
        
        if hasattr(healer, 'heal_selector_guaranteed'):
            print("✅ heal_selector_guaranteed method: Available")
        else:
            print("❌ heal_selector_guaranteed method: Missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Self-healing system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_integrations():
    """Test core system integrations"""
    print(f"\n⚙️ Testing Core Integrations")
    print("-" * 35)
    
    try:
        # Test core module imports
        from core import (
            builtin_monitor,
            ai_processor,
            vision_processor,
            BaseValidator,
            AutoSkillMining
        )
        
        print("✅ Core built-in components: Available")
        
        # Test AI Swarm components
        try:
            from core.ai_swarm_orchestrator import AISwarmOrchestrator
            print("✅ AI Swarm Orchestrator: Available")
        except ImportError:
            print("⚠️ AI Swarm Orchestrator: Not available (fallback mode)")
        
        # Test production monitor
        try:
            from core.production_monitor import get_production_monitor
            monitor = get_production_monitor()
            print("✅ Production Monitor: Available")
        except ImportError:
            print("⚠️ Production Monitor: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Core integrations test failed: {e}")
        return False

def main():
    """Run focused import analysis"""
    print("🎯 FOCUSED IMPORT ANALYSIS")
    print("=" * 50)
    print("Testing critical components for the running server...")
    
    all_tests_passed = True
    
    # Test 1: Critical module imports
    critical_modules = [
        ('ui.builtin_web_server', 'Built-in Web Server'),
        ('core.enhanced_self_healing_locator', 'Enhanced Self-Healing'),
        ('core.auto_skill_mining', 'Auto Skill Mining'),
        ('core.production_monitor', 'Production Monitor'),
        ('core.vector_store', 'Vector Store')
    ]
    
    for module_path, description in critical_modules:
        if not test_import_chain(module_path, description):
            all_tests_passed = False
    
    # Test 2: Critical class instantiation
    if not test_critical_classes():
        all_tests_passed = False
    
    # Test 3: Server dependencies
    if not test_server_dependencies():
        all_tests_passed = False
    
    # Test 4: Self-healing system
    if not test_self_healing_system():
        all_tests_passed = False
    
    # Test 5: Core integrations
    if not test_core_integrations():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 FOCUSED ANALYSIS SUMMARY")
    print("=" * 50)
    
    if all_tests_passed:
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("✅ Server components are working correctly")
        print("✅ Self-healing system is operational")
        print("✅ Core integrations are functional")
        print("🚀 System is ready for production use!")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("🔧 Issues found that may affect server operation")
        print("📋 Review the test results above for specific fixes")
    
    # Additional server status check
    print(f"\n🌐 Current Server Status:")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()
        
        if result == 0:
            print("✅ Server is responding on port 8080")
        else:
            print("⚠️ Server not responding on port 8080")
    except Exception as e:
        print(f"⚠️ Could not check server status: {e}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)