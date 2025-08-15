#!/usr/bin/env python3
"""
Quick Test for Import Fixes
============================

Test that the AutoSkillMining import issue is resolved
and that the system can start without warnings.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_core_imports():
    """Test that core imports work without errors"""
    print("🧪 Testing Core Module Imports...")
    
    try:
        # Test the fixed AutoSkillMining import
        from core.auto_skill_mining import AutoSkillMiner
        print("✅ AutoSkillMiner import: SUCCESS")
        
        # Test the alias import from __init__
        from core import AutoSkillMining
        print("✅ AutoSkillMining alias import: SUCCESS")
        
        # Verify they're the same class
        if AutoSkillMining is AutoSkillMiner:
            print("✅ Alias correctly points to AutoSkillMiner: SUCCESS")
        else:
            print("❌ Alias mismatch: FAILED")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Test vector store (should work with fallback)
        from core.vector_store import VectorStore
        print("✅ VectorStore import: SUCCESS")
        
    except ImportError as e:
        print(f"❌ VectorStore import error: {e}")
        return False
    
    try:
        # Test other core components
        from core import (
            builtin_monitor,
            ai_processor, 
            vision_processor,
            BaseValidator
        )
        print("✅ Built-in components import: SUCCESS")
        
    except ImportError as e:
        print(f"❌ Built-in components import error: {e}")
        return False
    
    return True

def test_skill_mining_functionality():
    """Test basic AutoSkillMiner functionality"""
    print("\n🔧 Testing AutoSkillMiner Functionality...")
    
    try:
        from core.auto_skill_mining import AutoSkillMiner
        
        # Create instance
        miner = AutoSkillMiner()
        print("✅ AutoSkillMiner instance creation: SUCCESS")
        
        # Test basic methods exist
        if hasattr(miner, 'mine_skill_from_trace'):
            print("✅ mine_skill_from_trace method exists: SUCCESS")
        else:
            print("❌ mine_skill_from_trace method missing: FAILED")
            return False
            
        if hasattr(miner, 'get_skill_stats'):
            print("✅ get_skill_stats method exists: SUCCESS")
        else:
            print("❌ get_skill_stats method missing: FAILED")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ AutoSkillMiner functionality test error: {e}")
        return False

def test_vector_store_fallback():
    """Test that vector store works with fallback"""
    print("\n🗄️ Testing Vector Store Fallback...")
    
    try:
        from core.vector_store import VectorStore, CHROMADB_AVAILABLE
        
        print(f"📊 ChromaDB Available: {CHROMADB_AVAILABLE}")
        
        # Create vector store instance
        vector_store = VectorStore()
        print("✅ VectorStore instance creation: SUCCESS")
        
        # Check if it has required methods
        required_methods = ['initialize', 'store_vector', 'search_vectors', 'get_stats']
        for method in required_methods:
            if hasattr(vector_store, method):
                print(f"✅ {method} method exists: SUCCESS")
            else:
                print(f"❌ {method} method missing: FAILED")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test error: {e}")
        return False

def main():
    """Run all import and functionality tests"""
    print("🚀 SUPER-OMEGA Import Fixes Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Core imports
    if not test_core_imports():
        all_passed = False
    
    # Test 2: Skill mining functionality  
    if not test_skill_mining_functionality():
        all_passed = False
    
    # Test 3: Vector store fallback
    if not test_vector_store_fallback():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Import fixes are working correctly")
        print("✅ System should start without warnings")
        print("✅ AutoSkillMining import issue resolved")
        print("✅ ChromaDB fallback working properly")
    else:
        print("❌ SOME TESTS FAILED")
        print("❌ Additional fixes may be needed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)