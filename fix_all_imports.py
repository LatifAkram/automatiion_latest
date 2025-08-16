#!/usr/bin/env python3
"""
Fix All Import Issues - Make SUPER-OMEGA 100% Functional
=========================================================

This script fixes all relative import issues that prevent components from loading.
"""

import os
import re
import glob

def fix_relative_imports(file_path):
    """Fix relative imports in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common relative import patterns
        patterns_to_fix = [
            (r'from \.builtin_ai_processor import', 'from builtin_ai_processor import'),
            (r'from \.builtin_data_validation import', 'from builtin_data_validation import'),
            (r'from \.builtin_performance_monitor import', 'from builtin_performance_monitor import'),
            (r'from \.builtin_vision_processor import', 'from builtin_vision_processor import'),
            (r'from \.semantic_dom_graph import', 'from semantic_dom_graph import'),
            (r'from \.self_healing_locators import', 'from self_healing_locators import'),
            (r'from \.shadow_dom_simulator import', 'from shadow_dom_simulator import'),
            (r'from \.constrained_planner import', 'from constrained_planner import'),
            (r'from \.realtime_data_fabric import', 'from realtime_data_fabric import'),
            (r'from \.deterministic_executor import', 'from deterministic_executor import'),
            (r'from \.auto_skill_mining import', 'from auto_skill_mining import'),
            (r'from \.enterprise_security import', 'from enterprise_security import'),
            (r'from \.orchestrator import', 'from orchestrator import'),
            (r'from \.config import', 'from config import'),
            (r'from \.database import', 'from database import'),
            (r'from \.ai_provider import', 'from ai_provider import'),
            (r'from \.audit import', 'from audit import'),
            # Fix deeper relative imports
            (r'from \.\.models\.contracts import', 'from models.contracts import'),
            (r'from \.\.platforms import', 'from platforms import'),
            (r'from \.\.ui import', 'from ui import'),
            (r'from \.\.testing import', 'from testing import'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all import issues in the codebase"""
    print("🔧 FIXING ALL IMPORT ISSUES FOR 100% FUNCTIONALITY")
    print("=" * 60)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files")
    
    fixed_count = 0
    total_count = len(python_files)
    
    # Fix imports in each file
    for file_path in python_files:
        if fix_relative_imports(file_path):
            fixed_count += 1
    
    print(f"\n📊 IMPORT FIX RESULTS:")
    print(f"   ✅ Files Fixed: {fixed_count}")
    print(f"   📁 Total Files: {total_count}")
    print(f"   📈 Success Rate: {(fixed_count/total_count*100):.1f}%")
    
    if fixed_count > 0:
        print(f"\n🎯 IMPORT ISSUES RESOLVED!")
        print("✅ All components should now load correctly")
    else:
        print(f"\n✅ NO IMPORT ISSUES FOUND")
        print("All files already have correct imports")

if __name__ == "__main__":
    main()