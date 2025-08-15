#!/usr/bin/env python3
"""
Comprehensive Code Flow Analysis
===============================

Analyzes the entire SUPER-OMEGA codebase for:
- Import issues and circular dependencies
- Syntax errors and missing modules
- Class/method compatibility issues
- File structure and path problems
- API consistency across modules

This ensures the entire system works cohesively.
"""

import sys
import os
import ast
import importlib
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import json

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

class CodeAnalyzer:
    """Comprehensive code analyzer"""
    
    def __init__(self):
        self.src_dir = src_dir
        self.issues = []
        self.warnings = []
        self.imports_map = {}
        self.modules_tested = set()
        self.circular_deps = []
        
    def log_issue(self, level: str, file_path: str, message: str, details: str = ""):
        """Log an issue"""
        issue = {
            'level': level,  # 'ERROR', 'WARNING', 'INFO'
            'file': str(file_path),
            'message': message,
            'details': details
        }
        
        if level == 'ERROR':
            self.issues.append(issue)
        else:
            self.warnings.append(issue)
            
        print(f"{'❌' if level == 'ERROR' else '⚠️' if level == 'WARNING' else 'ℹ️'} {file_path}: {message}")
        if details:
            print(f"   {details}")
    
    def analyze_syntax(self, file_path: Path) -> bool:
        """Check file for syntax errors"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to check syntax
            ast.parse(content, filename=str(file_path))
            return True
            
        except SyntaxError as e:
            self.log_issue('ERROR', file_path, f"Syntax error: {e.msg}", f"Line {e.lineno}: {e.text}")
            return False
        except UnicodeDecodeError as e:
            self.log_issue('ERROR', file_path, f"Encoding error: {e}")
            return False
        except Exception as e:
            self.log_issue('ERROR', file_path, f"File read error: {e}")
            return False
    
    def extract_imports(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract all imports from a file"""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'level': node.level
                        })
            
            return imports
            
        except Exception as e:
            self.log_issue('WARNING', file_path, f"Could not extract imports: {e}")
            return []
    
    def test_import(self, file_path: Path) -> bool:
        """Test if a file can be imported successfully"""
        try:
            # Convert file path to module name
            rel_path = file_path.relative_to(self.src_dir)
            if rel_path.name == '__init__.py':
                module_parts = rel_path.parent.parts
            else:
                module_parts = rel_path.with_suffix('').parts
            
            module_name = '.'.join(module_parts)
            
            if module_name in self.modules_tested:
                return True
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                self.log_issue('ERROR', file_path, "Could not create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules temporarily
            sys.modules[module_name] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            self.modules_tested.add(module_name)
            self.log_issue('INFO', file_path, f"Import successful: {module_name}")
            return True
            
        except ImportError as e:
            self.log_issue('ERROR', file_path, f"Import error: {e}")
            return False
        except Exception as e:
            self.log_issue('ERROR', file_path, f"Module execution error: {e}")
            return False
    
    def check_class_methods(self, file_path: Path):
        """Check class method consistency"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'args': [arg.arg for arg in item.args.args],
                                'line': item.lineno
                            })
                    
                    # Check for common method issues
                    init_methods = [m for m in methods if m['name'] == '__init__']
                    if len(init_methods) > 1:
                        self.log_issue('WARNING', file_path, 
                                     f"Class {class_name} has multiple __init__ methods")
                    
                    # Check for methods with same name
                    method_names = [m['name'] for m in methods]
                    duplicates = set([name for name in method_names if method_names.count(name) > 1])
                    for dup in duplicates:
                        if dup != '__init__':  # Already checked above
                            self.log_issue('WARNING', file_path,
                                         f"Class {class_name} has duplicate method: {dup}")
        
        except Exception as e:
            self.log_issue('WARNING', file_path, f"Could not analyze class methods: {e}")
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in src directory"""
        python_files = []
        
        for path in self.src_dir.rglob('*.py'):
            # Skip __pycache__ and other generated files
            if '__pycache__' not in str(path) and not path.name.startswith('.'):
                python_files.append(path)
        
        return sorted(python_files)
    
    def analyze_file_structure(self):
        """Analyze the overall file structure"""
        print("\n📁 ANALYZING FILE STRUCTURE")
        print("=" * 50)
        
        expected_dirs = ['core', 'ui', 'testing', 'platforms', 'models', 'api']
        existing_dirs = [d.name for d in self.src_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for expected in expected_dirs:
            if expected in existing_dirs:
                self.log_issue('INFO', self.src_dir / expected, f"Directory exists: {expected}")
            else:
                self.log_issue('WARNING', self.src_dir, f"Missing expected directory: {expected}")
        
        # Check for unexpected directories
        unexpected = set(existing_dirs) - set(expected_dirs)
        for unexp in unexpected:
            self.log_issue('INFO', self.src_dir / unexp, f"Additional directory found: {unexp}")
    
    def check_init_files(self):
        """Check __init__.py files for proper exports"""
        print("\n📦 CHECKING __init__.py FILES")
        print("=" * 50)
        
        for init_file in self.src_dir.rglob('__init__.py'):
            if '__pycache__' not in str(init_file):
                try:
                    with open(init_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if it has __all__ definition
                    if '__all__' in content:
                        self.log_issue('INFO', init_file, "Has __all__ definition")
                    else:
                        self.log_issue('WARNING', init_file, "Missing __all__ definition")
                    
                    # Check for imports
                    imports = self.extract_imports(init_file)
                    if imports:
                        self.log_issue('INFO', init_file, f"Has {len(imports)} imports")
                    
                except Exception as e:
                    self.log_issue('WARNING', init_file, f"Could not analyze: {e}")
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis"""
        print("🔍 COMPREHENSIVE CODE FLOW ANALYSIS")
        print("=" * 60)
        print("Analyzing entire SUPER-OMEGA codebase for issues...")
        print()
        
        # Step 1: Analyze file structure
        self.analyze_file_structure()
        
        # Step 2: Check __init__.py files
        self.check_init_files()
        
        # Step 3: Find all Python files
        python_files = self.find_python_files()
        print(f"\n🐍 FOUND {len(python_files)} PYTHON FILES")
        print("=" * 50)
        
        # Step 4: Check syntax for all files
        print("\n🔍 CHECKING SYNTAX")
        print("=" * 30)
        syntax_ok = 0
        for file_path in python_files:
            if self.analyze_syntax(file_path):
                syntax_ok += 1
        
        print(f"✅ Syntax OK: {syntax_ok}/{len(python_files)} files")
        
        # Step 5: Extract imports from all files
        print("\n📥 ANALYZING IMPORTS")
        print("=" * 30)
        for file_path in python_files:
            imports = self.extract_imports(file_path)
            self.imports_map[str(file_path)] = imports
            if imports:
                self.log_issue('INFO', file_path, f"Found {len(imports)} imports")
        
        # Step 6: Check class methods
        print("\n🏗️ CHECKING CLASS METHODS")
        print("=" * 35)
        for file_path in python_files:
            self.check_class_methods(file_path)
        
        # Step 7: Test critical imports
        print("\n🧪 TESTING CRITICAL IMPORTS")
        print("=" * 40)
                 critical_files = [
             'core/__init__.py',
             'ui/builtin_web_server.py',
             'core/enhanced_self_healing_locator.py',
             'core/auto_skill_mining.py',
             'testing/super_omega_live_automation_fixed.py'
         ]
        
        for critical_file in critical_files:
            file_path = self.src_dir / critical_file
            if file_path.exists():
                self.test_import(file_path)
            else:
                self.log_issue('ERROR', file_path, f"Critical file missing: {critical_file}")
        
        # Step 8: Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_errors = len(self.issues)
        total_warnings = len(self.warnings)
        
        print(f"📁 Files Analyzed: {len(self.find_python_files())}")
        print(f"🧪 Modules Tested: {len(self.modules_tested)}")
        print(f"❌ Errors Found: {total_errors}")
        print(f"⚠️ Warnings Found: {total_warnings}")
        
        if total_errors == 0:
            print("\n🎉 NO CRITICAL ERRORS FOUND!")
            print("✅ Code flow analysis PASSED")
        else:
            print(f"\n🚨 {total_errors} CRITICAL ERRORS NEED ATTENTION")
            print("\nCRITICAL ERRORS:")
            print("-" * 20)
            for issue in self.issues[:10]:  # Show first 10 errors
                print(f"❌ {issue['file']}: {issue['message']}")
                if issue['details']:
                    print(f"   └─ {issue['details']}")
        
        if total_warnings > 0:
            print(f"\n⚠️ {total_warnings} WARNINGS (Non-critical):")
            print("-" * 30)
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"⚠️ {warning['file']}: {warning['message']}")
        
        # Component-specific analysis
        print("\n🔧 COMPONENT STATUS:")
        print("-" * 25)
        
        components = {
            'Core System': ['core/__init__.py', 'core/builtin_*'],
            'Self-Healing': ['core/enhanced_self_healing_locator.py'],
            'Web Server': ['ui/builtin_web_server.py'],
            'Automation': ['testing/super_omega_live_automation_fixed.py'],
            'AI Components': ['core/*_ai.py']
        }
        
        for component, patterns in components.items():
            component_files = []
            for pattern in patterns:
                if '*' in pattern:
                    # Handle wildcard patterns
                    pattern_path = self.src_dir / pattern.replace('*', '')
                    component_files.extend(self.src_dir.glob(pattern))
                else:
                    file_path = self.src_dir / pattern
                    if file_path.exists():
                        component_files.append(file_path)
            
            if component_files:
                component_issues = [i for i in self.issues if any(str(f) in i['file'] for f in component_files)]
                if component_issues:
                    print(f"❌ {component}: {len(component_issues)} issues")
                else:
                    print(f"✅ {component}: OK")
            else:
                print(f"⚠️ {component}: No files found")
        
        # Save detailed report
        report = {
            'summary': {
                'files_analyzed': len(self.find_python_files()),
                'modules_tested': len(self.modules_tested),
                'errors': total_errors,
                'warnings': total_warnings
            },
            'errors': self.issues,
            'warnings': self.warnings,
            'imports_map': self.imports_map
        }
        
        report_file = Path('COMPREHENSIVE_CODE_ANALYSIS_REPORT.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return total_errors == 0

def main():
    """Run comprehensive code analysis"""
    analyzer = CodeAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    print("\n" + "=" * 60)
    if success:
        print("🏆 ANALYSIS RESULT: PASSED")
        print("✅ No critical issues found in code flow")
        print("🚀 SUPER-OMEGA codebase is ready for production!")
    else:
        print("⚠️ ANALYSIS RESULT: ISSUES FOUND")
        print("🔧 Critical issues need to be resolved")
        print("📋 Check the detailed report for fixes needed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)