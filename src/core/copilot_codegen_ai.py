#!/usr/bin/env python3
"""
Copilot/Codegen AI - Intelligent Code Generation & Automation
============================================================

AI-powered system that auto-generates:
- Postconditions and preconditions
- Fallback strategies and error handling
- Wait predicates and timing logic
- Unit tests and validation code
- Drift-repair patches and fixes
- Complete automation scripts

Accelerates automation development with intelligent code generation.
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
import ast
import inspect
from textwrap import dedent

# AI imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    np = None

# LLM provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Built-in fallbacks
from builtin_ai_processor import BuiltinAIProcessor
from builtin_data_validation import BaseValidator

logger = logging.getLogger(__name__)

class CodeType(Enum):
    """Types of code that can be generated"""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    FALLBACK_STRATEGY = "fallback_strategy"
    WAIT_PREDICATE = "wait_predicate"
    ERROR_HANDLER = "error_handler"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    VALIDATION_CODE = "validation_code"
    REPAIR_PATCH = "repair_patch"
    AUTOMATION_SCRIPT = "automation_script"

class FrameworkType(Enum):
    """Supported automation frameworks"""
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    CYPRESS = "cypress"
    PUPPETEER = "puppeteer"
    GENERIC = "generic"

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"

@dataclass
class CodeGenerationRequest:
    """Request for code generation"""
    request_id: str
    code_type: CodeType
    framework: FrameworkType
    language: LanguageType
    context: Dict[str, Any]
    requirements: List[str]
    existing_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedCode:
    """Generated code result"""
    request_id: str
    code_type: CodeType
    framework: FrameworkType
    language: LanguageType
    generated_code: str
    explanation: str
    confidence: float
    dependencies: List[str]
    test_cases: List[str]
    validation_notes: List[str]
    generation_time_ms: float
    used_ai: bool
    provider: Optional[str] = None

class CodeTemplateEngine:
    """Template engine for code generation patterns"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load code templates for different scenarios"""
        return {
            CodeType.PRECONDITION.value: {
                LanguageType.PYTHON.value: '''
def check_{condition_name}(context: Dict[str, Any]) -> bool:
    """Check if {condition_description}"""
    try:
        {condition_logic}
        return True
    except Exception as e:
        logger.warning(f"Precondition check failed: {{e}}")
        return False
''',
                LanguageType.JAVASCRIPT.value: '''
function check{ConditionName}(context) {
    // Check if {condition_description}
    try {
        {condition_logic}
        return true;
    } catch (error) {
        console.warn(`Precondition check failed: ${error.message}`);
        return false;
    }
}
''',
                LanguageType.TYPESCRIPT.value: '''
function check{ConditionName}(context: any): boolean {
    // Check if {condition_description}
    try {
        {condition_logic}
        return true;
    } catch (error) {
        console.warn(`Precondition check failed: ${error.message}`);
        return false;
    }
}
'''
            },
            CodeType.POSTCONDITION.value: {
                LanguageType.PYTHON.value: '''
def verify_{condition_name}(context: Dict[str, Any], result: Any) -> bool:
    """Verify that {condition_description}"""
    try:
        {verification_logic}
        return True
    except Exception as e:
        logger.error(f"Postcondition verification failed: {{e}}")
        return False
''',
                LanguageType.JAVASCRIPT.value: '''
function verify{ConditionName}(context, result) {
    // Verify that {condition_description}
    try {
        {verification_logic}
        return true;
    } catch (error) {
        console.error(`Postcondition verification failed: ${error.message}`);
        return false;
    }
}
'''
            },
            CodeType.WAIT_PREDICATE.value: {
                LanguageType.PYTHON.value: '''
async def wait_for_{condition_name}(page, timeout_ms: int = 30000) -> bool:
    """Wait until {condition_description}"""
    start_time = time.time()
    
    while (time.time() - start_time) * 1000 < timeout_ms:
        try:
            {wait_logic}
            return True
        except Exception:
            await asyncio.sleep(0.1)
    
    logger.warning(f"Wait condition timeout: {condition_name}")
    return False
''',
                LanguageType.JAVASCRIPT.value: '''
async function waitFor{ConditionName}(page, timeoutMs = 30000) {
    // Wait until {condition_description}
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
        try {
            {wait_logic}
            return true;
        } catch (error) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    console.warn(`Wait condition timeout: {condition_name}`);
    return false;
}
'''
            },
            CodeType.FALLBACK_STRATEGY.value: {
                LanguageType.PYTHON.value: '''
async def fallback_{strategy_name}(context: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """Fallback strategy for {strategy_description}"""
    logger.info(f"Executing fallback strategy: {strategy_name}")
    
    try:
        {fallback_logic}
        
        return {{
            "success": True,
            "strategy": "{strategy_name}",
            "result": result
        }}
    except Exception as e:
        logger.error(f"Fallback strategy failed: {{e}}")
        return {{
            "success": False,
            "strategy": "{strategy_name}",
            "error": str(e)
        }}
''',
                LanguageType.JAVASCRIPT.value: '''
async function fallback{StrategyName}(context, error) {
    // Fallback strategy for {strategy_description}
    console.info(`Executing fallback strategy: {strategy_name}`);
    
    try {
        {fallback_logic}
        
        return {
            success: true,
            strategy: "{strategy_name}",
            result: result
        };
    } catch (e) {
        console.error(`Fallback strategy failed: ${e.message}`);
        return {
            success: false,
            strategy: "{strategy_name}",
            error: e.message
        };
    }
}
'''
            },
            CodeType.UNIT_TEST.value: {
                LanguageType.PYTHON.value: '''
import pytest
import asyncio
from unittest.mock import Mock, patch

class Test{TestClassName}:
    """Unit tests for {test_description}"""
    
    def setup_method(self):
        """Setup test fixtures"""
        {setup_code}
    
    @pytest.mark.asyncio
    async def test_{test_name}_success(self):
        """Test successful {test_description}"""
        # Arrange
        {arrange_code}
        
        # Act
        result = await {function_call}
        
        # Assert
        {assert_code}
    
    @pytest.mark.asyncio
    async def test_{test_name}_failure(self):
        """Test {test_description} failure scenarios"""
        # Arrange
        {failure_arrange_code}
        
        # Act & Assert
        with pytest.raises({expected_exception}):
            await {function_call}
    
    def test_{test_name}_validation(self):
        """Test input validation for {test_description}"""
        {validation_test_code}
''',
                LanguageType.JAVASCRIPT.value: '''
describe('{test_description}', () => {
    let {test_variables};
    
    beforeEach(() => {
        {setup_code}
    });
    
    test('should successfully {test_description}', async () => {
        // Arrange
        {arrange_code}
        
        // Act
        const result = await {function_call};
        
        // Assert
        {assert_code}
    });
    
    test('should handle {test_description} failures', async () => {
        // Arrange
        {failure_arrange_code}
        
        // Act & Assert
        await expect({function_call}).rejects.toThrow({expected_error});
    });
    
    test('should validate input for {test_description}', () => {
        {validation_test_code}
    });
});
'''
            }
        }
    
    def get_template(self, code_type: CodeType, language: LanguageType) -> Optional[str]:
        """Get template for specific code type and language"""
        return self.templates.get(code_type.value, {}).get(language.value)

class AICodeGenerator:
    """AI-powered code generator using LLMs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_clients = {}
        self.fallback_processor = BuiltinAIProcessor()
        self.template_engine = CodeTemplateEngine()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients for code generation"""
        if OPENAI_AVAILABLE and self.config.get('openai_api_key'):
            try:
                self.llm_clients['openai'] = openai.AsyncOpenAI(
                    api_key=self.config['openai_api_key']
                )
                logger.info("✅ OpenAI client initialized for code generation")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        if ANTHROPIC_AVAILABLE and self.config.get('anthropic_api_key'):
            try:
                self.llm_clients['anthropic'] = anthropic.AsyncAnthropic(
                    api_key=self.config['anthropic_api_key']
                )
                logger.info("✅ Anthropic client initialized for code generation")
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
    
    async def generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code based on request"""
        start_time = time.time()
        
        # Try AI generation first
        if self.llm_clients:
            for provider, client in self.llm_clients.items():
                try:
                    result = await self._generate_with_llm(provider, client, request)
                    result.generation_time_ms = (time.time() - start_time) * 1000
                    result.used_ai = True
                    result.provider = provider
                    return result
                except Exception as e:
                    logger.warning(f"AI code generation failed with {provider}: {e}")
                    continue
        
        # Fallback to template-based generation
        result = self._generate_with_templates(request)
        result.generation_time_ms = (time.time() - start_time) * 1000
        result.used_ai = False
        result.provider = 'template'
        return result
    
    async def _generate_with_llm(self, provider: str, client: Any, 
                                request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code using specific LLM provider"""
        
        # Build generation prompt
        prompt = self._build_generation_prompt(request)
        
        if provider == 'openai':
            response = await client.chat.completions.create(
                model=self.config.get('openai_model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an expert automation engineer. Generate high-quality, production-ready code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return self._parse_llm_response(request, response.choices[0].message.content)
            
        elif provider == 'anthropic':
            response = await client.messages.create(
                model=self.config.get('anthropic_model', 'claude-3-sonnet-20240229'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            return self._parse_llm_response(request, response.content[0].text)
    
    def _build_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Build prompt for code generation"""
        context_str = json.dumps(request.context, indent=2)
        requirements_str = "\n".join(f"- {req}" for req in request.requirements)
        
        prompt = f"""
Generate {request.code_type.value} code for {request.framework.value} automation in {request.language.value}.

REQUIREMENTS:
{requirements_str}

CONTEXT:
{context_str}

EXISTING CODE (if any):
{request.existing_code or 'None'}

Please generate:
1. Clean, production-ready code
2. Proper error handling
3. Comprehensive comments
4. Type hints/annotations where applicable
5. Logging statements for debugging

Format your response as:
```{request.language.value}
[generated code here]
```

EXPLANATION:
[Brief explanation of the generated code]

DEPENDENCIES:
[List any required dependencies]

TEST_CASES:
[Suggest test cases for this code]
"""
        
        return prompt
    
    def _parse_llm_response(self, request: CodeGenerationRequest, response: str) -> GeneratedCode:
        """Parse LLM response into GeneratedCode object"""
        # Extract code block
        code_pattern = f"```{request.language.value}(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL | re.IGNORECASE)
        
        generated_code = ""
        if code_match:
            generated_code = code_match.group(1).strip()
        else:
            # Fallback: try to extract any code block
            code_match = re.search(r"```(.*?)```", response, re.DOTALL)
            if code_match:
                generated_code = code_match.group(1).strip()
        
        # Extract explanation
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:DEPENDENCIES:|TEST_CASES:|$)", response, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "AI-generated code"
        
        # Extract dependencies
        deps_match = re.search(r"DEPENDENCIES:\s*(.*?)(?:TEST_CASES:|$)", response, re.DOTALL | re.IGNORECASE)
        dependencies = []
        if deps_match:
            deps_text = deps_match.group(1).strip()
            dependencies = [dep.strip() for dep in deps_text.split('\n') if dep.strip()]
        
        # Extract test cases
        tests_match = re.search(r"TEST_CASES:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        test_cases = []
        if tests_match:
            tests_text = tests_match.group(1).strip()
            test_cases = [test.strip() for test in tests_text.split('\n') if test.strip()]
        
        return GeneratedCode(
            request_id=request.request_id,
            code_type=request.code_type,
            framework=request.framework,
            language=request.language,
            generated_code=generated_code,
            explanation=explanation,
            confidence=0.8,  # High confidence for AI-generated code
            dependencies=dependencies,
            test_cases=test_cases,
            validation_notes=[],
            generation_time_ms=0,  # Will be set by caller
            used_ai=True
        )
    
    def _generate_with_templates(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code using templates as fallback"""
        template = self.template_engine.get_template(request.code_type, request.language)
        
        if not template:
            # Basic fallback
            generated_code = f"# TODO: Implement {request.code_type.value} for {request.framework.value}"
            explanation = "Template-based placeholder code"
            confidence = 0.3
        else:
            # Fill template with context
            template_vars = self._extract_template_variables(request)
            try:
                generated_code = template.format(**template_vars)
                explanation = f"Template-based {request.code_type.value} code"
                confidence = 0.6
            except Exception as e:
                logger.warning(f"Template formatting failed: {e}")
                generated_code = template
                explanation = "Raw template code (variables not substituted)"
                confidence = 0.4
        
        return GeneratedCode(
            request_id=request.request_id,
            code_type=request.code_type,
            framework=request.framework,
            language=request.language,
            generated_code=generated_code,
            explanation=explanation,
            confidence=confidence,
            dependencies=[],
            test_cases=[],
            validation_notes=["Generated using templates - may need customization"],
            generation_time_ms=0,  # Will be set by caller
            used_ai=False
        )
    
    def _extract_template_variables(self, request: CodeGenerationRequest) -> Dict[str, str]:
        """Extract variables for template substitution"""
        context = request.context
        
        # Common template variables
        variables = {
            'condition_name': context.get('name', 'condition').lower().replace(' ', '_'),
            'ConditionName': context.get('name', 'Condition').replace(' ', ''),
            'condition_description': context.get('description', 'condition check'),
            'condition_logic': context.get('logic', 'pass  # TODO: Implement logic'),
            'verification_logic': context.get('verification', 'pass  # TODO: Implement verification'),
            'wait_logic': context.get('wait_condition', 'pass  # TODO: Implement wait condition'),
            'strategy_name': context.get('strategy', 'default').lower().replace(' ', '_'),
            'StrategyName': context.get('strategy', 'Default').replace(' ', ''),
            'strategy_description': context.get('strategy_description', 'fallback handling'),
            'fallback_logic': context.get('fallback_code', 'pass  # TODO: Implement fallback'),
            'test_name': context.get('test_name', 'function').lower().replace(' ', '_'),
            'TestClassName': context.get('class_name', 'TestClass').replace(' ', ''),
            'test_description': context.get('test_description', 'functionality'),
            'setup_code': context.get('setup', 'pass'),
            'arrange_code': context.get('arrange', 'pass'),
            'function_call': context.get('function_call', 'function()'),
            'assert_code': context.get('assertions', 'assert result is not None'),
            'failure_arrange_code': context.get('failure_setup', 'pass'),
            'expected_exception': context.get('expected_exception', 'Exception'),
            'expected_error': context.get('expected_error', 'Error'),
            'validation_test_code': context.get('validation_tests', 'pass'),
            'test_variables': context.get('test_variables', 'testData')
        }
        
        return variables

class CodeValidator:
    """Validates generated code for syntax and best practices"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, List[str]]:
        """Load validation rules for different languages"""
        return {
            LanguageType.PYTHON.value: [
                "Check for proper indentation",
                "Verify import statements",
                "Check for syntax errors",
                "Validate function definitions",
                "Check for proper exception handling"
            ],
            LanguageType.JAVASCRIPT.value: [
                "Check for proper bracket matching",
                "Verify semicolon usage",
                "Check for syntax errors",
                "Validate function definitions",
                "Check for proper error handling"
            ],
            LanguageType.TYPESCRIPT.value: [
                "Check for proper type annotations",
                "Verify interface definitions",
                "Check for syntax errors",
                "Validate function signatures",
                "Check for proper error handling"
            ]
        }
    
    def validate_code(self, generated_code: GeneratedCode) -> Tuple[bool, List[str]]:
        """Validate generated code"""
        issues = []
        
        # Basic syntax validation
        if generated_code.language == LanguageType.PYTHON:
            issues.extend(self._validate_python_syntax(generated_code.generated_code))
        elif generated_code.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
            issues.extend(self._validate_javascript_syntax(generated_code.generated_code))
        
        # Code quality checks
        issues.extend(self._validate_code_quality(generated_code))
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_python_syntax(self, code: str) -> List[str]:
        """Validate Python syntax"""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Python syntax error: {e}")
        
        # Check for common issues
        if 'import' not in code and 'from' not in code:
            if any(keyword in code for keyword in ['asyncio', 'time', 'logging', 'json']):
                issues.append("Missing import statements for used modules")
        
        return issues
    
    def _validate_javascript_syntax(self, code: str) -> List[str]:
        """Validate JavaScript/TypeScript syntax (basic checks)"""
        issues = []
        
        # Basic bracket matching
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    issues.append("Bracket mismatch detected")
                    break
        
        if stack:
            issues.append("Unclosed brackets detected")
        
        return issues
    
    def _validate_code_quality(self, generated_code: GeneratedCode) -> List[str]:
        """Validate code quality and best practices"""
        issues = []
        code = generated_code.generated_code
        
        # Check for TODO comments (may indicate incomplete code)
        if 'TODO' in code.upper():
            issues.append("Code contains TODO comments - may be incomplete")
        
        # Check for error handling
        if generated_code.code_type in [CodeType.FALLBACK_STRATEGY, CodeType.ERROR_HANDLER]:
            if 'try' not in code and 'catch' not in code:
                issues.append("Error handling code should include try/catch blocks")
        
        # Check for logging
        if 'log' not in code.lower() and generated_code.code_type != CodeType.UNIT_TEST:
            issues.append("Consider adding logging statements for debugging")
        
        return issues

class CopilotCodegenAI:
    """Main Copilot/Codegen AI system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ai_generator = AICodeGenerator(config)
        self.validator = CodeValidator()
        
        # Generation cache
        self.generation_cache: Dict[str, GeneratedCode] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'cache_hits': 0,
            'avg_generation_time_ms': 0,
            'code_type_distribution': {},
            'language_distribution': {}
        }
    
    async def generate_precondition(self, name: str, description: str, 
                                   logic: str, language: LanguageType = LanguageType.PYTHON,
                                   framework: FrameworkType = FrameworkType.GENERIC) -> GeneratedCode:
        """Generate precondition check code"""
        request = CodeGenerationRequest(
            request_id=f"precond_{hashlib.md5(f'{name}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.PRECONDITION,
            framework=framework,
            language=language,
            context={
                'name': name,
                'description': description,
                'logic': logic
            },
            requirements=[
                "Generate robust precondition check",
                "Include proper error handling",
                "Return boolean result",
                "Add logging for debugging"
            ]
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_postcondition(self, name: str, description: str, 
                                    verification_logic: str, language: LanguageType = LanguageType.PYTHON,
                                    framework: FrameworkType = FrameworkType.GENERIC) -> GeneratedCode:
        """Generate postcondition verification code"""
        request = CodeGenerationRequest(
            request_id=f"postcond_{hashlib.md5(f'{name}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.POSTCONDITION,
            framework=framework,
            language=language,
            context={
                'name': name,
                'description': description,
                'verification': verification_logic
            },
            requirements=[
                "Generate comprehensive postcondition verification",
                "Validate expected results",
                "Include error handling",
                "Return boolean result"
            ]
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_fallback_strategy(self, strategy_name: str, description: str,
                                       fallback_logic: str, error_context: Dict[str, Any],
                                       language: LanguageType = LanguageType.PYTHON,
                                       framework: FrameworkType = FrameworkType.GENERIC) -> GeneratedCode:
        """Generate fallback strategy code"""
        request = CodeGenerationRequest(
            request_id=f"fallback_{hashlib.md5(f'{strategy_name}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.FALLBACK_STRATEGY,
            framework=framework,
            language=language,
            context={
                'strategy': strategy_name,
                'strategy_description': description,
                'fallback_code': fallback_logic,
                'error_context': error_context
            },
            requirements=[
                "Generate robust fallback strategy",
                "Handle multiple error scenarios",
                "Return structured result",
                "Include comprehensive logging"
            ]
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_wait_predicate(self, condition_name: str, description: str,
                                     wait_logic: str, timeout_ms: int = 30000,
                                     language: LanguageType = LanguageType.PYTHON,
                                     framework: FrameworkType = FrameworkType.PLAYWRIGHT) -> GeneratedCode:
        """Generate wait predicate code"""
        request = CodeGenerationRequest(
            request_id=f"wait_{hashlib.md5(f'{condition_name}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.WAIT_PREDICATE,
            framework=framework,
            language=language,
            context={
                'name': condition_name,
                'description': description,
                'wait_condition': wait_logic,
                'timeout_ms': timeout_ms
            },
            requirements=[
                "Generate efficient wait predicate",
                "Include configurable timeout",
                "Handle polling gracefully",
                "Return boolean result"
            ]
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_unit_tests(self, function_name: str, function_code: str,
                                 test_scenarios: List[str], language: LanguageType = LanguageType.PYTHON,
                                 framework: FrameworkType = FrameworkType.GENERIC) -> GeneratedCode:
        """Generate comprehensive unit tests"""
        request = CodeGenerationRequest(
            request_id=f"test_{hashlib.md5(f'{function_name}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.UNIT_TEST,
            framework=framework,
            language=language,
            context={
                'function_name': function_name,
                'function_code': function_code,
                'test_scenarios': test_scenarios,
                'test_name': function_name,
                'class_name': f"Test{function_name.title()}",
                'test_description': f"{function_name} functionality"
            },
            requirements=[
                "Generate comprehensive test coverage",
                "Include positive and negative test cases",
                "Test edge cases and error conditions",
                "Use appropriate testing framework"
            ],
            existing_code=function_code
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_repair_patch(self, broken_code: str, error_message: str,
                                   context: Dict[str, Any], language: LanguageType = LanguageType.PYTHON,
                                   framework: FrameworkType = FrameworkType.GENERIC) -> GeneratedCode:
        """Generate repair patch for broken code"""
        request = CodeGenerationRequest(
            request_id=f"repair_{hashlib.md5(f'{error_message}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.REPAIR_PATCH,
            framework=framework,
            language=language,
            context={
                'error_message': error_message,
                'repair_context': context
            },
            requirements=[
                "Analyze and fix the reported error",
                "Maintain original functionality",
                "Add defensive programming practices",
                "Include comments explaining the fix"
            ],
            existing_code=broken_code
        )
        
        return await self._generate_and_validate(request)
    
    async def generate_automation_script(self, goal: str, steps: List[Dict[str, Any]],
                                       language: LanguageType = LanguageType.PYTHON,
                                       framework: FrameworkType = FrameworkType.PLAYWRIGHT) -> GeneratedCode:
        """Generate complete automation script"""
        request = CodeGenerationRequest(
            request_id=f"script_{hashlib.md5(f'{goal}_{time.time()}'.encode()).hexdigest()[:8]}",
            code_type=CodeType.AUTOMATION_SCRIPT,
            framework=framework,
            language=language,
            context={
                'goal': goal,
                'steps': steps,
                'framework': framework.value
            },
            requirements=[
                "Generate complete automation script",
                "Include proper setup and teardown",
                "Add error handling and recovery",
                "Include comprehensive logging",
                "Make script maintainable and readable"
            ]
        )
        
        return await self._generate_and_validate(request)
    
    async def _generate_and_validate(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code and validate it"""
        self.stats['total_requests'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.generation_cache:
            self.stats['cache_hits'] += 1
            return self.generation_cache[cache_key]
        
        # Generate code
        generated_code = await self.ai_generator.generate_code(request)
        
        # Validate generated code
        is_valid, validation_issues = self.validator.validate_code(generated_code)
        generated_code.validation_notes.extend(validation_issues)
        
        if is_valid:
            self.stats['successful_generations'] += 1
        
        # Update statistics
        self.stats['avg_generation_time_ms'] = (
            self.stats['avg_generation_time_ms'] * 0.9 + 
            generated_code.generation_time_ms * 0.1
        )
        
        code_type = request.code_type.value
        language = request.language.value
        
        self.stats['code_type_distribution'][code_type] = \
            self.stats['code_type_distribution'].get(code_type, 0) + 1
        self.stats['language_distribution'][language] = \
            self.stats['language_distribution'].get(language, 0) + 1
        
        # Cache result
        self.generation_cache[cache_key] = generated_code
        
        logger.info(f"✅ Generated {request.code_type.value} code: {generated_code.confidence:.2f} confidence")
        return generated_code
    
    def _generate_cache_key(self, request: CodeGenerationRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'code_type': request.code_type.value,
            'framework': request.framework.value,
            'language': request.language.value,
            'context': request.context,
            'requirements': request.requirements
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get code generation statistics"""
        total_requests = self.stats['total_requests']
        successful_requests = self.stats['successful_generations']
        
        return {
            'total_requests': total_requests,
            'successful_generations': successful_requests,
            'success_rate': (successful_requests / max(1, total_requests)) * 100,
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': (self.stats['cache_hits'] / max(1, total_requests)) * 100,
            'avg_generation_time_ms': self.stats['avg_generation_time_ms'],
            'code_type_distribution': self.stats['code_type_distribution'],
            'language_distribution': self.stats['language_distribution'],
            'cached_generations': len(self.generation_cache)
        }
    
    def get_copilot_stats(self) -> Dict[str, Any]:
        """Get copilot statistics - Fixed: add missing method"""
        total_requests = self.stats['total_requests']
        successful_requests = self.stats['successful_generations']
        
        return {
            'code_generations': total_requests,  # Fixed: add missing code_generations
            'successful_generations': successful_requests,
            'success_rate': (successful_requests / max(1, total_requests)) * 100,
            'avg_generation_time': self.stats['avg_generation_time_ms'],  # Fixed: add missing avg_generation_time
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': (self.stats['cache_hits'] / max(1, total_requests)) * 100,
            'code_type_distribution': self.stats['code_type_distribution'],
            'language_distribution': self.stats['language_distribution'],
            'cached_generations': len(self.generation_cache)
        }
    
    def clear_cache(self):
        """Clear generation cache"""
        self.generation_cache.clear()
        logger.info("🧹 Cleared code generation cache")

# Global instance
_copilot_ai_instance = None

def get_copilot_ai(config: Dict[str, Any] = None) -> CopilotCodegenAI:
    """Get global copilot AI instance"""
    global _copilot_ai_instance
    
    if _copilot_ai_instance is None:
        default_config = {
            'openai_api_key': None,
            'anthropic_api_key': None,
            'openai_model': 'gpt-4',
            'anthropic_model': 'claude-3-sonnet-20240229',
            'max_code_length': 2000,
            'cache_size': 1000
        }
        
        _copilot_ai_instance = CopilotCodegenAI(config or default_config)
    
    return _copilot_ai_instance

if __name__ == "__main__":
    # Demo the copilot system
    async def demo():
        print("🤖 Copilot/Codegen AI Demo")
        print("=" * 50)
        
        copilot = get_copilot_ai()
        
        # Test precondition generation
        print("🔍 Generating Precondition Code...")
        precond = await copilot.generate_precondition(
            name="page_loaded",
            description="page has finished loading",
            logic="return page.url != 'about:blank' and page.title() != ''"
        )
        
        print(f"✅ Generated precondition ({precond.confidence:.2f} confidence):")
        print(precond.generated_code[:200] + "..." if len(precond.generated_code) > 200 else precond.generated_code)
        print()
        
        # Test fallback strategy generation
        print("🔄 Generating Fallback Strategy...")
        fallback = await copilot.generate_fallback_strategy(
            strategy_name="retry_with_different_selector",
            description="retry action with alternative selector when primary fails",
            fallback_logic="alternative_selector = context.get('fallback_selector')\nreturn await page.click(alternative_selector)",
            error_context={"primary_selector": "#submit-btn", "fallback_selector": "input[type='submit']"}
        )
        
        print(f"✅ Generated fallback strategy ({fallback.confidence:.2f} confidence):")
        print(fallback.generated_code[:200] + "..." if len(fallback.generated_code) > 200 else fallback.generated_code)
        print()
        
        # Test wait predicate generation
        print("⏱️ Generating Wait Predicate...")
        wait_pred = await copilot.generate_wait_predicate(
            condition_name="element_visible",
            description="element becomes visible on page",
            wait_logic="element = page.locator(selector)\nreturn element.is_visible()"
        )
        
        print(f"✅ Generated wait predicate ({wait_pred.confidence:.2f} confidence):")
        print(wait_pred.generated_code[:200] + "..." if len(wait_pred.generated_code) > 200 else wait_pred.generated_code)
        print()
        
        # Test unit test generation
        print("🧪 Generating Unit Tests...")
        tests = await copilot.generate_unit_tests(
            function_name="click_button",
            function_code="async def click_button(page, selector): return await page.click(selector)",
            test_scenarios=["successful click", "element not found", "element not clickable"]
        )
        
        print(f"✅ Generated unit tests ({tests.confidence:.2f} confidence):")
        print(tests.generated_code[:300] + "..." if len(tests.generated_code) > 300 else tests.generated_code)
        print()
        
        # Test automation script generation
        print("📝 Generating Automation Script...")
        script = await copilot.generate_automation_script(
            goal="Login to website",
            steps=[
                {"action": "navigate", "url": "https://example.com/login"},
                {"action": "type", "selector": "#username", "text": "user@example.com"},
                {"action": "type", "selector": "#password", "text": "password"},
                {"action": "click", "selector": "#login-btn"}
            ]
        )
        
        print(f"✅ Generated automation script ({script.confidence:.2f} confidence):")
        print(script.generated_code[:400] + "..." if len(script.generated_code) > 400 else script.generated_code)
        print()
        
        # Show statistics
        stats = copilot.get_generation_stats()
        print("📊 Generation Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Copilot demo complete!")
        print("🏆 AI-powered code generation with intelligent templates!")
    
    asyncio.run(demo())