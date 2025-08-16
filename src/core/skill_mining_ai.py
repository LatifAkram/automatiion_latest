#!/usr/bin/env python3
"""
Auto Skill-Mining AI - Pattern Abstraction & Skill Pack Generation
=================================================================

AI-powered system that:
- Watches successful automation runs
- Identifies reusable patterns and skills
- Abstracts them into validated Skill Packs
- Generates pre/postconditions automatically
- Creates test cases for validation

Learns and improves automation capabilities over time.
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict

# AI imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    np = None

# Built-in fallbacks
from builtin_ai_processor import BuiltinAIProcessor
from builtin_data_validation import BaseValidator

logger = logging.getLogger(__name__)

class SkillType(Enum):
    """Types of skills that can be mined"""
    NAVIGATION = "navigation"
    FORM_FILLING = "form_filling"
    DATA_EXTRACTION = "data_extraction"
    AUTHENTICATION = "authentication"
    SEARCH_OPERATION = "search_operation"
    FILE_OPERATION = "file_operation"
    UI_INTERACTION = "ui_interaction"
    WORKFLOW_SEQUENCE = "workflow_sequence"

class SkillComplexity(Enum):
    """Complexity levels of skills"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class ExecutionTrace:
    """Trace of a successful execution"""
    trace_id: str
    goal: str
    steps: List[Dict[str, Any]]
    execution_time_ms: float
    success_rate: float
    context: Dict[str, Any]
    timestamp: float
    platform: str
    domain: str

@dataclass
class SkillPattern:
    """Identified pattern that could become a skill"""
    pattern_id: str
    skill_type: SkillType
    complexity: SkillComplexity
    step_sequence: List[Dict[str, Any]]
    preconditions: List[str]
    postconditions: List[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    confidence_score: float
    frequency_count: int
    platforms: Set[str]
    domains: Set[str]
    variations: List[Dict[str, Any]]

@dataclass
class SkillPack:
    """Validated, reusable skill pack"""
    skill_id: str
    name: str
    description: str
    skill_type: SkillType
    complexity: SkillComplexity
    template_steps: List[Dict[str, Any]]
    preconditions: List[Dict[str, Any]]
    postconditions: List[Dict[str, Any]]
    parameters: List[Dict[str, Any]]
    test_cases: List[Dict[str, Any]]
    success_rate: float
    usage_count: int
    last_updated: float
    metadata: Dict[str, Any]

class PatternRecognitionAI:
    """AI-powered pattern recognition for skill mining"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_model = None
        self.sequence_model = None
        self.fallback_processor = BuiltinAIProcessor()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for pattern recognition"""
        if AI_AVAILABLE:
            try:
                # Text embedding for semantic analysis
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Sequence model for pattern analysis
                self.sequence_model = AutoModel.from_pretrained('distilbert-base-uncased')
                
                logger.info("✅ Pattern recognition models loaded")
            except Exception as e:
                logger.warning(f"AI models loading failed, using fallback: {e}")
    
    def identify_patterns_in_trace(self, trace: ExecutionTrace) -> List[SkillPattern]:
        """Identify reusable patterns in execution trace"""
        patterns = []
        
        if self.text_model:
            # Use AI for pattern recognition
            patterns.extend(self._ai_pattern_recognition(trace))
        else:
            # Fallback to rule-based pattern recognition
            patterns.extend(self._fallback_pattern_recognition(trace))
        
        return patterns
    
    def _ai_pattern_recognition(self, trace: ExecutionTrace) -> List[SkillPattern]:
        """AI-powered pattern recognition"""
        patterns = []
        
        try:
            # Analyze step sequences using embeddings
            step_embeddings = []
            step_texts = []
            
            for step in trace.steps:
                step_text = f"{step.get('goal', '')} {step.get('action', {}).get('type', '')}"
                step_texts.append(step_text)
                
                if step_text.strip():
                    embedding = self.text_model.encode([step_text])[0]
                    step_embeddings.append(embedding)
            
            if len(step_embeddings) >= 2:
                # Find similar step sequences (simplified)
                patterns.extend(self._find_sequence_patterns(trace, step_embeddings, step_texts))
                
                # Find semantic clusters
                patterns.extend(self._find_semantic_clusters(trace, step_embeddings, step_texts))
            
        except Exception as e:
            logger.warning(f"AI pattern recognition failed: {e}")
            # Fallback to rule-based
            patterns.extend(self._fallback_pattern_recognition(trace))
        
        return patterns
    
    def _find_sequence_patterns(self, trace: ExecutionTrace, embeddings: List, texts: List[str]) -> List[SkillPattern]:
        """Find sequential patterns in steps"""
        patterns = []
        
        # Look for common sequences of 2-5 steps
        for seq_len in range(2, min(6, len(trace.steps) + 1)):
            for i in range(len(trace.steps) - seq_len + 1):
                sequence = trace.steps[i:i + seq_len]
                
                # Analyze sequence semantics
                seq_texts = texts[i:i + seq_len]
                pattern_type = self._classify_sequence_type(' '.join(seq_texts))
                
                if pattern_type != SkillType.UI_INTERACTION:  # More specific than generic
                    pattern = SkillPattern(
                        pattern_id=f"seq_{hashlib.md5(''.join(seq_texts).encode()).hexdigest()[:8]}",
                        skill_type=pattern_type,
                        complexity=self._assess_complexity(sequence),
                        step_sequence=sequence,
                        preconditions=self._extract_preconditions(sequence),
                        postconditions=self._extract_postconditions(sequence),
                        success_indicators=[],
                        failure_indicators=[],
                        confidence_score=0.7,
                        frequency_count=1,
                        platforms={trace.platform},
                        domains={trace.domain},
                        variations=[]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_semantic_clusters(self, trace: ExecutionTrace, embeddings: List, texts: List[str]) -> List[SkillPattern]:
        """Find semantically similar steps that could form patterns"""
        patterns = []
        
        # Simple clustering based on step similarity
        clustered_steps = defaultdict(list)
        
        for i, (embedding, text, step) in enumerate(zip(embeddings, texts, trace.steps)):
            # Find cluster based on action type and semantic content
            cluster_key = f"{step.get('action', {}).get('type', 'unknown')}"
            
            # Add semantic context
            if 'login' in text.lower() or 'auth' in text.lower():
                cluster_key = 'authentication'
            elif 'form' in text.lower() or 'input' in text.lower():
                cluster_key = 'form_filling'
            elif 'search' in text.lower() or 'find' in text.lower():
                cluster_key = 'search_operation'
            
            clustered_steps[cluster_key].append((i, step, text))
        
        # Create patterns from clusters with multiple steps
        for cluster_type, steps in clustered_steps.items():
            if len(steps) >= 2:
                step_sequence = [step for _, step, _ in steps]
                
                pattern = SkillPattern(
                    pattern_id=f"cluster_{cluster_type}_{hashlib.md5(str(steps).encode()).hexdigest()[:8]}",
                    skill_type=self._map_cluster_to_skill_type(cluster_type),
                    complexity=self._assess_complexity(step_sequence),
                    step_sequence=step_sequence,
                    preconditions=self._extract_preconditions(step_sequence),
                    postconditions=self._extract_postconditions(step_sequence),
                    success_indicators=[],
                    failure_indicators=[],
                    confidence_score=0.6,
                    frequency_count=len(steps),
                    platforms={trace.platform},
                    domains={trace.domain},
                    variations=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _fallback_pattern_recognition(self, trace: ExecutionTrace) -> List[SkillPattern]:
        """Rule-based fallback pattern recognition"""
        patterns = []
        
        # Look for common automation patterns
        steps = trace.steps
        
        # Navigation patterns
        nav_pattern = self._find_navigation_pattern(steps)
        if nav_pattern:
            patterns.append(nav_pattern)
        
        # Form filling patterns
        form_pattern = self._find_form_pattern(steps)
        if form_pattern:
            patterns.append(form_pattern)
        
        # Authentication patterns
        auth_pattern = self._find_auth_pattern(steps)
        if auth_pattern:
            patterns.append(auth_pattern)
        
        # Search patterns
        search_pattern = self._find_search_pattern(steps)
        if search_pattern:
            patterns.append(search_pattern)
        
        return patterns
    
    def _find_navigation_pattern(self, steps: List[Dict[str, Any]]) -> Optional[SkillPattern]:
        """Find navigation patterns in steps"""
        nav_steps = []
        
        for step in steps:
            action = step.get('action', {})
            if action.get('type') in ['navigate', 'click'] and 'url' in action:
                nav_steps.append(step)
        
        if len(nav_steps) >= 2:
            return SkillPattern(
                pattern_id=f"nav_{hashlib.md5(str(nav_steps).encode()).hexdigest()[:8]}",
                skill_type=SkillType.NAVIGATION,
                complexity=SkillComplexity.BASIC,
                step_sequence=nav_steps,
                preconditions=['browser_ready'],
                postconditions=['page_loaded'],
                success_indicators=['page_title_matches', 'url_matches'],
                failure_indicators=['page_error', 'timeout'],
                confidence_score=0.8,
                frequency_count=1,
                platforms=set(),
                domains=set(),
                variations=[]
            )
        
        return None
    
    def _find_form_pattern(self, steps: List[Dict[str, Any]]) -> Optional[SkillPattern]:
        """Find form filling patterns"""
        form_steps = []
        
        for step in steps:
            action = step.get('action', {})
            if action.get('type') in ['type', 'click', 'select'] and \
               any(keyword in str(action).lower() for keyword in ['input', 'textbox', 'form', 'submit']):
                form_steps.append(step)
        
        if len(form_steps) >= 2:
            return SkillPattern(
                pattern_id=f"form_{hashlib.md5(str(form_steps).encode()).hexdigest()[:8]}",
                skill_type=SkillType.FORM_FILLING,
                complexity=SkillComplexity.INTERMEDIATE,
                step_sequence=form_steps,
                preconditions=['form_visible', 'page_loaded'],
                postconditions=['form_submitted', 'success_message_shown'],
                success_indicators=['form_submitted', 'no_validation_errors'],
                failure_indicators=['validation_error', 'form_reset'],
                confidence_score=0.7,
                frequency_count=1,
                platforms=set(),
                domains=set(),
                variations=[]
            )
        
        return None
    
    def _find_auth_pattern(self, steps: List[Dict[str, Any]]) -> Optional[SkillPattern]:
        """Find authentication patterns"""
        auth_steps = []
        
        for step in steps:
            action = step.get('action', {})
            step_text = str(step).lower()
            
            if any(keyword in step_text for keyword in ['login', 'password', 'username', 'auth', 'signin']):
                auth_steps.append(step)
        
        if len(auth_steps) >= 2:
            return SkillPattern(
                pattern_id=f"auth_{hashlib.md5(str(auth_steps).encode()).hexdigest()[:8]}",
                skill_type=SkillType.AUTHENTICATION,
                complexity=SkillComplexity.INTERMEDIATE,
                step_sequence=auth_steps,
                preconditions=['login_page_loaded', 'credentials_available'],
                postconditions=['user_authenticated', 'dashboard_visible'],
                success_indicators=['login_successful', 'user_menu_visible'],
                failure_indicators=['invalid_credentials', 'captcha_required'],
                confidence_score=0.9,
                frequency_count=1,
                platforms=set(),
                domains=set(),
                variations=[]
            )
        
        return None
    
    def _find_search_pattern(self, steps: List[Dict[str, Any]]) -> Optional[SkillPattern]:
        """Find search operation patterns"""
        search_steps = []
        
        for step in steps:
            step_text = str(step).lower()
            if any(keyword in step_text for keyword in ['search', 'find', 'query', 'filter']):
                search_steps.append(step)
        
        if len(search_steps) >= 1:
            return SkillPattern(
                pattern_id=f"search_{hashlib.md5(str(search_steps).encode()).hexdigest()[:8]}",
                skill_type=SkillType.SEARCH_OPERATION,
                complexity=SkillComplexity.BASIC,
                step_sequence=search_steps,
                preconditions=['search_interface_available'],
                postconditions=['search_results_displayed'],
                success_indicators=['results_found', 'no_error_message'],
                failure_indicators=['no_results', 'search_error'],
                confidence_score=0.6,
                frequency_count=1,
                platforms=set(),
                domains=set(),
                variations=[]
            )
        
        return None
    
    def _classify_sequence_type(self, sequence_text: str) -> SkillType:
        """Classify sequence type based on content"""
        text = sequence_text.lower()
        
        if any(word in text for word in ['navigate', 'url', 'page', 'link']):
            return SkillType.NAVIGATION
        elif any(word in text for word in ['form', 'input', 'submit', 'fill']):
            return SkillType.FORM_FILLING
        elif any(word in text for word in ['login', 'auth', 'password', 'signin']):
            return SkillType.AUTHENTICATION
        elif any(word in text for word in ['search', 'find', 'query', 'filter']):
            return SkillType.SEARCH_OPERATION
        elif any(word in text for word in ['extract', 'scrape', 'data', 'text']):
            return SkillType.DATA_EXTRACTION
        elif any(word in text for word in ['file', 'download', 'upload', 'save']):
            return SkillType.FILE_OPERATION
        elif any(word in text for word in ['workflow', 'sequence', 'process']):
            return SkillType.WORKFLOW_SEQUENCE
        else:
            return SkillType.UI_INTERACTION
    
    def _map_cluster_to_skill_type(self, cluster_type: str) -> SkillType:
        """Map cluster type to skill type"""
        mapping = {
            'authentication': SkillType.AUTHENTICATION,
            'form_filling': SkillType.FORM_FILLING,
            'search_operation': SkillType.SEARCH_OPERATION,
            'navigate': SkillType.NAVIGATION,
            'click': SkillType.UI_INTERACTION,
            'type': SkillType.FORM_FILLING
        }
        return mapping.get(cluster_type, SkillType.UI_INTERACTION)
    
    def _assess_complexity(self, steps: List[Dict[str, Any]]) -> SkillComplexity:
        """Assess complexity of step sequence"""
        if len(steps) <= 2:
            return SkillComplexity.BASIC
        elif len(steps) <= 5:
            return SkillComplexity.INTERMEDIATE
        elif len(steps) <= 10:
            return SkillComplexity.ADVANCED
        else:
            return SkillComplexity.EXPERT
    
    def _extract_preconditions(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract preconditions from step sequence"""
        preconditions = []
        
        if not steps:
            return preconditions
        
        first_step = steps[0]
        action = first_step.get('action', {})
        
        # Common preconditions based on first step
        if action.get('type') == 'navigate':
            preconditions.extend(['browser_ready', 'internet_connected'])
        elif action.get('type') in ['click', 'type']:
            preconditions.extend(['page_loaded', 'element_visible'])
        elif 'login' in str(first_step).lower():
            preconditions.extend(['login_page_loaded', 'credentials_available'])
        
        return preconditions
    
    def _extract_postconditions(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract postconditions from step sequence"""
        postconditions = []
        
        if not steps:
            return postconditions
        
        last_step = steps[-1]
        action = last_step.get('action', {})
        
        # Common postconditions based on last step
        if action.get('type') == 'navigate':
            postconditions.extend(['page_loaded', 'url_changed'])
        elif action.get('type') == 'click' and 'submit' in str(last_step).lower():
            postconditions.extend(['form_submitted', 'success_message'])
        elif 'login' in str(last_step).lower():
            postconditions.extend(['user_authenticated', 'dashboard_visible'])
        
        return postconditions

class SkillValidator:
    """Validates and tests skill patterns before creating skill packs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different skill types"""
        return {
            SkillType.NAVIGATION: {
                'min_steps': 1,
                'required_preconditions': ['browser_ready'],
                'required_postconditions': ['page_loaded'],
                'min_confidence': 0.6
            },
            SkillType.FORM_FILLING: {
                'min_steps': 2,
                'required_preconditions': ['form_visible'],
                'required_postconditions': ['form_submitted'],
                'min_confidence': 0.7
            },
            SkillType.AUTHENTICATION: {
                'min_steps': 2,
                'required_preconditions': ['login_page_loaded'],
                'required_postconditions': ['user_authenticated'],
                'min_confidence': 0.8
            }
        }
    
    async def validate_pattern(self, pattern: SkillPattern) -> Tuple[bool, List[str]]:
        """Validate a skill pattern"""
        issues = []
        
        # Get validation rules for this skill type
        rules = self.validation_rules.get(pattern.skill_type, {})
        
        # Check minimum steps
        min_steps = rules.get('min_steps', 1)
        if len(pattern.step_sequence) < min_steps:
            issues.append(f"Too few steps: {len(pattern.step_sequence)} < {min_steps}")
        
        # Check confidence threshold
        min_confidence = rules.get('min_confidence', 0.5)
        if pattern.confidence_score < min_confidence:
            issues.append(f"Low confidence: {pattern.confidence_score} < {min_confidence}")
        
        # Check required preconditions
        required_pre = rules.get('required_preconditions', [])
        missing_pre = [cond for cond in required_pre if cond not in pattern.preconditions]
        if missing_pre:
            issues.append(f"Missing preconditions: {missing_pre}")
        
        # Check required postconditions
        required_post = rules.get('required_postconditions', [])
        missing_post = [cond for cond in required_post if cond not in pattern.postconditions]
        if missing_post:
            issues.append(f"Missing postconditions: {missing_post}")
        
        # Validate step consistency
        consistency_issues = self._validate_step_consistency(pattern.step_sequence)
        issues.extend(consistency_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_step_consistency(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Validate consistency within step sequence"""
        issues = []
        
        # Check for missing required fields
        for i, step in enumerate(steps):
            if not step.get('goal'):
                issues.append(f"Step {i}: Missing goal")
            
            if not step.get('action'):
                issues.append(f"Step {i}: Missing action")
            
            action = step.get('action', {})
            if not action.get('type'):
                issues.append(f"Step {i}: Missing action type")
        
        return issues
    
    def generate_test_cases(self, pattern: SkillPattern) -> List[Dict[str, Any]]:
        """Generate test cases for skill pattern"""
        test_cases = []
        
        # Basic success test case
        success_test = {
            'name': f"test_{pattern.skill_type.value}_success",
            'description': f"Test successful {pattern.skill_type.value} execution",
            'preconditions': pattern.preconditions.copy(),
            'expected_postconditions': pattern.postconditions.copy(),
            'test_data': self._generate_test_data(pattern),
            'expected_result': 'success'
        }
        test_cases.append(success_test)
        
        # Failure test cases
        if pattern.failure_indicators:
            for indicator in pattern.failure_indicators[:2]:  # Max 2 failure tests
                failure_test = {
                    'name': f"test_{pattern.skill_type.value}_failure_{indicator}",
                    'description': f"Test {pattern.skill_type.value} failure: {indicator}",
                    'preconditions': pattern.preconditions.copy(),
                    'expected_postconditions': [],
                    'test_data': self._generate_failure_test_data(pattern, indicator),
                    'expected_result': 'failure',
                    'expected_failure': indicator
                }
                test_cases.append(failure_test)
        
        # Edge case test
        if len(pattern.step_sequence) > 1:
            edge_test = {
                'name': f"test_{pattern.skill_type.value}_partial",
                'description': f"Test partial {pattern.skill_type.value} execution",
                'preconditions': pattern.preconditions.copy(),
                'expected_postconditions': [],
                'test_data': self._generate_edge_test_data(pattern),
                'expected_result': 'partial_success'
            }
            test_cases.append(edge_test)
        
        return test_cases
    
    def _generate_test_data(self, pattern: SkillPattern) -> Dict[str, Any]:
        """Generate test data for pattern"""
        test_data = {}
        
        # Extract parameters from steps
        for step in pattern.step_sequence:
            action = step.get('action', {})
            
            if action.get('type') == 'type' and 'text' in action:
                test_data['input_text'] = "test_input"
            elif action.get('type') == 'navigate' and 'url' in action:
                test_data['target_url'] = "https://example.com"
            elif action.get('type') == 'click':
                test_data['click_target'] = action.get('target', {})
        
        return test_data
    
    def _generate_failure_test_data(self, pattern: SkillPattern, failure_type: str) -> Dict[str, Any]:
        """Generate test data that should cause specific failure"""
        test_data = self._generate_test_data(pattern)
        
        # Modify test data to trigger failure
        if failure_type == 'invalid_credentials':
            test_data['username'] = 'invalid_user'
            test_data['password'] = 'wrong_password'
        elif failure_type == 'timeout':
            test_data['timeout_ms'] = 1  # Very short timeout
        elif failure_type == 'validation_error':
            test_data['input_text'] = ''  # Empty input
        
        return test_data
    
    def _generate_edge_test_data(self, pattern: SkillPattern) -> Dict[str, Any]:
        """Generate edge case test data"""
        test_data = self._generate_test_data(pattern)
        
        # Add edge case modifications
        test_data['partial_execution'] = True
        test_data['stop_at_step'] = len(pattern.step_sequence) // 2
        
        return test_data

class AutoSkillMiningAI:
    """Main auto skill-mining system with AI capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_recognizer = PatternRecognitionAI(config)
        self.skill_validator = SkillValidator(config)
        
        # Storage
        self.execution_traces: List[ExecutionTrace] = []
        self.identified_patterns: Dict[str, SkillPattern] = {}
        self.validated_skills: Dict[str, SkillPack] = {}
        
        # Statistics
        self.stats = {
            'traces_analyzed': 0,
            'patterns_identified': 0,
            'skills_created': 0,
            'skill_usage_count': 0,
            'success_rate': 0.0
        }
        
        # Load existing skills
        self._load_existing_skills()
    
    def _load_existing_skills(self):
        """Load existing skill packs from storage"""
        skills_dir = Path('data/skills')
        if skills_dir.exists():
            for skill_file in skills_dir.glob('*.json'):
                try:
                    with open(skill_file, 'r') as f:
                        skill_data = json.load(f)
                        skill = SkillPack(**skill_data)
                        self.validated_skills[skill.skill_id] = skill
                    logger.info(f"Loaded skill: {skill.name}")
                except Exception as e:
                    logger.warning(f"Failed to load skill {skill_file}: {e}")
    
    async def analyze_execution_trace(self, trace: ExecutionTrace) -> List[SkillPattern]:
        """Analyze execution trace and identify new patterns"""
        logger.info(f"🔍 Analyzing trace: {trace.trace_id}")
        
        self.execution_traces.append(trace)
        self.stats['traces_analyzed'] += 1
        
        # Identify patterns in this trace
        new_patterns = self.pattern_recognizer.identify_patterns_in_trace(trace)
        
        # Merge with existing patterns or create new ones
        merged_patterns = []
        
        for pattern in new_patterns:
            existing_pattern = self._find_similar_pattern(pattern)
            
            if existing_pattern:
                # Update existing pattern
                updated_pattern = self._merge_patterns(existing_pattern, pattern)
                self.identified_patterns[updated_pattern.pattern_id] = updated_pattern
                merged_patterns.append(updated_pattern)
            else:
                # Add new pattern
                self.identified_patterns[pattern.pattern_id] = pattern
                merged_patterns.append(pattern)
                self.stats['patterns_identified'] += 1
        
        logger.info(f"✅ Found {len(merged_patterns)} patterns in trace")
        return merged_patterns
    
    async def mine_skills_from_patterns(self, min_frequency: int = 2) -> List[SkillPack]:
        """Mine validated skills from identified patterns"""
        logger.info(f"⛏️ Mining skills from {len(self.identified_patterns)} patterns")
        
        new_skills = []
        
        for pattern in self.identified_patterns.values():
            # Only consider patterns with sufficient frequency
            if pattern.frequency_count >= min_frequency:
                
                # Validate pattern
                is_valid, issues = await self.skill_validator.validate_pattern(pattern)
                
                if is_valid:
                    # Create skill pack
                    skill_pack = await self._create_skill_pack(pattern)
                    
                    if skill_pack:
                        self.validated_skills[skill_pack.skill_id] = skill_pack
                        new_skills.append(skill_pack)
                        self.stats['skills_created'] += 1
                        
                        logger.info(f"✅ Created skill: {skill_pack.name}")
                else:
                    logger.warning(f"❌ Pattern validation failed: {issues}")
        
        # Save new skills
        await self._save_skills(new_skills)
        
        logger.info(f"⛏️ Mined {len(new_skills)} new skills")
        return new_skills
    
    def _find_similar_pattern(self, pattern: SkillPattern) -> Optional[SkillPattern]:
        """Find similar existing pattern"""
        for existing in self.identified_patterns.values():
            if (existing.skill_type == pattern.skill_type and
                self._calculate_pattern_similarity(existing, pattern) > 0.7):
                return existing
        return None
    
    def _calculate_pattern_similarity(self, pattern1: SkillPattern, pattern2: SkillPattern) -> float:
        """Calculate similarity between two patterns"""
        if pattern1.skill_type != pattern2.skill_type:
            return 0.0
        
        # Compare step sequences
        steps1 = [step.get('action', {}).get('type', '') for step in pattern1.step_sequence]
        steps2 = [step.get('action', {}).get('type', '') for step in pattern2.step_sequence]
        
        # Simple sequence similarity
        if len(steps1) == 0 and len(steps2) == 0:
            return 1.0
        
        common_steps = len(set(steps1).intersection(set(steps2)))
        total_steps = len(set(steps1).union(set(steps2)))
        
        step_similarity = common_steps / total_steps if total_steps > 0 else 0.0
        
        # Compare preconditions
        pre1 = set(pattern1.preconditions)
        pre2 = set(pattern2.preconditions)
        pre_similarity = len(pre1.intersection(pre2)) / len(pre1.union(pre2)) if pre1.union(pre2) else 1.0
        
        # Weighted average
        return 0.7 * step_similarity + 0.3 * pre_similarity
    
    def _merge_patterns(self, existing: SkillPattern, new: SkillPattern) -> SkillPattern:
        """Merge new pattern into existing pattern"""
        # Update frequency
        existing.frequency_count += 1
        
        # Merge platforms and domains
        existing.platforms.update(new.platforms)
        existing.domains.update(new.domains)
        
        # Add variation if different
        if new.step_sequence != existing.step_sequence:
            existing.variations.append({
                'sequence': new.step_sequence,
                'platforms': list(new.platforms),
                'domains': list(new.domains)
            })
        
        # Update confidence (running average)
        existing.confidence_score = (existing.confidence_score + new.confidence_score) / 2
        
        return existing
    
    async def _create_skill_pack(self, pattern: SkillPattern) -> Optional[SkillPack]:
        """Create validated skill pack from pattern"""
        try:
            # Generate test cases
            test_cases = self.skill_validator.generate_test_cases(pattern)
            
            # Extract parameters from steps
            parameters = self._extract_parameters(pattern.step_sequence)
            
            # Create skill pack
            skill_pack = SkillPack(
                skill_id=f"skill_{pattern.pattern_id}",
                name=self._generate_skill_name(pattern),
                description=self._generate_skill_description(pattern),
                skill_type=pattern.skill_type,
                complexity=pattern.complexity,
                template_steps=self._create_template_steps(pattern.step_sequence),
                preconditions=self._format_conditions(pattern.preconditions),
                postconditions=self._format_conditions(pattern.postconditions),
                parameters=parameters,
                test_cases=test_cases,
                success_rate=min(pattern.confidence_score, 0.95),  # Conservative estimate
                usage_count=0,
                last_updated=time.time(),
                metadata={
                    'source_pattern_id': pattern.pattern_id,
                    'frequency_count': pattern.frequency_count,
                    'platforms': list(pattern.platforms),
                    'domains': list(pattern.domains),
                    'variations_count': len(pattern.variations)
                }
            )
            
            return skill_pack
            
        except Exception as e:
            logger.error(f"Failed to create skill pack: {e}")
            return None
    
    def _extract_parameters(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract parameters from step sequence"""
        parameters = []
        param_names = set()
        
        for step in steps:
            action = step.get('action', {})
            
            # Extract text inputs
            if action.get('type') == 'type' and 'text' in action:
                if 'input_text' not in param_names:
                    parameters.append({
                        'name': 'input_text',
                        'type': 'string',
                        'description': 'Text to input',
                        'required': True,
                        'default': ''
                    })
                    param_names.add('input_text')
            
            # Extract URLs
            if action.get('type') == 'navigate' and 'url' in action:
                if 'target_url' not in param_names:
                    parameters.append({
                        'name': 'target_url',
                        'type': 'string',
                        'description': 'Target URL to navigate to',
                        'required': True,
                        'default': ''
                    })
                    param_names.add('target_url')
            
            # Extract selectors
            if 'target' in action:
                if 'selector' not in param_names:
                    parameters.append({
                        'name': 'selector',
                        'type': 'object',
                        'description': 'Element selector',
                        'required': True,
                        'default': {}
                    })
                    param_names.add('selector')
        
        return parameters
    
    def _create_template_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create parameterized template steps"""
        template_steps = []
        
        for step in steps:
            template_step = step.copy()
            action = template_step.get('action', {})
            
            # Parameterize text inputs
            if action.get('type') == 'type' and 'text' in action:
                action['text'] = '${input_text}'
            
            # Parameterize URLs
            if action.get('type') == 'navigate' and 'url' in action:
                action['url'] = '${target_url}'
            
            # Parameterize selectors
            if 'target' in action:
                action['target'] = '${selector}'
            
            template_steps.append(template_step)
        
        return template_steps
    
    def _format_conditions(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Format conditions as structured objects"""
        formatted = []
        
        for condition in conditions:
            formatted.append({
                'condition': condition,
                'type': 'boolean',
                'description': f"Check if {condition.replace('_', ' ')}",
                'timeout_ms': 5000
            })
        
        return formatted
    
    def _generate_skill_name(self, pattern: SkillPattern) -> str:
        """Generate human-readable skill name"""
        skill_type_names = {
            SkillType.NAVIGATION: "Navigate",
            SkillType.FORM_FILLING: "Fill Form",
            SkillType.AUTHENTICATION: "Login",
            SkillType.SEARCH_OPERATION: "Search",
            SkillType.DATA_EXTRACTION: "Extract Data",
            SkillType.FILE_OPERATION: "File Operation",
            SkillType.UI_INTERACTION: "UI Interaction",
            SkillType.WORKFLOW_SEQUENCE: "Workflow"
        }
        
        base_name = skill_type_names.get(pattern.skill_type, "Unknown Skill")
        
        # Add domain context if available
        if pattern.domains:
            domain = list(pattern.domains)[0]
            return f"{base_name} - {domain.title()}"
        
        return base_name
    
    def _generate_skill_description(self, pattern: SkillPattern) -> str:
        """Generate skill description"""
        desc = f"Automated {pattern.skill_type.value.replace('_', ' ')} skill"
        
        if pattern.domains:
            domains = ', '.join(pattern.domains)
            desc += f" for {domains}"
        
        desc += f" with {len(pattern.step_sequence)} steps"
        
        if pattern.frequency_count > 1:
            desc += f" (learned from {pattern.frequency_count} executions)"
        
        return desc
    
    async def _save_skills(self, skills: List[SkillPack]):
        """Save skills to persistent storage"""
        skills_dir = Path('data/skills')
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        for skill in skills:
            skill_file = skills_dir / f"{skill.skill_id}.json"
            
            try:
                with open(skill_file, 'w') as f:
                    json.dump(asdict(skill), f, indent=2, default=str)
                logger.info(f"💾 Saved skill: {skill_file}")
            except Exception as e:
                logger.error(f"Failed to save skill {skill.skill_id}: {e}")
    
    def get_skill_recommendations(self, goal: str, context: Dict[str, Any]) -> List[SkillPack]:
        """Get skill recommendations for a goal"""
        recommendations = []
        goal_lower = goal.lower()
        
        for skill in self.validated_skills.values():
            score = 0.0
            
            # Match by skill type keywords
            if skill.skill_type == SkillType.AUTHENTICATION and any(word in goal_lower for word in ['login', 'auth', 'signin']):
                score += 0.8
            elif skill.skill_type == SkillType.FORM_FILLING and any(word in goal_lower for word in ['form', 'fill', 'submit']):
                score += 0.8
            elif skill.skill_type == SkillType.NAVIGATION and any(word in goal_lower for word in ['navigate', 'go to', 'visit']):
                score += 0.8
            elif skill.skill_type == SkillType.SEARCH_OPERATION and any(word in goal_lower for word in ['search', 'find', 'query']):
                score += 0.8
            
            # Match by domain
            domains = skill.metadata.get('domains', [])
            for domain in domains:
                if domain.lower() in goal_lower:
                    score += 0.3
            
            # Boost by success rate and usage
            score *= skill.success_rate
            if skill.usage_count > 0:
                score += min(skill.usage_count * 0.1, 0.2)
            
            if score > 0.3:
                recommendations.append((skill, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in recommendations[:5]]
    
    def get_mining_stats(self) -> Dict[str, Any]:
        """Get skill mining statistics"""
        return {
            'traces_analyzed': self.stats['traces_analyzed'],
            'patterns_identified': self.stats['patterns_identified'],
            'skills_created': self.stats['skills_created'],
            'skill_usage_count': self.stats['skill_usage_count'],
            'total_skills': len(self.validated_skills),
            'active_patterns': self.stats['patterns_identified'],  # Fixed: add missing active_patterns
            'learning_rate': min(1.0, self.stats['skills_created'] / max(1, self.stats['traces_analyzed'])),  # Fixed: add missing learning_rate
            'skill_types': dict(Counter(skill.skill_type.value for skill in self.validated_skills.values())),
            'avg_skill_success_rate': statistics.mean([skill.success_rate for skill in self.validated_skills.values()]) if self.validated_skills else 0.0
        }

# Global instance
_skill_mining_ai_instance = None

def get_skill_mining_ai(config: Dict[str, Any] = None) -> AutoSkillMiningAI:
    """Get global skill mining AI instance"""
    global _skill_mining_ai_instance
    
    if _skill_mining_ai_instance is None:
        default_config = {
            'min_pattern_frequency': 2,
            'min_confidence_threshold': 0.6,
            'max_skill_complexity': SkillComplexity.EXPERT
        }
        
        _skill_mining_ai_instance = AutoSkillMiningAI(config or default_config)
    
    return _skill_mining_ai_instance

if __name__ == "__main__":
    # Demo the skill mining system
    async def demo():
        print("⛏️ Auto Skill-Mining AI Demo")
        print("=" * 50)
        
        mining_ai = get_skill_mining_ai()
        
        # Mock execution trace
        mock_trace = ExecutionTrace(
            trace_id="trace_001",
            goal="Login to email account",
            steps=[
                {
                    "id": "step_1",
                    "goal": "navigate_to_login",
                    "action": {"type": "navigate", "url": "https://mail.example.com/login"}
                },
                {
                    "id": "step_2",
                    "goal": "enter_username",
                    "action": {"type": "type", "target": {"role": "textbox", "name": "username"}, "text": "user@example.com"}
                },
                {
                    "id": "step_3",
                    "goal": "enter_password",
                    "action": {"type": "type", "target": {"role": "textbox", "name": "password"}, "text": "password123"}
                },
                {
                    "id": "step_4",
                    "goal": "click_login",
                    "action": {"type": "click", "target": {"role": "button", "name": "Login"}}
                }
            ],
            execution_time_ms=3500,
            success_rate=0.95,
            context={"platform": "web", "domain": "email"},
            timestamp=time.time(),
            platform="web",
            domain="email"
        )
        
        # Analyze trace
        patterns = await mining_ai.analyze_execution_trace(mock_trace)
        print(f"✅ Identified {len(patterns)} patterns")
        
        for pattern in patterns:
            print(f"  - {pattern.skill_type.value} (confidence: {pattern.confidence_score:.2f})")
        
        # Mine skills
        skills = await mining_ai.mine_skills_from_patterns(min_frequency=1)  # Lower threshold for demo
        print(f"\n⛏️ Mined {len(skills)} skills")
        
        for skill in skills:
            print(f"  - {skill.name}: {skill.description}")
            print(f"    Success rate: {skill.success_rate:.2f}")
            print(f"    Test cases: {len(skill.test_cases)}")
        
        # Test skill recommendations
        recommendations = mining_ai.get_skill_recommendations("Login to my account", {})
        print(f"\n💡 Skill recommendations for 'Login to my account':")
        
        for skill in recommendations:
            print(f"  - {skill.name} (success rate: {skill.success_rate:.2f})")
        
        # Show stats
        stats = mining_ai.get_mining_stats()
        print(f"\n📊 Mining Stats:")
        print(f"  Traces analyzed: {stats['traces_analyzed']}")
        print(f"  Patterns identified: {stats['patterns_identified']}")
        print(f"  Skills created: {stats['skills_created']}")
        print(f"  Avg success rate: {stats['avg_skill_success_rate']:.2f}")
        
        print("\n✅ Skill mining demo complete!")
        print("🏆 AI-powered pattern recognition and skill abstraction!")
    
    asyncio.run(demo())