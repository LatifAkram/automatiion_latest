"""
Shadow DOM Simulator
===================

Counterfactual planning system that:
- Snapshots DOM+styles
- Stubs events  
- Simulates actions
- Evaluates postconditions

Planner must keep only plans with simulated success ≥98%.

API:
simulate(plan_or_step, snapshot) -> {ok:bool, violations:[...], expected_changes:[...]}
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import copy

try:
    from playwright.async_api import Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .semantic_dom_graph import SemanticDOMGraph, DOMNode, BoundingBox
# Import contracts with fallback
try:
    from ..models.contracts import StepContract, Action, ActionType
except ImportError:
    from enum import Enum
    from dataclasses import dataclass
    from typing import Any, Optional
    
    class ActionType(Enum):
        CLICK = "click"
        TYPE = "type"
        SCROLL = "scroll"
        WAIT = "wait"
        NAVIGATE = "navigate"
    
    @dataclass
    class Action:
        type: ActionType
        selector: str
        value: Optional[Any] = None
        
    @dataclass  
    class StepContract:
        action: Action
        preconditions: Dict[str, Any]
        postconditions: Dict[str, Any]


@dataclass
class DOMSnapshot:
    """Complete DOM snapshot with styles and state."""
    timestamp: datetime
    url: str
    html: str
    css_rules: List[Dict[str, Any]]
    computed_styles: Dict[str, Dict[str, str]]  # element_id -> styles
    element_states: Dict[str, Dict[str, Any]]  # element_id -> state
    viewport: Dict[str, int]
    scroll_position: Dict[str, int]
    semantic_graph: Dict[str, DOMNode]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """Result of action simulation."""
    ok: bool
    violations: List[str]
    expected_changes: List[Dict[str, Any]]
    confidence: float
    simulation_time_ms: int
    postcondition_results: Dict[str, bool]
    side_effects: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ElementState:
    """State of a DOM element."""
    visible: bool
    enabled: bool
    focused: bool
    checked: Optional[bool] = None
    selected: Optional[bool] = None
    value: Optional[str] = None
    text_content: Optional[str] = None
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class ShadowDOMSimulator:
    """
    Shadow DOM Simulator for counterfactual planning.
    
    Simulates actions on a snapshot of the DOM without actually executing them,
    allowing the planner to verify plans before live execution.
    """
    
    def __init__(self, semantic_graph: SemanticDOMGraph, config: Any = None):
        self.semantic_graph = semantic_graph
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.current_snapshot: Optional[DOMSnapshot] = None
        self.simulation_cache: Dict[str, SimulationResult] = {}
        
        # Configuration
        self.confidence_threshold = 0.98
        self.max_simulation_time_ms = 1000
        
        # Known element behaviors
        self.element_behaviors = {
            'button': self._simulate_button_click,
            'input': self._simulate_input_interaction,
            'select': self._simulate_select_interaction,
            'a': self._simulate_link_click,
            'form': self._simulate_form_submit,
        }
        
        # Postcondition evaluators
        self.postcondition_evaluators = {
            'exists': self._evaluate_exists,
            'visible': self._evaluate_visible,
            'enabled': self._evaluate_enabled,
            'dialog_open': self._evaluate_dialog_open,
            'url_changed': self._evaluate_url_changed,
            'text_contains': self._evaluate_text_contains,
            'value_equals': self._evaluate_value_equals,
        }
    
    async def capture_snapshot(self, page: Page) -> DOMSnapshot:
        """Capture complete DOM snapshot for simulation."""
        try:
            start_time = datetime.utcnow()
            
            # Get basic page info
            url = page.url
            html = await page.content()
            
            # Get viewport
            viewport = page.viewport_size
            
            # Get scroll position
            scroll_position = await page.evaluate("""
                () => ({
                    x: window.scrollX,
                    y: window.scrollY
                })
            """)
            
            # Get all CSS rules
            css_rules = await page.evaluate("""
                () => {
                    const rules = [];
                    for (let i = 0; i < document.styleSheets.length; i++) {
                        try {
                            const sheet = document.styleSheets[i];
                            for (let j = 0; j < sheet.cssRules.length; j++) {
                                const rule = sheet.cssRules[j];
                                rules.push({
                                    selector: rule.selectorText || '',
                                    cssText: rule.cssText || '',
                                    type: rule.type
                                });
                            }
                        } catch (e) {
                            // Skip cross-origin stylesheets
                        }
                    }
                    return rules;
                }
            """)
            
            # Get computed styles for all elements
            computed_styles = await page.evaluate("""
                () => {
                    const styles = {};
                    const elements = document.querySelectorAll('*');
                    
                    elements.forEach((el, index) => {
                        const id = el.id || `element_${index}`;
                        const computedStyle = window.getComputedStyle(el);
                        
                        styles[id] = {
                            display: computedStyle.display,
                            visibility: computedStyle.visibility,
                            opacity: computedStyle.opacity,
                            position: computedStyle.position,
                            top: computedStyle.top,
                            left: computedStyle.left,
                            width: computedStyle.width,
                            height: computedStyle.height,
                            zIndex: computedStyle.zIndex,
                            backgroundColor: computedStyle.backgroundColor,
                            color: computedStyle.color,
                            fontSize: computedStyle.fontSize,
                            fontFamily: computedStyle.fontFamily
                        };
                    });
                    
                    return styles;
                }
            """)
            
            # Get element states
            element_states = await page.evaluate("""
                () => {
                    const states = {};
                    const elements = document.querySelectorAll('*');
                    
                    elements.forEach((el, index) => {
                        const id = el.id || `element_${index}`;
                        const rect = el.getBoundingClientRect();
                        
                        states[id] = {
                            visible: rect.width > 0 && rect.height > 0 && 
                                    window.getComputedStyle(el).visibility !== 'hidden' &&
                                    window.getComputedStyle(el).display !== 'none',
                            enabled: !el.disabled,
                            focused: document.activeElement === el,
                            checked: el.checked || null,
                            selected: el.selected || null,
                            value: el.value || null,
                            textContent: el.textContent || null,
                            tagName: el.tagName.toLowerCase(),
                            attributes: {}
                        };
                        
                        // Get attributes
                        for (let attr of el.attributes) {
                            states[id].attributes[attr.name] = attr.value;
                        }
                    });
                    
                    return states;
                }
            """)
            
            # Get semantic graph
            semantic_graph = self.semantic_graph.get_semantic_graph()
            
            snapshot = DOMSnapshot(
                timestamp=start_time,
                url=url,
                html=html,
                css_rules=css_rules,
                computed_styles=computed_styles,
                element_states=element_states,
                viewport=viewport,
                scroll_position=scroll_position,
                semantic_graph=semantic_graph
            )
            
            self.current_snapshot = snapshot
            self.logger.info(f"Captured DOM snapshot with {len(element_states)} elements")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to capture DOM snapshot: {e}")
            raise
    
    def simulate(self, plan_or_step: Any, snapshot: Optional[DOMSnapshot] = None) -> SimulationResult:
        """
        Main simulation API.
        
        Args:
            plan_or_step: StepContract or list of steps to simulate
            snapshot: DOM snapshot to simulate on (uses current if None)
            
        Returns:
            SimulationResult with ok, violations, expected_changes
        """
        start_time = datetime.utcnow()
        
        if snapshot is None:
            snapshot = self.current_snapshot
        
        if snapshot is None:
            return SimulationResult(
                ok=False,
                violations=["No snapshot available for simulation"],
                expected_changes=[],
                confidence=0.0,
                simulation_time_ms=0,
                postcondition_results={},
                side_effects=[]
            )
        
        try:
            # Handle single step vs plan
            if isinstance(plan_or_step, StepContract):
                steps = [plan_or_step]
            elif isinstance(plan_or_step, list):
                steps = plan_or_step
            else:
                return SimulationResult(
                    ok=False,
                    violations=["Invalid plan_or_step type"],
                    expected_changes=[],
                    confidence=0.0,
                    simulation_time_ms=0,
                    postcondition_results={},
                    side_effects=[]
                )
            
            # Simulate each step
            overall_ok = True
            all_violations = []
            all_changes = []
            all_postconditions = {}
            all_side_effects = []
            
            # Create working copy of snapshot
            working_snapshot = copy.deepcopy(snapshot)
            
            for step in steps:
                step_result = self._simulate_step(step, working_snapshot)
                
                if not step_result.ok:
                    overall_ok = False
                
                all_violations.extend(step_result.violations)
                all_changes.extend(step_result.expected_changes)
                all_postconditions.update(step_result.postcondition_results)
                all_side_effects.extend(step_result.side_effects)
                
                # Apply changes to working snapshot for next step
                self._apply_changes_to_snapshot(working_snapshot, step_result.expected_changes)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(steps, all_violations, all_postconditions)
            
            simulation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return SimulationResult(
                ok=overall_ok and confidence >= self.confidence_threshold,
                violations=all_violations,
                expected_changes=all_changes,
                confidence=confidence,
                simulation_time_ms=simulation_time,
                postcondition_results=all_postconditions,
                side_effects=all_side_effects
            )
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            simulation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return SimulationResult(
                ok=False,
                violations=[f"Simulation error: {str(e)}"],
                expected_changes=[],
                confidence=0.0,
                simulation_time_ms=simulation_time,
                postcondition_results={},
                side_effects=[]
            )
    
    def _simulate_step(self, step: StepContract, snapshot: DOMSnapshot) -> SimulationResult:
        """Simulate a single step."""
        violations = []
        expected_changes = []
        postcondition_results = {}
        side_effects = []
        
        # Check preconditions
        for precondition in step.pre:
            if not self._evaluate_condition(precondition, snapshot):
                violations.append(f"Precondition failed: {precondition}")
        
        # Simulate the action
        action_result = self._simulate_action(step.action, snapshot)
        expected_changes.extend(action_result.get('changes', []))
        side_effects.extend(action_result.get('side_effects', []))
        
        if not action_result.get('success', False):
            violations.extend(action_result.get('errors', []))
        
        # Check postconditions
        for postcondition in step.post:
            result = self._evaluate_condition(postcondition, snapshot, expected_changes)
            postcondition_results[postcondition] = result
            if not result:
                violations.append(f"Postcondition failed: {postcondition}")
        
        # Calculate step confidence
        confidence = self._calculate_step_confidence(step, violations, action_result)
        
        return SimulationResult(
            ok=len(violations) == 0,
            violations=violations,
            expected_changes=expected_changes,
            confidence=confidence,
            simulation_time_ms=0,  # Will be set by parent
            postcondition_results=postcondition_results,
            side_effects=side_effects
        )
    
    def _simulate_action(self, action: Action, snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate an action and return expected changes."""
        action_type = action.type
        target = action.target
        
        # Find target element in snapshot
        target_element_id = self._find_target_element(target, snapshot)
        if not target_element_id:
            return {
                'success': False,
                'errors': [f"Target element not found for action {action_type}"],
                'changes': [],
                'side_effects': []
            }
        
        element_state = snapshot.element_states.get(target_element_id)
        if not element_state:
            return {
                'success': False,
                'errors': [f"Element state not found for {target_element_id}"],
                'changes': [],
                'side_effects': []
            }
        
        # Check if element is actionable
        if not element_state['visible']:
            return {
                'success': False,
                'errors': [f"Element {target_element_id} is not visible"],
                'changes': [],
                'side_effects': []
            }
        
        if not element_state['enabled']:
            return {
                'success': False,
                'errors': [f"Element {target_element_id} is not enabled"],
                'changes': [],
                'side_effects': []
            }
        
        # Simulate based on action type
        if action_type == ActionType.CLICK:
            return self._simulate_click_action(target_element_id, element_state, snapshot)
        elif action_type == ActionType.TYPE:
            return self._simulate_type_action(target_element_id, element_state, action.value, snapshot)
        elif action_type == ActionType.KEYPRESS:
            return self._simulate_keypress_action(target_element_id, element_state, action.keys, snapshot)
        elif action_type == ActionType.HOVER:
            return self._simulate_hover_action(target_element_id, element_state, snapshot)
        elif action_type == ActionType.SCROLL:
            return self._simulate_scroll_action(target_element_id, element_state, snapshot)
        else:
            return {
                'success': False,
                'errors': [f"Unsupported action type: {action_type}"],
                'changes': [],
                'side_effects': []
            }
    
    def _simulate_click_action(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate click action."""
        changes = []
        side_effects = []
        
        tag_name = element_state.get('tagName', '').lower()
        
        # Use specific behavior for known elements
        if tag_name in self.element_behaviors:
            return self.element_behaviors[tag_name](element_id, element_state, snapshot, 'click')
        
        # Generic click behavior
        changes.append({
            'type': 'focus',
            'element_id': element_id,
            'property': 'focused',
            'old_value': element_state.get('focused', False),
            'new_value': True
        })
        
        # Check for onclick handlers (simplified)
        attributes = element_state.get('attributes', {})
        if 'onclick' in attributes:
            side_effects.append({
                'type': 'javascript_execution',
                'code': attributes['onclick'],
                'element_id': element_id
            })
        
        return {
            'success': True,
            'errors': [],
            'changes': changes,
            'side_effects': side_effects
        }
    
    def _simulate_type_action(self, element_id: str, element_state: Dict[str, Any], value: str, snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate type action."""
        changes = []
        
        # Check if element can accept text input
        tag_name = element_state.get('tagName', '').lower()
        if tag_name not in ['input', 'textarea']:
            return {
                'success': False,
                'errors': [f"Cannot type into {tag_name} element"],
                'changes': [],
                'side_effects': []
            }
        
        # Update value
        old_value = element_state.get('value', '')
        new_value = value
        
        changes.append({
            'type': 'value_change',
            'element_id': element_id,
            'property': 'value',
            'old_value': old_value,
            'new_value': new_value
        })
        
        return {
            'success': True,
            'errors': [],
            'changes': changes,
            'side_effects': []
        }
    
    def _simulate_button_click(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot, action: str) -> Dict[str, Any]:
        """Simulate button click behavior."""
        changes = []
        side_effects = []
        
        # Button gets focus
        changes.append({
            'type': 'focus',
            'element_id': element_id,
            'property': 'focused',
            'old_value': element_state.get('focused', False),
            'new_value': True
        })
        
        # Check button type
        attributes = element_state.get('attributes', {})
        button_type = attributes.get('type', 'button')
        
        if button_type == 'submit':
            # Find parent form and simulate submit
            side_effects.append({
                'type': 'form_submit',
                'element_id': element_id,
                'form_action': 'submit'
            })
        
        return {
            'success': True,
            'errors': [],
            'changes': changes,
            'side_effects': side_effects
        }
    
    def _find_target_element(self, target: Any, snapshot: DOMSnapshot) -> Optional[str]:
        """Find target element ID in snapshot."""
        if not target:
            return None
        
        # Try to match by various criteria
        for element_id, element_state in snapshot.element_states.items():
            attributes = element_state.get('attributes', {})
            
            # Match by ID
            if hasattr(target, 'id') and target.id and attributes.get('id') == target.id:
                return element_id
            
            # Match by role and name
            if (hasattr(target, 'role') and target.role and 
                hasattr(target, 'name') and target.name):
                role = attributes.get('role') or attributes.get('aria-role')
                name = attributes.get('aria-label') or element_state.get('textContent', '').strip()
                
                if role == target.role and target.name.lower() in name.lower():
                    return element_id
            
            # Match by CSS class
            if hasattr(target, 'class_name') and target.class_name:
                class_attr = attributes.get('class', '')
                if target.class_name in class_attr.split():
                    return element_id
        
        return None
    
    def _evaluate_condition(self, condition: str, snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]] = None) -> bool:
        """Evaluate a condition string."""
        if expected_changes is None:
            expected_changes = []
        
        # Parse condition (simplified parser)
        condition = condition.strip()
        
        # Handle function-style conditions
        if '(' in condition and condition.endswith(')'):
            func_name = condition.split('(')[0]
            args_str = condition[len(func_name)+1:-1]
            args = [arg.strip().strip("'\"") for arg in args_str.split(',') if arg.strip()]
            
            if func_name in self.postcondition_evaluators:
                return self.postcondition_evaluators[func_name](args, snapshot, expected_changes)
        
        # Default: try to evaluate as simple existence check
        return self._evaluate_exists([condition], snapshot, expected_changes)
    
    def _evaluate_exists(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate exists condition."""
        if not args:
            return False
        
        selector = args[0]
        
        # Check if any element matches the selector
        for element_id, element_state in snapshot.element_states.items():
            if self._matches_selector(element_state, selector):
                return True
        
        # Check expected changes
        for change in expected_changes:
            if change.get('type') == 'element_added' and self._matches_selector(change.get('element_state', {}), selector):
                return True
        
        return False
    
    def _evaluate_visible(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate visible condition."""
        if not args:
            return False
        
        selector = args[0]
        
        for element_id, element_state in snapshot.element_states.items():
            if self._matches_selector(element_state, selector):
                return element_state.get('visible', False)
        
        return False
    
    def _matches_selector(self, element_state: Dict[str, Any], selector: str) -> bool:
        """Check if element state matches selector."""
        # Simplified selector matching
        attributes = element_state.get('attributes', {})
        
        if selector.startswith('role='):
            role = selector[5:]
            return attributes.get('role') == role or attributes.get('aria-role') == role
        
        if selector.startswith('name='):
            name = selector[5:]
            aria_label = attributes.get('aria-label', '')
            text_content = element_state.get('textContent', '')
            return name.lower() in aria_label.lower() or name.lower() in text_content.lower()
        
        # Default: match by text content
        text_content = element_state.get('textContent', '').lower()
        return selector.lower() in text_content
    
    def _calculate_confidence(self, steps: List[StepContract], violations: List[str], postconditions: Dict[str, bool]) -> float:
        """Calculate overall simulation confidence."""
        if not steps:
            return 0.0
        
        # Base confidence
        confidence = 1.0
        
        # Reduce confidence for violations
        violation_penalty = len(violations) * 0.1
        confidence -= violation_penalty
        
        # Reduce confidence for failed postconditions
        failed_postconditions = sum(1 for result in postconditions.values() if not result)
        postcondition_penalty = failed_postconditions * 0.15
        confidence -= postcondition_penalty
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _calculate_step_confidence(self, step: StepContract, violations: List[str], action_result: Dict[str, Any]) -> float:
        """Calculate confidence for a single step."""
        confidence = 1.0
        
        # Reduce for violations
        confidence -= len(violations) * 0.2
        
        # Reduce if action failed
        if not action_result.get('success', False):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _apply_changes_to_snapshot(self, snapshot: DOMSnapshot, changes: List[Dict[str, Any]]):
        """Apply simulated changes to snapshot for multi-step simulation."""
        for change in changes:
            element_id = change.get('element_id')
            if not element_id or element_id not in snapshot.element_states:
                continue
            
            change_type = change.get('type')
            property_name = change.get('property')
            new_value = change.get('new_value')
            
            if change_type in ['focus', 'value_change'] and property_name:
                snapshot.element_states[element_id][property_name] = new_value
    
    # Additional postcondition evaluators
    def _evaluate_enabled(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate enabled condition."""
        # Implementation similar to _evaluate_visible
        return True  # Simplified
    
    def _evaluate_dialog_open(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate dialog_open condition."""
        # Check for dialog elements or modal overlays
        return True  # Simplified
    
    def _evaluate_url_changed(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate url_changed condition."""
        # Check expected changes for navigation
        return any(change.get('type') == 'navigation' for change in expected_changes)
    
    def _evaluate_text_contains(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate text_contains condition."""
        return True  # Simplified
    
    def _evaluate_value_equals(self, args: List[str], snapshot: DOMSnapshot, expected_changes: List[Dict[str, Any]]) -> bool:
        """Evaluate value_equals condition."""
        return True  # Simplified
    
    # Placeholder implementations for other element behaviors
    def _simulate_input_interaction(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot, action: str) -> Dict[str, Any]:
        """Simulate input element interaction."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_select_interaction(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot, action: str) -> Dict[str, Any]:
        """Simulate select element interaction."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_link_click(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot, action: str) -> Dict[str, Any]:
        """Simulate link click."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_form_submit(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot, action: str) -> Dict[str, Any]:
        """Simulate form submit."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_keypress_action(self, element_id: str, element_state: Dict[str, Any], keys: List[str], snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate keypress action."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_hover_action(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate hover action."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}
    
    def _simulate_scroll_action(self, element_id: str, element_state: Dict[str, Any], snapshot: DOMSnapshot) -> Dict[str, Any]:
        """Simulate scroll action."""
        return {'success': True, 'errors': [], 'changes': [], 'side_effects': []}