"""
DYNAMIC WAITING SYSTEM
======================

Intelligent waiting strategies for automation that adapt to page conditions,
network speeds, and element states for optimal performance.

✅ FEATURES:
- Adaptive waiting based on page load states
- Network condition awareness
- Element-specific waiting strategies
- Performance optimization
- Timeout management
- Error recovery
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from playwright.async_api import Page, Locator

logger = logging.getLogger(__name__)

class WaitStrategy(Enum):
    FIXED_TIME = "fixed_time"
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_HIDDEN = "element_hidden"
    ELEMENT_STABLE = "element_stable"
    NETWORK_IDLE = "network_idle"
    DOM_CONTENT_LOADED = "dom_content_loaded"
    LOAD_STATE = "load_state"
    ADAPTIVE = "adaptive"
    CUSTOM_CONDITION = "custom_condition"

class NetworkCondition(Enum):
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"
    VERY_SLOW = "very_slow"

@dataclass
class WaitCondition:
    """Wait condition configuration"""
    strategy: WaitStrategy
    selector: str = ""
    timeout: float = 30.0
    poll_interval: float = 0.5
    custom_function: Optional[Callable] = None
    network_aware: bool = True
    adaptive: bool = True

class DynamicWaitingSystem:
    """Intelligent dynamic waiting system"""
    
    def __init__(self):
        self.network_condition = NetworkCondition.NORMAL
        self.page_performance_history = {}
        self.element_load_times = {}
        self.adaptive_timeouts = {}
        
        # Base timeouts for different network conditions
        self.base_timeouts = {
            NetworkCondition.FAST: {
                'element_wait': 5.0,
                'network_idle': 2.0,
                'page_load': 10.0
            },
            NetworkCondition.NORMAL: {
                'element_wait': 10.0,
                'network_idle': 5.0,
                'page_load': 20.0
            },
            NetworkCondition.SLOW: {
                'element_wait': 20.0,
                'network_idle': 10.0,
                'page_load': 40.0
            },
            NetworkCondition.VERY_SLOW: {
                'element_wait': 30.0,
                'network_idle': 15.0,
                'page_load': 60.0
            }
        }
    
    async def smart_wait(self, page: Page, condition: WaitCondition) -> Dict[str, Any]:
        """Intelligent waiting with adaptive strategies"""
        start_time = time.time()
        
        try:
            # Detect network condition if not already known
            if condition.network_aware:
                await self._detect_network_condition(page)
            
            # Adjust timeout based on network condition and history
            adjusted_timeout = self._calculate_adaptive_timeout(condition)
            
            # Execute appropriate waiting strategy
            if condition.strategy == WaitStrategy.ADAPTIVE:
                result = await self._adaptive_wait(page, condition, adjusted_timeout)
            else:
                result = await self._execute_wait_strategy(page, condition, adjusted_timeout)
            
            # Record performance for future optimization
            wait_time = time.time() - start_time
            self._record_wait_performance(condition, wait_time, result['success'])
            
            result['wait_time'] = wait_time
            result['timeout_used'] = adjusted_timeout
            result['network_condition'] = self.network_condition.value
            
            return result
            
        except Exception as e:
            logger.error(f"Smart wait failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'wait_time': time.time() - start_time
            }
    
    async def _detect_network_condition(self, page: Page):
        """Detect current network condition"""
        try:
            # Measure page response time
            start_time = time.time()
            await page.evaluate('document.readyState')
            response_time = time.time() - start_time
            
            # Classify network condition based on response time
            if response_time < 0.1:
                self.network_condition = NetworkCondition.FAST
            elif response_time < 0.5:
                self.network_condition = NetworkCondition.NORMAL
            elif response_time < 2.0:
                self.network_condition = NetworkCondition.SLOW
            else:
                self.network_condition = NetworkCondition.VERY_SLOW
                
        except:
            # Default to normal if detection fails
            self.network_condition = NetworkCondition.NORMAL
    
    def _calculate_adaptive_timeout(self, condition: WaitCondition) -> float:
        """Calculate adaptive timeout based on history and network condition"""
        base_timeout = condition.timeout
        
        if condition.adaptive:
            # Get base timeout for current network condition
            network_timeouts = self.base_timeouts[self.network_condition]
            
            if condition.strategy == WaitStrategy.ELEMENT_VISIBLE:
                base_timeout = network_timeouts['element_wait']
            elif condition.strategy == WaitStrategy.NETWORK_IDLE:
                base_timeout = network_timeouts['network_idle']
            elif condition.strategy == WaitStrategy.LOAD_STATE:
                base_timeout = network_timeouts['page_load']
            
            # Adjust based on historical performance
            selector_key = f"{condition.strategy.value}:{condition.selector}"
            if selector_key in self.adaptive_timeouts:
                historical_avg = self.adaptive_timeouts[selector_key]['avg_time']
                success_rate = self.adaptive_timeouts[selector_key]['success_rate']
                
                # Increase timeout if success rate is low
                if success_rate < 0.8:
                    base_timeout *= 1.5
                elif success_rate > 0.95:
                    base_timeout *= 0.8
                
                # Adjust based on historical average
                base_timeout = max(base_timeout, historical_avg * 1.2)
        
        return min(base_timeout, 120.0)  # Cap at 2 minutes
    
    async def _adaptive_wait(self, page: Page, condition: WaitCondition, timeout: float) -> Dict[str, Any]:
        """Adaptive waiting that combines multiple strategies"""
        try:
            # Start with fastest strategy and progressively use more robust ones
            strategies_to_try = [
                WaitStrategy.ELEMENT_VISIBLE,
                WaitStrategy.ELEMENT_STABLE,
                WaitStrategy.DOM_CONTENT_LOADED,
                WaitStrategy.NETWORK_IDLE
            ]
            
            for strategy in strategies_to_try:
                try:
                    condition_copy = WaitCondition(
                        strategy=strategy,
                        selector=condition.selector,
                        timeout=timeout / len(strategies_to_try),
                        poll_interval=condition.poll_interval
                    )
                    
                    result = await self._execute_wait_strategy(page, condition_copy, condition_copy.timeout)
                    
                    if result['success']:
                        return {
                            'success': True,
                            'strategy_used': strategy.value,
                            'message': f'Adaptive wait succeeded with {strategy.value}'
                        }
                        
                except:
                    continue
            
            return {
                'success': False,
                'error': 'All adaptive strategies failed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Adaptive wait failed: {e}'
            }
    
    async def _execute_wait_strategy(self, page: Page, condition: WaitCondition, timeout: float) -> Dict[str, Any]:
        """Execute specific waiting strategy"""
        try:
            if condition.strategy == WaitStrategy.FIXED_TIME:
                await asyncio.sleep(timeout)
                return {'success': True, 'message': f'Fixed wait of {timeout}s completed'}
            
            elif condition.strategy == WaitStrategy.ELEMENT_VISIBLE:
                if condition.selector:
                    await page.wait_for_selector(condition.selector, timeout=timeout * 1000)
                    return {'success': True, 'message': f'Element {condition.selector} became visible'}
                else:
                    return {'success': False, 'error': 'No selector provided for element wait'}
            
            elif condition.strategy == WaitStrategy.ELEMENT_HIDDEN:
                if condition.selector:
                    await page.wait_for_selector(condition.selector, state='hidden', timeout=timeout * 1000)
                    return {'success': True, 'message': f'Element {condition.selector} became hidden'}
                else:
                    return {'success': False, 'error': 'No selector provided for element hidden wait'}
            
            elif condition.strategy == WaitStrategy.ELEMENT_STABLE:
                if condition.selector:
                    # Wait for element to be stable (not changing position/size)
                    element = page.locator(condition.selector)
                    await self._wait_for_element_stable(element, timeout)
                    return {'success': True, 'message': f'Element {condition.selector} became stable'}
                else:
                    return {'success': False, 'error': 'No selector provided for stability wait'}
            
            elif condition.strategy == WaitStrategy.NETWORK_IDLE:
                await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
                return {'success': True, 'message': 'Network became idle'}
            
            elif condition.strategy == WaitStrategy.DOM_CONTENT_LOADED:
                await page.wait_for_load_state('domcontentloaded', timeout=timeout * 1000)
                return {'success': True, 'message': 'DOM content loaded'}
            
            elif condition.strategy == WaitStrategy.LOAD_STATE:
                await page.wait_for_load_state('load', timeout=timeout * 1000)
                return {'success': True, 'message': 'Page fully loaded'}
            
            elif condition.strategy == WaitStrategy.CUSTOM_CONDITION:
                if condition.custom_function:
                    result = await self._wait_for_custom_condition(page, condition.custom_function, timeout)
                    return result
                else:
                    return {'success': False, 'error': 'No custom function provided'}
            
            else:
                return {'success': False, 'error': f'Unknown wait strategy: {condition.strategy}'}
                
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f'Wait strategy {condition.strategy.value} timed out after {timeout}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Wait strategy {condition.strategy.value} failed: {e}'
            }
    
    async def _wait_for_element_stable(self, element: Locator, timeout: float):
        """Wait for element to be stable (position and size not changing)"""
        start_time = time.time()
        stable_duration = 1.0  # Element must be stable for 1 second
        last_box = None
        stable_start = None
        
        while time.time() - start_time < timeout:
            try:
                box = await element.bounding_box()
                
                if box:
                    if last_box and self._boxes_equal(box, last_box):
                        if stable_start is None:
                            stable_start = time.time()
                        elif time.time() - stable_start >= stable_duration:
                            return  # Element is stable
                    else:
                        stable_start = None
                    
                    last_box = box
                
                await asyncio.sleep(0.1)
                
            except:
                await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"Element did not become stable within {timeout}s")
    
    def _boxes_equal(self, box1: Dict, box2: Dict, tolerance: float = 1.0) -> bool:
        """Check if two bounding boxes are equal within tolerance"""
        return (
            abs(box1['x'] - box2['x']) <= tolerance and
            abs(box1['y'] - box2['y']) <= tolerance and
            abs(box1['width'] - box2['width']) <= tolerance and
            abs(box1['height'] - box2['height']) <= tolerance
        )
    
    async def _wait_for_custom_condition(self, page: Page, custom_function: Callable, timeout: float) -> Dict[str, Any]:
        """Wait for custom condition to be met"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if asyncio.iscoroutinefunction(custom_function):
                    result = await custom_function(page)
                else:
                    result = custom_function(page)
                
                if result:
                    return {
                        'success': True,
                        'message': 'Custom condition met'
                    }
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Custom condition failed: {e}'
                }
        
        return {
            'success': False,
            'error': f'Custom condition not met within {timeout}s'
        }
    
    def _record_wait_performance(self, condition: WaitCondition, wait_time: float, success: bool):
        """Record wait performance for future optimization"""
        selector_key = f"{condition.strategy.value}:{condition.selector}"
        
        if selector_key not in self.adaptive_timeouts:
            self.adaptive_timeouts[selector_key] = {
                'total_time': 0.0,
                'total_attempts': 0,
                'successful_attempts': 0,
                'avg_time': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.adaptive_timeouts[selector_key]
        stats['total_time'] += wait_time
        stats['total_attempts'] += 1
        
        if success:
            stats['successful_attempts'] += 1
        
        stats['avg_time'] = stats['total_time'] / stats['total_attempts']
        stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']
    
    async def wait_for_multiple_conditions(self, page: Page, conditions: List[WaitCondition], 
                                         strategy: str = "any") -> Dict[str, Any]:
        """Wait for multiple conditions with different strategies"""
        try:
            if strategy == "any":
                # Wait for any condition to be met
                tasks = [self.smart_wait(page, condition) for condition in conditions]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Get result from completed task
                completed_task = list(done)[0]
                result = await completed_task
                
                return {
                    'success': result['success'],
                    'strategy': 'any',
                    'completed_condition': result,
                    'total_conditions': len(conditions)
                }
            
            elif strategy == "all":
                # Wait for all conditions to be met
                results = await asyncio.gather(*[self.smart_wait(page, condition) for condition in conditions])
                
                success = all(result['success'] for result in results)
                
                return {
                    'success': success,
                    'strategy': 'all',
                    'results': results,
                    'total_conditions': len(conditions)
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Unknown multi-condition strategy: {strategy}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Multi-condition wait failed: {e}'
            }
    
    async def progressive_wait(self, page: Page, base_condition: WaitCondition, 
                            max_attempts: int = 3) -> Dict[str, Any]:
        """Progressive waiting with increasing timeouts"""
        for attempt in range(max_attempts):
            # Increase timeout with each attempt
            timeout_multiplier = 1.5 ** attempt
            adjusted_condition = WaitCondition(
                strategy=base_condition.strategy,
                selector=base_condition.selector,
                timeout=base_condition.timeout * timeout_multiplier,
                poll_interval=base_condition.poll_interval,
                custom_function=base_condition.custom_function,
                network_aware=base_condition.network_aware,
                adaptive=base_condition.adaptive
            )
            
            result = await self.smart_wait(page, adjusted_condition)
            
            if result['success']:
                result['attempt'] = attempt + 1
                result['total_attempts'] = max_attempts
                return result
            
            # Wait a bit before retrying
            if attempt < max_attempts - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
        
        return {
            'success': False,
            'error': f'Progressive wait failed after {max_attempts} attempts',
            'total_attempts': max_attempts
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for optimization"""
        return {
            'network_condition': self.network_condition.value,
            'adaptive_timeouts': self.adaptive_timeouts.copy(),
            'total_selectors_tracked': len(self.adaptive_timeouts)
        }
    
    def optimize_timeouts(self):
        """Optimize timeouts based on historical performance"""
        for selector_key, stats in self.adaptive_timeouts.items():
            if stats['total_attempts'] >= 10:  # Only optimize with sufficient data
                if stats['success_rate'] > 0.9:
                    # Reduce timeout for consistently successful operations
                    stats['avg_time'] *= 0.9
                elif stats['success_rate'] < 0.7:
                    # Increase timeout for frequently failing operations
                    stats['avg_time'] *= 1.2

# Global dynamic waiting system instance
_global_waiting_system: Optional[DynamicWaitingSystem] = None

def get_dynamic_waiting_system() -> DynamicWaitingSystem:
    """Get or create the global dynamic waiting system"""
    global _global_waiting_system
    
    if _global_waiting_system is None:
        _global_waiting_system = DynamicWaitingSystem()
    
    return _global_waiting_system

# Convenience functions for common wait operations
async def smart_wait_for_element(page: Page, selector: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Smart wait for element with adaptive timeout"""
    waiting_system = get_dynamic_waiting_system()
    condition = WaitCondition(
        strategy=WaitStrategy.ADAPTIVE,
        selector=selector,
        timeout=timeout
    )
    return await waiting_system.smart_wait(page, condition)

async def smart_wait_for_page_load(page: Page, timeout: float = 60.0) -> Dict[str, Any]:
    """Smart wait for page load with network awareness"""
    waiting_system = get_dynamic_waiting_system()
    condition = WaitCondition(
        strategy=WaitStrategy.LOAD_STATE,
        timeout=timeout
    )
    return await waiting_system.smart_wait(page, condition)