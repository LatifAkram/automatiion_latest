"""
Advanced Learning and Auto-Heal System
=====================================

Continuous learning system that improves automation performance
through pattern recognition, error analysis, and self-healing.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import statistics

from ..core.vector_store import VectorStore
from ..core.ai_provider import AIProvider


class AdvancedLearningSystem:
    """Advanced learning and auto-heal system for continuous improvement."""
    
    def __init__(self, config, ai_provider: AIProvider, vector_store: VectorStore):
        self.config = config
        self.ai_provider = ai_provider
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        # Learning metrics
        self.success_patterns = {}
        self.failure_patterns = {}
        self.performance_metrics = {}
        self.auto_heal_history = {}
        
    async def learn_from_execution(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from automation execution results."""
        try:
            self.logger.info("Learning from automation execution")
            
            # Extract learning data
            learning_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "success": execution_result.get("success", False),
                "execution_time": execution_result.get("execution_time", 0),
                "steps_completed": len(execution_result.get("completed_steps", [])),
                "total_steps": len(execution_result.get("all_steps", [])),
                "errors": execution_result.get("errors", []),
                "selectors_used": execution_result.get("selectors_used", []),
                "domain": execution_result.get("domain", ""),
                "automation_type": execution_result.get("automation_type", ""),
                "ai_confidence": execution_result.get("ai_confidence", 0.0)
            }
            
            # Analyze patterns
            patterns = await self._analyze_execution_patterns(learning_data)
            
            # Store learning data
            await self._store_learning_data(learning_data, patterns)
            
            # Update performance metrics
            await self._update_performance_metrics(learning_data)
            
            # Generate improvement suggestions
            improvements = await self._generate_improvement_suggestions(learning_data, patterns)
            
            return {
                "learning_applied": True,
                "patterns_identified": len(patterns),
                "performance_improvement": improvements.get("performance_gain", 0),
                "suggestions": improvements.get("suggestions", [])
            }
            
        except Exception as e:
            self.logger.error(f"Learning from execution failed: {e}")
            return {"learning_applied": False, "error": str(e)}

    async def auto_heal_selector(self, failed_selector: str, context: Dict[str, Any]) -> Optional[str]:
        """Auto-heal a failed selector using learned patterns."""
        try:
            self.logger.info(f"Attempting to auto-heal selector: {failed_selector}")
            
            # Search for similar successful patterns
            similar_patterns = await self.vector_store.search(
                query=f"successful selector pattern similar to {failed_selector}",
                limit=5
            )
            
            # Analyze failure context
            failure_context = {
                "failed_selector": failed_selector,
                "domain": context.get("domain", ""),
                "element_type": context.get("element_type", ""),
                "page_context": context.get("page_context", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Generate healing strategies
            healing_strategies = await self._generate_healing_strategies(failed_selector, failure_context, similar_patterns)
            
            # Test healing strategies
            for strategy in healing_strategies:
                healed_selector = await self._test_healing_strategy(strategy, context)
                if healed_selector:
                    # Store successful healing
                    await self._store_healing_success(failed_selector, healed_selector, strategy)
                    self.logger.info(f"Successfully healed selector: {failed_selector} -> {healed_selector}")
                    return healed_selector
            
            # Store failed healing attempt
            await self._store_healing_failure(failed_selector, failure_context, healing_strategies)
            self.logger.warning(f"Failed to heal selector: {failed_selector}")
            return None
            
        except Exception as e:
            self.logger.error(f"Auto-heal selector failed: {e}")
            return None

    async def predict_automation_success(self, automation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Predict automation success probability using learned patterns."""
        try:
            # Extract prediction features
            features = {
                "domain": automation_request.get("domain", ""),
                "automation_type": automation_request.get("automation_type", ""),
                "complexity": automation_request.get("complexity", "medium"),
                "selectors_available": len(automation_request.get("selectors", [])),
                "ai_confidence": automation_request.get("ai_confidence", 0.0)
            }
            
            # Search for similar past automations
            similar_automations = await self.vector_store.search(
                query=f"automation success pattern for {features['domain']} {features['automation_type']}",
                limit=10
            )
            
            # Calculate success probability
            success_probability = await self._calculate_success_probability(features, similar_automations)
            
            # Generate recommendations
            recommendations = await self._generate_success_recommendations(features, similar_automations)
            
            return {
                "success_probability": success_probability,
                "confidence": self._calculate_prediction_confidence(features, similar_automations),
                "recommendations": recommendations,
                "risk_factors": self._identify_risk_factors(features, similar_automations)
            }
            
        except Exception as e:
            self.logger.error(f"Success prediction failed: {e}")
            return {
                "success_probability": 0.5,
                "confidence": 0.0,
                "recommendations": ["Unable to predict due to insufficient data"],
                "risk_factors": ["Unknown"]
            }

    async def optimize_automation_plan(self, automation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize automation plan using learned patterns."""
        try:
            self.logger.info("Optimizing automation plan with learned patterns")
            
            # Analyze plan efficiency
            efficiency_analysis = await self._analyze_plan_efficiency(automation_plan)
            
            # Generate optimizations
            optimizations = await self._generate_plan_optimizations(automation_plan, efficiency_analysis)
            
            # Apply optimizations
            optimized_plan = await self._apply_optimizations(automation_plan, optimizations)
            
            # Validate optimized plan
            validation_result = await self._validate_optimized_plan(optimized_plan)
            
            return {
                "original_plan": automation_plan,
                "optimized_plan": optimized_plan,
                "optimizations_applied": optimizations,
                "efficiency_improvement": efficiency_analysis.get("improvement_potential", 0),
                "validation": validation_result
            }
            
        except Exception as e:
            self.logger.error(f"Plan optimization failed: {e}")
            return {"optimized_plan": automation_plan, "error": str(e)}

    async def _analyze_execution_patterns(self, learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze execution patterns for learning."""
        try:
            patterns = []
            
            # Success patterns
            if learning_data["success"]:
                success_pattern = {
                    "type": "success",
                    "domain": learning_data["domain"],
                    "automation_type": learning_data["automation_type"],
                    "selectors": learning_data["selectors_used"],
                    "execution_time": learning_data["execution_time"],
                    "ai_confidence": learning_data["ai_confidence"],
                    "timestamp": learning_data["timestamp"]
                }
                patterns.append(success_pattern)
            
            # Failure patterns
            if not learning_data["success"] and learning_data["errors"]:
                failure_pattern = {
                    "type": "failure",
                    "domain": learning_data["domain"],
                    "automation_type": learning_data["automation_type"],
                    "errors": learning_data["errors"],
                    "failed_selectors": [e.get("selector") for e in learning_data["errors"] if e.get("selector")],
                    "timestamp": learning_data["timestamp"]
                }
                patterns.append(failure_pattern)
            
            # Performance patterns
            performance_pattern = {
                "type": "performance",
                "execution_time": learning_data["execution_time"],
                "steps_completed": learning_data["steps_completed"],
                "total_steps": learning_data["total_steps"],
                "efficiency": learning_data["steps_completed"] / max(learning_data["total_steps"], 1),
                "timestamp": learning_data["timestamp"]
            }
            patterns.append(performance_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return []

    async def _store_learning_data(self, learning_data: Dict[str, Any], patterns: List[Dict[str, Any]]):
        """Store learning data in vector store."""
        try:
            # Store execution result
            await self.vector_store.add_document(
                content=f"Automation execution result: {learning_data['success']} for {learning_data['domain']}",
                metadata={
                    **learning_data,
                    "patterns": patterns,
                    "document_type": "execution_result"
                }
            )
            
            # Store individual patterns
            for pattern in patterns:
                await self.vector_store.add_document(
                    content=f"Automation pattern: {pattern['type']} for {pattern.get('domain', 'unknown')}",
                    metadata={
                        **pattern,
                        "document_type": "pattern"
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store learning data: {e}")

    async def _update_performance_metrics(self, learning_data: Dict[str, Any]):
        """Update performance metrics."""
        try:
            domain = learning_data["domain"]
            automation_type = learning_data["automation_type"]
            
            # Initialize metrics if not exists
            if domain not in self.performance_metrics:
                self.performance_metrics[domain] = {}
            if automation_type not in self.performance_metrics[domain]:
                self.performance_metrics[domain][automation_type] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "execution_times": [],
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0
                }
            
            metrics = self.performance_metrics[domain][automation_type]
            
            # Update metrics
            metrics["total_executions"] += 1
            if learning_data["success"]:
                metrics["successful_executions"] += 1
            
            metrics["execution_times"].append(learning_data["execution_time"])
            metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
            metrics["avg_execution_time"] = statistics.mean(metrics["execution_times"])
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")

    async def _generate_improvement_suggestions(self, learning_data: Dict[str, Any], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate improvement suggestions based on learning data."""
        try:
            suggestions = []
            performance_gain = 0.0
            
            # Analyze success rate
            if not learning_data["success"]:
                suggestions.append({
                    "type": "error_prevention",
                    "description": "Implement additional error handling for failed selectors",
                    "priority": "high"
                })
            
            # Analyze execution time
            if learning_data["execution_time"] > 30:  # More than 30 seconds
                suggestions.append({
                    "type": "performance_optimization",
                    "description": "Optimize selector strategies for faster execution",
                    "priority": "medium"
                })
                performance_gain += 0.2
            
            # Analyze AI confidence
            if learning_data["ai_confidence"] < 0.7:
                suggestions.append({
                    "type": "ai_improvement",
                    "description": "Enhance AI analysis for better confidence scores",
                    "priority": "medium"
                })
            
            # Analyze step completion
            completion_rate = learning_data["steps_completed"] / max(learning_data["total_steps"], 1)
            if completion_rate < 0.8:
                suggestions.append({
                    "type": "planning_improvement",
                    "description": "Improve automation planning for better step completion",
                    "priority": "high"
                })
            
            return {
                "suggestions": suggestions,
                "performance_gain": performance_gain,
                "completion_rate": completion_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate improvement suggestions: {e}")
            return {"suggestions": [], "performance_gain": 0.0}

    async def _generate_healing_strategies(self, failed_selector: str, failure_context: Dict[str, Any], similar_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate healing strategies for failed selectors."""
        try:
            strategies = []
            
            # Strategy 1: Use similar successful selectors
            for pattern in similar_patterns:
                if pattern.get("metadata", {}).get("success_rate", 0) > 0.8:
                    strategies.append({
                        "type": "similar_selector",
                        "selector": pattern.get("metadata", {}).get("selector"),
                        "confidence": pattern.get("metadata", {}).get("success_rate", 0)
                    })
            
            # Strategy 2: AI-generated alternative selectors
            ai_strategies = await self._generate_ai_healing_strategies(failed_selector, failure_context)
            strategies.extend(ai_strategies)
            
            # Strategy 3: Fallback selectors
            fallback_selectors = self._generate_fallback_selectors(failed_selector, failure_context)
            strategies.extend(fallback_selectors)
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Failed to generate healing strategies: {e}")
            return []

    async def _generate_ai_healing_strategies(self, failed_selector: str, failure_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered healing strategies."""
        try:
            prompt = f"""
            Generate alternative selectors for a failed automation selector:
            
            Failed Selector: {failed_selector}
            Domain: {failure_context.get('domain', '')}
            Element Type: {failure_context.get('element_type', '')}
            Page Context: {failure_context.get('page_context', '')}
            
            Generate 3 alternative selectors that are:
            1. More robust and less likely to fail
            2. Specific to the element type and context
            3. Following best practices for web automation
            
            Return as JSON array of selectors with confidence scores:
            [
                {{
                    "selector": "alternative_selector_1",
                    "confidence": 0.85,
                    "strategy": "description_of_strategy"
                }}
            ]
            """
            
            response = await self.ai_provider.generate_response(prompt)
            
            try:
                strategies = json.loads(response)
                return [
                    {
                        "type": "ai_generated",
                        "selector": s.get("selector"),
                        "confidence": s.get("confidence", 0.5),
                        "strategy": s.get("strategy", "AI-generated alternative")
                    }
                    for s in strategies
                ]
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            self.logger.error(f"AI healing strategy generation failed: {e}")
            return []

    def _generate_fallback_selectors(self, failed_selector: str, failure_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback selectors based on common patterns."""
        try:
            fallbacks = []
            element_type = failure_context.get("element_type", "")
            
            # Generic fallback selectors
            if "input" in element_type.lower():
                fallbacks.extend([
                    {"type": "fallback", "selector": "input[type='text']", "confidence": 0.3},
                    {"type": "fallback", "selector": "input", "confidence": 0.2},
                    {"type": "fallback", "selector": "form input", "confidence": 0.4}
                ])
            elif "button" in element_type.lower():
                fallbacks.extend([
                    {"type": "fallback", "selector": "button", "confidence": 0.3},
                    {"type": "fallback", "selector": "input[type='submit']", "confidence": 0.3},
                    {"type": "fallback", "selector": "a[href]", "confidence": 0.2}
                ])
            elif "link" in element_type.lower():
                fallbacks.extend([
                    {"type": "fallback", "selector": "a[href]", "confidence": 0.4},
                    {"type": "fallback", "selector": "a", "confidence": 0.3}
                ])
            
            return fallbacks
            
        except Exception as e:
            self.logger.error(f"Fallback selector generation failed: {e}")
            return []

    async def _test_healing_strategy(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Test a healing strategy to see if it works."""
        try:
            # This would typically test the selector against the actual page
            # For now, we'll simulate testing
            selector = strategy.get("selector")
            confidence = strategy.get("confidence", 0.0)
            
            # Simulate testing (in real implementation, this would test against the page)
            if confidence > 0.5:
                return selector
            
            return None
            
        except Exception as e:
            self.logger.error(f"Strategy testing failed: {e}")
            return None

    async def _store_healing_success(self, original_selector: str, healed_selector: str, strategy: Dict[str, Any]):
        """Store successful healing attempt."""
        try:
            await self.vector_store.add_document(
                content=f"Successful selector healing: {original_selector} -> {healed_selector}",
                metadata={
                    "original_selector": original_selector,
                    "healed_selector": healed_selector,
                    "strategy": strategy,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "document_type": "healing_success"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store healing success: {e}")

    async def _store_healing_failure(self, failed_selector: str, failure_context: Dict[str, Any], strategies: List[Dict[str, Any]]):
        """Store failed healing attempt."""
        try:
            await self.vector_store.add_document(
                content=f"Failed selector healing: {failed_selector}",
                metadata={
                    "failed_selector": failed_selector,
                    "failure_context": failure_context,
                    "attempted_strategies": strategies,
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "document_type": "healing_failure"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store healing failure: {e}")

    async def _calculate_success_probability(self, features: Dict[str, Any], similar_automations: List[Dict[str, Any]]) -> float:
        """Calculate success probability based on similar automations."""
        try:
            if not similar_automations:
                return 0.5  # Default probability
            
            success_count = 0
            total_count = len(similar_automations)
            
            for automation in similar_automations:
                metadata = automation.get("metadata", {})
                if metadata.get("success", False):
                    success_count += 1
            
            base_probability = success_count / total_count
            
            # Adjust based on features
            adjustments = []
            
            # AI confidence adjustment
            if features["ai_confidence"] > 0.8:
                adjustments.append(0.1)
            elif features["ai_confidence"] < 0.5:
                adjustments.append(-0.1)
            
            # Selector availability adjustment
            if features["selectors_available"] > 5:
                adjustments.append(0.05)
            elif features["selectors_available"] < 2:
                adjustments.append(-0.1)
            
            # Complexity adjustment
            if features["complexity"] == "simple":
                adjustments.append(0.05)
            elif features["complexity"] == "complex":
                adjustments.append(-0.1)
            
            final_probability = base_probability + sum(adjustments)
            return max(0.0, min(1.0, final_probability))
            
        except Exception as e:
            self.logger.error(f"Success probability calculation failed: {e}")
            return 0.5

    def _calculate_prediction_confidence(self, features: Dict[str, Any], similar_automations: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the prediction."""
        try:
            if not similar_automations:
                return 0.0
            
            # Confidence based on number of similar cases
            base_confidence = min(len(similar_automations) / 10, 1.0)
            
            # Confidence based on feature similarity
            feature_confidence = 0.5  # Default
            
            return (base_confidence + feature_confidence) / 2
            
        except Exception as e:
            self.logger.error(f"Prediction confidence calculation failed: {e}")
            return 0.0

    async def _generate_success_recommendations(self, features: Dict[str, Any], similar_automations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving success probability."""
        try:
            recommendations = []
            
            # Analyze similar automations for patterns
            successful_automations = [a for a in similar_automations if a.get("metadata", {}).get("success", False)]
            failed_automations = [a for a in similar_automations if not a.get("metadata", {}).get("success", False)]
            
            if successful_automations:
                # Learn from successful patterns
                avg_confidence = statistics.mean([a.get("metadata", {}).get("ai_confidence", 0) for a in successful_automations])
                if features["ai_confidence"] < avg_confidence:
                    recommendations.append(f"Increase AI confidence to at least {avg_confidence:.2f}")
            
            if failed_automations:
                # Learn from failure patterns
                common_errors = []
                for automation in failed_automations:
                    errors = automation.get("metadata", {}).get("errors", [])
                    common_errors.extend(errors)
                
                if common_errors:
                    recommendations.append("Implement additional error handling for common failure patterns")
            
            # General recommendations
            if features["selectors_available"] < 3:
                recommendations.append("Ensure multiple selector strategies are available")
            
            if features["complexity"] == "complex":
                recommendations.append("Consider breaking complex automation into smaller steps")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations"]

    def _identify_risk_factors(self, features: Dict[str, Any], similar_automations: List[Dict[str, Any]]) -> List[str]:
        """Identify risk factors for automation failure."""
        try:
            risk_factors = []
            
            if features["ai_confidence"] < 0.6:
                risk_factors.append("Low AI confidence")
            
            if features["selectors_available"] < 2:
                risk_factors.append("Limited selector options")
            
            if features["complexity"] == "complex":
                risk_factors.append("High complexity automation")
            
            # Analyze similar failures
            failed_automations = [a for a in similar_automations if not a.get("metadata", {}).get("success", False)]
            if failed_automations:
                risk_factors.append(f"History of failures in similar automations ({len(failed_automations)} cases)")
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Risk factor identification failed: {e}")
            return ["Unknown risks"]

    async def _analyze_plan_efficiency(self, automation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze automation plan efficiency."""
        try:
            steps = automation_plan.get("steps", [])
            
            analysis = {
                "total_steps": len(steps),
                "estimated_time": sum(step.get("estimated_time", 2) for step in steps),
                "complexity_score": self._calculate_complexity_score(steps),
                "redundancy_score": self._calculate_redundancy_score(steps),
                "improvement_potential": 0.0
            }
            
            # Calculate improvement potential
            if analysis["redundancy_score"] > 0.3:
                analysis["improvement_potential"] += 0.2
            
            if analysis["complexity_score"] > 0.7:
                analysis["improvement_potential"] += 0.15
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Plan efficiency analysis failed: {e}")
            return {"improvement_potential": 0.0}

    def _calculate_complexity_score(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for automation steps."""
        try:
            if not steps:
                return 0.0
            
            complexity_factors = []
            
            for step in steps:
                # Action complexity
                action = step.get("action", "")
                if action in ["wait", "navigate"]:
                    complexity_factors.append(0.1)
                elif action in ["click", "type"]:
                    complexity_factors.append(0.3)
                elif action in ["conditional", "loop"]:
                    complexity_factors.append(0.8)
                else:
                    complexity_factors.append(0.5)
                
                # Selector complexity
                selectors = step.get("fallback_selectors", [])
                if len(selectors) > 3:
                    complexity_factors.append(0.2)
            
            return statistics.mean(complexity_factors) if complexity_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Complexity score calculation failed: {e}")
            return 0.5

    def _calculate_redundancy_score(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate redundancy score for automation steps."""
        try:
            if not steps:
                return 0.0
            
            redundant_steps = 0
            
            for i, step in enumerate(steps):
                # Check for similar actions
                for j, other_step in enumerate(steps[i+1:], i+1):
                    if (step.get("action") == other_step.get("action") and 
                        step.get("selector") == other_step.get("selector")):
                        redundant_steps += 1
            
            return redundant_steps / len(steps) if steps else 0.0
            
        except Exception as e:
            self.logger.error(f"Redundancy score calculation failed: {e}")
            return 0.0

    async def _generate_plan_optimizations(self, automation_plan: Dict[str, Any], efficiency_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimizations for automation plan."""
        try:
            optimizations = []
            steps = automation_plan.get("steps", [])
            
            # Remove redundant steps
            if efficiency_analysis.get("redundancy_score", 0) > 0.2:
                optimizations.append({
                    "type": "remove_redundancy",
                    "description": "Remove redundant steps to improve efficiency",
                    "impact": "high"
                })
            
            # Optimize complex steps
            if efficiency_analysis.get("complexity_score", 0) > 0.6:
                optimizations.append({
                    "type": "simplify_complexity",
                    "description": "Break down complex steps into simpler ones",
                    "impact": "medium"
                })
            
            # Add parallel execution where possible
            parallel_opportunities = self._identify_parallel_opportunities(steps)
            if parallel_opportunities:
                optimizations.append({
                    "type": "parallel_execution",
                    "description": f"Execute {len(parallel_opportunities)} steps in parallel",
                    "impact": "high"
                })
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Plan optimization generation failed: {e}")
            return []

    def _identify_parallel_opportunities(self, steps: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify steps that can be executed in parallel."""
        try:
            parallel_groups = []
            current_group = []
            
            for i, step in enumerate(steps):
                action = step.get("action", "")
                
                # Steps that can be parallelized
                if action in ["wait", "type", "click"] and not step.get("depends_on"):
                    current_group.append(i)
                else:
                    if current_group:
                        parallel_groups.append(current_group)
                        current_group = []
            
            if current_group:
                parallel_groups.append(current_group)
            
            return [group for group in parallel_groups if len(group) > 1]
            
        except Exception as e:
            self.logger.error(f"Parallel opportunity identification failed: {e}")
            return []

    async def _apply_optimizations(self, automation_plan: Dict[str, Any], optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply optimizations to automation plan."""
        try:
            optimized_plan = automation_plan.copy()
            steps = optimized_plan.get("steps", [])
            
            for optimization in optimizations:
                if optimization["type"] == "remove_redundancy":
                    steps = self._remove_redundant_steps(steps)
                elif optimization["type"] == "simplify_complexity":
                    steps = self._simplify_complex_steps(steps)
                elif optimization["type"] == "parallel_execution":
                    steps = self._add_parallel_execution(steps)
            
            optimized_plan["steps"] = steps
            optimized_plan["optimizations_applied"] = [opt["type"] for opt in optimizations]
            
            return optimized_plan
            
        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")
            return automation_plan

    def _remove_redundant_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant steps from automation plan."""
        try:
            unique_steps = []
            seen_actions = set()
            
            for step in steps:
                action_key = f"{step.get('action')}_{step.get('selector', '')}"
                if action_key not in seen_actions:
                    unique_steps.append(step)
                    seen_actions.add(action_key)
            
            return unique_steps
            
        except Exception as e:
            self.logger.error(f"Redundant step removal failed: {e}")
            return steps

    def _simplify_complex_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simplify complex steps in automation plan."""
        try:
            simplified_steps = []
            
            for step in steps:
                if step.get("action") == "conditional" and step.get("complexity", 0) > 0.7:
                    # Break down complex conditional into simpler steps
                    simplified_steps.extend(self._break_down_conditional(step))
                else:
                    simplified_steps.append(step)
            
            return simplified_steps
            
        except Exception as e:
            self.logger.error(f"Complex step simplification failed: {e}")
            return steps

    def _break_down_conditional(self, conditional_step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down complex conditional step into simpler steps."""
        try:
            simplified_steps = []
            
            # Add condition check step
            simplified_steps.append({
                "action": "check_condition",
                "description": f"Check condition: {conditional_step.get('description', '')}",
                "selector": conditional_step.get("selector"),
                "expected_result": conditional_step.get("expected_result")
            })
            
            # Add action step
            simplified_steps.append({
                "action": conditional_step.get("action_on_true", "click"),
                "description": f"Execute action if condition is true",
                "selector": conditional_step.get("action_selector"),
                "depends_on": "condition_check"
            })
            
            return simplified_steps
            
        except Exception as e:
            self.logger.error(f"Conditional breakdown failed: {e}")
            return [conditional_step]

    def _add_parallel_execution(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add parallel execution markers to automation plan."""
        try:
            parallel_groups = self._identify_parallel_opportunities(steps)
            
            for group in parallel_groups:
                # Mark steps as parallel
                for step_index in group:
                    if step_index < len(steps):
                        steps[step_index]["parallel_group"] = group[0]
                        steps[step_index]["can_parallelize"] = True
            
            return steps
            
        except Exception as e:
            self.logger.error(f"Parallel execution addition failed: {e}")
            return steps

    async def _validate_optimized_plan(self, optimized_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimized automation plan."""
        try:
            validation_result = {
                "is_valid": True,
                "warnings": [],
                "errors": []
            }
            
            steps = optimized_plan.get("steps", [])
            
            # Check for empty plan
            if not steps:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Optimized plan has no steps")
            
            # Check for circular dependencies
            dependencies = [step.get("depends_on") for step in steps if step.get("depends_on")]
            if len(dependencies) != len(set(dependencies)):
                validation_result["warnings"].append("Potential circular dependencies detected")
            
            # Check for missing selectors
            for step in steps:
                if step.get("action") in ["click", "type"] and not step.get("selector"):
                    validation_result["warnings"].append(f"Step {step.get('step', 'unknown')} missing selector")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Plan validation failed: {e}")
            return {"is_valid": False, "errors": [str(e)]}