#!/usr/bin/env python3
"""
Real AI Engine - Genuine Machine Learning Implementation
======================================================

Actual AI/ML algorithms implemented using only Python standard library.
No external dependencies - pure mathematical implementations.
"""

import math
import random
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics
import re
import hashlib

@dataclass
class TrainingExample:
    """Training example for machine learning"""
    inputs: List[float]
    expected_output: float
    metadata: Dict[str, Any] = None

class NeuralNetwork:
    """Simple neural network implementation using only stdlib"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] 
                                   for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] 
                                    for _ in range(hidden_size)]
        
        # Initialize biases
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
        
        # Learning rate
        self.learning_rate = 0.1
        
        # Training history
        self.training_history = []
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def sigmoid_derivative(self, x: float) -> float:
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(self, inputs: List[float]) -> Tuple[List[float], List[float]]:
        """Forward pass through the network"""
        # Input to hidden layer
        hidden_inputs = []
        for j in range(self.hidden_size):
            weighted_sum = sum(inputs[i] * self.weights_input_hidden[i][j] 
                             for i in range(self.input_size))
            hidden_inputs.append(weighted_sum + self.bias_hidden[j])
        
        hidden_outputs = [self.sigmoid(x) for x in hidden_inputs]
        
        # Hidden to output layer
        output_inputs = []
        for j in range(self.output_size):
            weighted_sum = sum(hidden_outputs[i] * self.weights_hidden_output[i][j] 
                             for i in range(self.hidden_size))
            output_inputs.append(weighted_sum + self.bias_output[j])
        
        final_outputs = [self.sigmoid(x) for x in output_inputs]
        
        return hidden_outputs, final_outputs
    
    def backward_pass(self, inputs: List[float], hidden_outputs: List[float], 
                     final_outputs: List[float], expected_outputs: List[float]):
        """Backward pass - update weights using backpropagation"""
        # Calculate output layer errors
        output_errors = []
        for i in range(self.output_size):
            error = expected_outputs[i] - final_outputs[i]
            output_errors.append(error * self.sigmoid_derivative(final_outputs[i]))
        
        # Calculate hidden layer errors
        hidden_errors = []
        for i in range(self.hidden_size):
            error = sum(output_errors[j] * self.weights_hidden_output[i][j] 
                       for j in range(self.output_size))
            hidden_errors.append(error * self.sigmoid_derivative(hidden_outputs[i]))
        
        # Update weights and biases
        # Hidden to output weights
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] += (self.learning_rate * 
                                                   output_errors[j] * hidden_outputs[i])
        
        # Input to hidden weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += (self.learning_rate * 
                                                  hidden_errors[j] * inputs[i])
        
        # Update biases
        for i in range(self.output_size):
            self.bias_output[i] += self.learning_rate * output_errors[i]
        
        for i in range(self.hidden_size):
            self.bias_hidden[i] += self.learning_rate * hidden_errors[i]
    
    def train(self, training_data: List[TrainingExample], epochs: int = 1000):
        """Train the neural network"""
        for epoch in range(epochs):
            total_error = 0
            
            for example in training_data:
                hidden_outputs, final_outputs = self.forward_pass(example.inputs)
                expected_outputs = [example.expected_output]
                
                # Calculate error
                error = sum((expected_outputs[i] - final_outputs[i]) ** 2 
                          for i in range(len(expected_outputs)))
                total_error += error
                
                # Backpropagation
                self.backward_pass(example.inputs, hidden_outputs, 
                                 final_outputs, expected_outputs)
            
            # Record training progress
            if epoch % 100 == 0:
                avg_error = total_error / len(training_data)
                self.training_history.append({
                    'epoch': epoch,
                    'average_error': avg_error,
                    'timestamp': time.time()
                })
    
    def predict(self, inputs: List[float]) -> float:
        """Make a prediction"""
        _, outputs = self.forward_pass(inputs)
        return outputs[0]

class PatternRecognizer:
    """Pattern recognition using statistical analysis"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_counts = Counter()
        self.sequence_patterns = defaultdict(list)
    
    def learn_pattern(self, data: Any, label: str):
        """Learn a new pattern"""
        # Convert data to feature vector
        features = self._extract_features(data)
        
        if label not in self.patterns:
            self.patterns[label] = []
        
        self.patterns[label].append(features)
        self.pattern_counts[label] += 1
    
    def _extract_features(self, data: Any) -> List[float]:
        """Extract numerical features from data"""
        if isinstance(data, str):
            return self._text_features(data)
        elif isinstance(data, (list, tuple)):
            return self._sequence_features(data)
        elif isinstance(data, dict):
            return self._dict_features(data)
        else:
            return [float(hash(str(data)) % 1000) / 1000.0]
    
    def _text_features(self, text: str) -> List[float]:
        """Extract features from text"""
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(text.split()) / 100.0)  # Word count
        
        # Character distribution
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        # Common character frequencies
        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        for char in common_chars:
            features.append(char_counts[char] / max(total_chars, 1))
        
        # Pattern features
        features.append(text.count(' ') / max(len(text), 1))  # Space ratio
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # Uppercase ratio
        features.append(sum(1 for c in text if c.isdigit()) / max(len(text), 1))  # Digit ratio
        
        return features[:50]  # Limit feature vector size
    
    def _sequence_features(self, sequence: List) -> List[float]:
        """Extract features from sequence"""
        if not sequence:
            return [0.0] * 10
        
        features = []
        
        # Basic statistics
        numeric_values = [x for x in sequence if isinstance(x, (int, float))]
        if numeric_values:
            features.extend([
                len(numeric_values),
                sum(numeric_values),
                statistics.mean(numeric_values),
                statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                min(numeric_values),
                max(numeric_values)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Sequence properties
        features.append(len(sequence))
        features.append(len(set(str(x) for x in sequence)))  # Unique elements
        
        return features[:20]
    
    def _dict_features(self, data: dict) -> List[float]:
        """Extract features from dictionary"""
        features = []
        
        # Structure features
        features.append(len(data))
        features.append(len([v for v in data.values() if isinstance(v, str)]))
        features.append(len([v for v in data.values() if isinstance(v, (int, float))]))
        features.append(len([v for v in data.values() if isinstance(v, (list, dict))]))
        
        # Value features
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if numeric_values:
            features.append(sum(numeric_values))
            features.append(statistics.mean(numeric_values))
        else:
            features.extend([0.0, 0.0])
        
        return features[:15]
    
    def recognize_pattern(self, data: Any) -> Dict[str, float]:
        """Recognize pattern in data"""
        if not self.patterns:
            return {'unknown': 1.0}
        
        features = self._extract_features(data)
        similarities = {}
        
        for label, pattern_examples in self.patterns.items():
            # Calculate average similarity to all examples of this pattern
            total_similarity = 0
            for example in pattern_examples:
                similarity = self._calculate_similarity(features, example)
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(pattern_examples)
            similarities[label] = avg_similarity
        
        # Normalize similarities to probabilities
        total_sim = sum(similarities.values())
        if total_sim > 0:
            return {label: sim / total_sim for label, sim in similarities.items()}
        else:
            return {'unknown': 1.0}
    
    def _calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        # Ensure same length
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(f1, f2))
        magnitude1 = math.sqrt(sum(a * a for a in f1))
        magnitude2 = math.sqrt(sum(a * a for a in f2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

class AdaptiveLearningSystem:
    """Adaptive learning system that improves over time"""
    
    def __init__(self):
        self.experience_buffer = []
        self.performance_history = []
        self.adaptation_rules = {}
        self.learning_rate = 0.1
        
    def record_experience(self, context: Dict[str, Any], action: str, 
                         outcome: float, feedback: Dict[str, Any] = None):
        """Record an experience for learning"""
        experience = {
            'context': context,
            'action': action,
            'outcome': outcome,
            'feedback': feedback or {},
            'timestamp': time.time(),
            'id': hashlib.md5(f"{context}{action}{time.time()}".encode()).hexdigest()[:8]
        }
        
        self.experience_buffer.append(experience)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-1000:]
        
        # Update performance tracking
        self.performance_history.append({
            'timestamp': time.time(),
            'outcome': outcome,
            'action': action
        })
    
    def adapt_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy based on past experiences"""
        if not self.experience_buffer:
            return {'strategy': 'explore', 'confidence': 0.0}
        
        # Find similar contexts
        similar_experiences = self._find_similar_experiences(context)
        
        if not similar_experiences:
            return {'strategy': 'explore', 'confidence': 0.0}
        
        # Analyze what worked best in similar contexts
        action_outcomes = defaultdict(list)
        for exp in similar_experiences:
            action_outcomes[exp['action']].append(exp['outcome'])
        
        # Calculate average outcomes for each action
        action_scores = {}
        for action, outcomes in action_outcomes.items():
            action_scores[action] = {
                'average_outcome': statistics.mean(outcomes),
                'consistency': 1.0 - statistics.stdev(outcomes) if len(outcomes) > 1 else 1.0,
                'frequency': len(outcomes)
            }
        
        # Choose best action
        best_action = max(action_scores.keys(), 
                         key=lambda a: action_scores[a]['average_outcome'])
        
        confidence = min(1.0, action_scores[best_action]['frequency'] / 10.0)
        
        return {
            'strategy': 'exploit',
            'recommended_action': best_action,
            'confidence': confidence,
            'expected_outcome': action_scores[best_action]['average_outcome'],
            'alternatives': action_scores
        }
    
    def _find_similar_experiences(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find experiences with similar contexts"""
        similar = []
        
        for exp in self.experience_buffer:
            similarity = self._calculate_context_similarity(context, exp['context'])
            if similarity > 0.5:  # Threshold for similarity
                similar.append(exp)
        
        return sorted(similar, key=lambda x: self._calculate_context_similarity(context, x['context']), 
                     reverse=True)[:20]  # Top 20 most similar
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                sim = 1.0 if val1 == val2 else (len(set(val1) & set(val2)) / 
                                               len(set(val1) | set(val2)) if val1 or val2 else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2), 1)
                sim = 1.0 - abs(val1 - val2) / max_val
            else:
                # Generic similarity
                sim = 1.0 if str(val1) == str(val2) else 0.0
            
            similarities.append(sim)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        if not self.performance_history:
            return {'experiences': 0, 'average_outcome': 0.0}
        
        recent_outcomes = [h['outcome'] for h in self.performance_history[-100:]]
        
        return {
            'total_experiences': len(self.experience_buffer),
            'recent_average_outcome': statistics.mean(recent_outcomes),
            'improvement_trend': self._calculate_improvement_trend(),
            'adaptation_rules_learned': len(self.adaptation_rules),
            'learning_active': True
        }
    
    def _calculate_improvement_trend(self) -> float:
        """Calculate if performance is improving over time"""
        if len(self.performance_history) < 20:
            return 0.0
        
        # Compare first half vs second half of recent history
        recent = self.performance_history[-40:]
        first_half = [h['outcome'] for h in recent[:20]]
        second_half = [h['outcome'] for h in recent[20:]]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        return second_avg - first_avg

class RealAIEngine:
    """Complete real AI engine combining all components"""
    
    def __init__(self):
        self.neural_network = None
        self.pattern_recognizer = PatternRecognizer()
        self.adaptive_learner = AdaptiveLearningSystem()
        self.training_data = []
        self.is_trained = False
        
    def initialize_neural_network(self, input_size: int = 10, hidden_size: int = 5):
        """Initialize neural network with specified architecture"""
        self.neural_network = NeuralNetwork(input_size, hidden_size, 1)
    
    def learn_from_data(self, data: List[Dict[str, Any]]):
        """Learn from training data"""
        # Extract patterns
        for item in data:
            if 'input' in item and 'label' in item:
                self.pattern_recognizer.learn_pattern(item['input'], item['label'])
        
        # Prepare neural network training data
        training_examples = []
        for item in data:
            if 'features' in item and 'target' in item:
                features = item['features'] if isinstance(item['features'], list) else [item['features']]
                target = float(item['target'])
                training_examples.append(TrainingExample(features, target))
        
        # Train neural network if we have data
        if training_examples:
            if not self.neural_network:
                input_size = len(training_examples[0].inputs)
                self.initialize_neural_network(input_size)
            
            self.neural_network.train(training_examples, epochs=500)
            self.is_trained = True
        
        self.training_data.extend(data)
    
    def make_intelligent_decision(self, context: Dict[str, Any], 
                                options: List[str]) -> Dict[str, Any]:
        """Make an intelligent decision using all AI components"""
        start_time = time.time()
        
        # Pattern recognition
        pattern_analysis = self.pattern_recognizer.recognize_pattern(context)
        
        # Adaptive learning recommendation
        adaptation = self.adaptive_learner.adapt_strategy(context)
        
        # Neural network prediction if trained
        nn_prediction = None
        if self.neural_network and self.is_trained:
            # Convert context to feature vector
            features = self.pattern_recognizer._extract_features(context)
            if len(features) >= self.neural_network.input_size:
                features = features[:self.neural_network.input_size]
            else:
                features.extend([0.0] * (self.neural_network.input_size - len(features)))
            
            nn_prediction = self.neural_network.predict(features)
        
        # Combine all AI insights to make decision
        decision_scores = {}
        for option in options:
            score = 0.0
            
            # Pattern-based scoring
            if option in pattern_analysis:
                score += pattern_analysis[option] * 0.4
            
            # Adaptive learning scoring
            if adaptation['strategy'] == 'exploit' and option == adaptation.get('recommended_action'):
                score += adaptation['confidence'] * 0.4
            
            # Neural network scoring
            if nn_prediction is not None:
                # Simple mapping of NN output to option preference
                option_index = options.index(option)
                option_score = abs(nn_prediction - (option_index / len(options)))
                score += (1.0 - option_score) * 0.2
            
            # Add some exploration randomness
            score += random.uniform(0, 0.1)
            
            decision_scores[option] = score
        
        # Select best option
        best_option = max(decision_scores.keys(), key=lambda x: decision_scores[x])
        confidence = decision_scores[best_option]
        
        # Record this decision for learning
        self.adaptive_learner.record_experience(
            context=context,
            action=best_option,
            outcome=confidence,  # We'll update this later with actual outcome
            feedback={'decision_scores': decision_scores}
        )
        
        processing_time = time.time() - start_time
        
        return {
            'decision': best_option,
            'confidence': min(1.0, confidence),
            'reasoning': {
                'pattern_analysis': pattern_analysis,
                'adaptive_recommendation': adaptation,
                'neural_network_prediction': nn_prediction,
                'decision_scores': decision_scores
            },
            'processing_time': processing_time,
            'ai_components_used': ['pattern_recognition', 'adaptive_learning'] + 
                                (['neural_network'] if nn_prediction else []),
            'learning_active': True
        }
    
    def update_decision_outcome(self, decision_id: str, actual_outcome: float):
        """Update the outcome of a previous decision for learning"""
        # Find the most recent experience and update it
        if self.adaptive_learner.experience_buffer:
            self.adaptive_learner.experience_buffer[-1]['outcome'] = actual_outcome
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI engine status"""
        return {
            'neural_network_trained': self.is_trained,
            'neural_network_architecture': {
                'input_size': self.neural_network.input_size if self.neural_network else 0,
                'hidden_size': self.neural_network.hidden_size if self.neural_network else 0,
                'output_size': self.neural_network.output_size if self.neural_network else 0
            } if self.neural_network else None,
            'patterns_learned': len(self.pattern_recognizer.patterns),
            'pattern_types': list(self.pattern_recognizer.patterns.keys()),
            'learning_stats': self.adaptive_learner.get_learning_stats(),
            'training_examples': len(self.training_data),
            'real_ai_active': True,
            'zero_external_dependencies': True
        }

# Global AI engine instance
_real_ai_engine = None

def get_real_ai_engine() -> RealAIEngine:
    """Get global real AI engine instance"""
    global _real_ai_engine
    if _real_ai_engine is None:
        _real_ai_engine = RealAIEngine()
        
        # Initialize with some basic training data
        basic_training = [
            {'input': 'optimize performance', 'label': 'optimization'},
            {'input': 'handle error', 'label': 'error_handling'},
            {'input': 'process data', 'label': 'data_processing'},
            {'input': 'coordinate tasks', 'label': 'coordination'},
            {'features': [1.0, 0.5, 0.2], 'target': 0.8},
            {'features': [0.2, 0.8, 0.1], 'target': 0.3},
            {'features': [0.7, 0.3, 0.9], 'target': 0.9}
        ]
        _real_ai_engine.learn_from_data(basic_training)
    
    return _real_ai_engine

if __name__ == "__main__":
    # Demo of real AI engine
    print("🧠 REAL AI ENGINE DEMO")
    print("=" * 50)
    
    ai_engine = get_real_ai_engine()
    
    # Test intelligent decision making
    context = {
        'task_type': 'automation',
        'complexity': 'high',
        'priority': 'urgent',
        'resources': ['cpu', 'memory', 'network']
    }
    
    options = ['optimize_first', 'execute_immediately', 'schedule_later']
    
    decision = ai_engine.make_intelligent_decision(context, options)
    
    print(f"Context: {context}")
    print(f"Options: {options}")
    print(f"AI Decision: {decision['decision']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"AI Components Used: {decision['ai_components_used']}")
    print(f"Processing Time: {decision['processing_time']:.3f}s")
    
    # Show AI status
    print(f"\n📊 AI Engine Status:")
    status = ai_engine.get_ai_status()
    print(f"Neural Network Trained: {status['neural_network_trained']}")
    print(f"Patterns Learned: {status['patterns_learned']}")
    print(f"Learning Active: {status['learning_stats']['learning_active']}")
    print(f"Real AI Active: {status['real_ai_active']}")
    
    print("\n✅ Real AI Engine working with zero external dependencies!")