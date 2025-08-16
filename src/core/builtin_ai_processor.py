#!/usr/bin/env python3
"""
Built-in AI Processor - 100% Zero Dependencies
===============================================

Advanced AI processing using only Python standard library.
Provides text analysis, decision making, pattern recognition, and entity extraction
without any external dependencies.

✅ FEATURES:
- Text Analysis: Sentiment analysis, keyword extraction
- Decision Making: Multi-option decision with confidence scoring  
- Pattern Recognition: Learning from examples
- Entity Extraction: Email, phone, URL detection
"""

import re
import json
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
import string
import random
from datetime import datetime
import hashlib

class BuiltinAIProcessor:
    """Advanced AI processor with zero external dependencies"""
    
    def __init__(self):
        self.patterns_learned = {}
        self.decision_history = []
        self.sentiment_keywords = {
            'positive': [
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                'awesome', 'perfect', 'love', 'like', 'happy', 'pleased', 'satisfied',
                'success', 'successful', 'win', 'winner', 'best', 'better', 'improve',
                'beautiful', 'brilliant', 'outstanding', 'superb', 'magnificent'
            ],
            'negative': [
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
                'angry', 'frustrated', 'disappointed', 'fail', 'failure', 'worst', 'worse',
                'problem', 'issue', 'error', 'broken', 'wrong', 'difficult', 'hard',
                'impossible', 'useless', 'stupid', 'ridiculous', 'annoying'
            ],
            'neutral': [
                'okay', 'fine', 'normal', 'average', 'standard', 'typical', 'usual',
                'regular', 'common', 'ordinary', 'moderate', 'medium', 'fair'
            ]
        }
        
        # Enhanced entity extraction patterns
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b',
            'url': r'https?://[^\s<>"{}|\\^`[\]]+|www\.[^\s<>"{}|\\^`[\]]+',
            'ip': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP|INR|dollars?|euros?)',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'zipcode': r'\b\d{5}(?:-\d{4})?\b',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
            'percentage': r'\b\d+(?:\.\d+)?%'
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text analysis including sentiment, keywords, and entities
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': {'score': 0.0, 'label': 'neutral', 'confidence': 0.0},
                'keywords': [],
                'entities': {},
                'statistics': {},
                'language_features': {}
            }
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        words = self._tokenize(cleaned_text)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(words)
        
        # Keyword extraction
        keywords = self._extract_keywords(words, text)
        
        # Entity extraction
        entities = self._extract_entities(text)
        
        # Text statistics
        statistics_data = self._calculate_statistics(text, words)
        
        # Language features
        language_features = self._analyze_language_features(text, words)
        
        return {
            'sentiment': sentiment,
            'keywords': keywords,
            'entities': entities,
            'statistics': statistics_data,
            'language_features': language_features,
            'processing_timestamp': datetime.now().isoformat()
        }

    def make_decision(self, options: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        AI-powered decision making with confidence scoring
        
        Args:
            options: List of available options
            context: Additional context for decision making
            
        Returns:
            Decision result with confidence and reasoning
        """
        if not options:
            return {
                'decision': None,
                'confidence': 0.0,
                'reasoning': 'No options provided',
                'scores': {}
            }
        
        context = context or {}
        
        # Score each option based on multiple factors
        scores = {}
        reasoning_factors = []
        
        for option in options:
            score = self._score_option(option, context)
            scores[option] = score
            
        # Find best option
        best_option = max(scores.keys(), key=lambda x: scores[x])
        best_score = scores[best_option]
        
        # Calculate confidence based on score distribution
        confidence = self._calculate_decision_confidence(scores)
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(best_option, scores, context)
        
        # Store decision history for learning
        decision_record = {
            'options': options,
            'decision': best_option,
            'confidence': confidence,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.decision_history.append(decision_record)
        
        return {
            'decision': best_option,
            'confidence': confidence,
            'reasoning': reasoning,
            'scores': scores,
            'all_options': options
        }

    def recognize_patterns(self, examples: List[Dict[str, Any]], 
                          new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pattern recognition and classification based on examples
        
        Args:
            examples: List of example data with labels
            new_data: New data to classify
            
        Returns:
            Classification result with confidence
        """
        if not examples:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'similar_examples': [],
                'pattern_features': {}
            }
        
        # Extract features from examples
        pattern_features = self._extract_pattern_features(examples)
        
        # Extract features from new data
        new_features = self._extract_features(new_data)
        
        # Find most similar examples
        similarities = []
        for example in examples:
            example_features = self._extract_features(example)
            similarity = self._calculate_similarity(new_features, example_features)
            similarities.append({
                'example': example,
                'similarity': similarity,
                'label': example.get('label', 'unknown')
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Classify based on most similar examples
        classification = self._classify_by_similarity(similarities)
        
        return {
            'classification': classification['label'],
            'confidence': classification['confidence'],
            'similar_examples': similarities[:3],
            'pattern_features': pattern_features,
            'feature_match_score': classification.get('feature_score', 0.0)
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract various entities from text (emails, phones, URLs, etc.)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and found entities
        """
        return self._extract_entities(text)

    def learn_from_feedback(self, decision_id: str, outcome: str, 
                           feedback_score: float) -> bool:
        """
        Learn from decision outcomes to improve future decisions
        
        Args:
            decision_id: ID of the decision
            outcome: Outcome description
            feedback_score: Score from 0.0 to 1.0
            
        Returns:
            Success status
        """
        try:
            # Find the decision in history
            for decision in self.decision_history:
                decision_hash = hashlib.md5(
                    json.dumps(decision, sort_keys=True).encode()
                ).hexdigest()[:8]
                
                if decision_hash == decision_id:
                    decision['outcome'] = outcome
                    decision['feedback_score'] = feedback_score
                    decision['learned'] = True
                    
                    # Update patterns learned
                    self._update_learned_patterns(decision)
                    return True
            
            return False
        except Exception:
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics"""
        total_decisions = len(self.decision_history)
        successful_decisions = len([d for d in self.decision_history 
                                  if d.get('feedback_score', 0) > 0.7])
        
        avg_confidence = 0.0
        if self.decision_history:
            avg_confidence = statistics.mean([d['confidence'] for d in self.decision_history])
        
        return {
            'total_decisions': total_decisions,
            'successful_decisions': successful_decisions,
            'success_rate': successful_decisions / max(total_decisions, 1),
            'average_confidence': avg_confidence,
            'patterns_learned': len(self.patterns_learned),
            'entity_types_supported': len(self.entity_patterns),
            'sentiment_keywords': sum(len(words) for words in self.sentiment_keywords.values())
        }

    # Private helper methods
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase for processing
        return text.lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation)
        cleaned = text.translate(translator)
        return cleaned.split()
    
    def _analyze_sentiment(self, words: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        positive_score = 0
        negative_score = 0
        neutral_score = 0
        
        for word in words:
            if word in self.sentiment_keywords['positive']:
                positive_score += 1
            elif word in self.sentiment_keywords['negative']:
                negative_score += 1
            elif word in self.sentiment_keywords['neutral']:
                neutral_score += 1
        
        total_sentiment_words = positive_score + negative_score + neutral_score
        
        if total_sentiment_words == 0:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_score - negative_score) / len(words)
        
        # Determine label
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence
        confidence = total_sentiment_words / len(words)
        confidence = min(confidence, 1.0)
        
        return {
            'score': sentiment_score,
            'label': label,
            'confidence': confidence,
            'word_counts': {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score
            }
        }
    
    def _extract_keywords(self, words: List[str], original_text: str) -> List[Dict[str, Any]]:
        """Extract important keywords from text"""
        # Count word frequencies
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = []
        for word, freq in word_freq.most_common(10):
            if word not in stop_words and len(word) > 2:
                # Calculate importance score
                importance = freq / len(words)
                keywords.append({
                    'word': word,
                    'frequency': freq,
                    'importance': importance,
                    'positions': [i for i, w in enumerate(words) if w == word]
                })
        
        return keywords
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _calculate_statistics(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Calculate text statistics"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': statistics.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(words)),
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0
        }
    
    def _analyze_language_features(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze language features and complexity"""
        # Count different types of words
        capitalized_words = len([w for w in text.split() if w[0].isupper()])
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        # Readability metrics (simplified)
        avg_sentence_length = len(words) / max(text.count('.'), 1)
        complex_words = len([w for w in words if len(w) > 6])
        
        return {
            'capitalized_words': capitalized_words,
            'question_count': question_marks,
            'exclamation_count': exclamation_marks,
            'avg_sentence_length': avg_sentence_length,
            'complex_words': complex_words,
            'complexity_ratio': complex_words / len(words) if words else 0,
            'punctuation_density': (question_marks + exclamation_marks) / len(text) if text else 0
        }
    
    def _score_option(self, option: str, context: Dict[str, Any]) -> float:
        """Score an option based on context and learned patterns"""
        base_score = 0.5  # Neutral starting point
        
        # Score based on context
        if context:
            # Check for positive indicators in context
            score_value = context.get('score', 0)
            if isinstance(score_value, (int, float)):
                base_score += score_value * 0.3
            
            # Check for priority indicators
            priority = context.get('priority', 'medium')
            if priority == 'high':
                base_score += 0.2
            elif priority == 'low':
                base_score -= 0.1
            
            # Check for historical success
            if option in context.get('successful_options', []):
                base_score += 0.3
            
            # Check for failure history
            if option in context.get('failed_options', []):
                base_score -= 0.2
        
        # Score based on learned patterns
        if option in self.patterns_learned:
            pattern_score = self.patterns_learned[option].get('success_rate', 0.5)
            base_score = (base_score + pattern_score) / 2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def _calculate_decision_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence based on score distribution"""
        if not scores:
            return 0.0
        
        score_values = list(scores.values())
        if len(score_values) == 1:
            return score_values[0]
        
        # Higher confidence if there's a clear winner
        max_score = max(score_values)
        second_max = sorted(score_values, reverse=True)[1]
        
        # Confidence based on gap between best and second best
        confidence = max_score - ((max_score - second_max) * 0.5)
        return min(1.0, max(0.0, confidence))
    
    def _generate_decision_reasoning(self, decision: str, scores: Dict[str, float], 
                                   context: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for decision"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Selected '{decision}' with score {scores[decision]:.2f}")
        
        # Compare with other options
        other_scores = {k: v for k, v in scores.items() if k != decision}
        if other_scores:
            avg_other = statistics.mean(other_scores.values())
            if scores[decision] > avg_other:
                reasoning_parts.append(f"outperformed other options by {scores[decision] - avg_other:.2f}")
        
        # Context-based reasoning
        if context:
            if context.get('priority') == 'high':
                reasoning_parts.append("high priority context")
            if decision in context.get('successful_options', []):
                reasoning_parts.append("historically successful option")
        
        return "; ".join(reasoning_parts)
    
    def _extract_pattern_features(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common features from examples"""
        features = defaultdict(list)
        
        for example in examples:
            for key, value in example.items():
                if key != 'label':
                    features[key].append(value)
        
        # Calculate feature statistics
        pattern_features = {}
        for feature, values in features.items():
            if all(isinstance(v, (int, float)) for v in values):
                pattern_features[feature] = {
                    'type': 'numeric',
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
            else:
                pattern_features[feature] = {
                    'type': 'categorical',
                    'values': list(set(str(v) for v in values)),
                    'most_common': Counter(str(v) for v in values).most_common(1)[0][0]
                }
        
        return pattern_features
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from data"""
        features = {}
        for key, value in data.items():
            if key != 'label':
                if isinstance(value, (int, float)):
                    features[key] = float(value)
                else:
                    features[key] = str(value)
        return features
    
    def _calculate_similarity(self, features1: Dict[str, Any], 
                            features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    sim = 1.0
                else:
                    sim = 1.0 - abs(val1 - val2) / max_val
            else:
                # String similarity (simple exact match)
                sim = 1.0 if str(val1) == str(val2) else 0.0
            
            similarities.append(sim)
        
        return statistics.mean(similarities)
    
    def _classify_by_similarity(self, similarities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify based on similarity scores"""
        if not similarities:
            return {'label': 'unknown', 'confidence': 0.0}
        
        # Group by label and calculate average similarity
        label_scores = defaultdict(list)
        for sim_data in similarities:
            label_scores[sim_data['label']].append(sim_data['similarity'])
        
        # Calculate average score for each label
        label_averages = {}
        for label, scores in label_scores.items():
            label_averages[label] = statistics.mean(scores)
        
        # Find best label
        best_label = max(label_averages.keys(), key=lambda x: label_averages[x])
        confidence = label_averages[best_label]
        
        return {
            'label': best_label,
            'confidence': confidence,
            'feature_score': confidence
        }
    
    def _update_learned_patterns(self, decision: Dict[str, Any]) -> None:
        """Update learned patterns based on feedback"""
        decision_option = decision['decision']
        feedback_score = decision.get('feedback_score', 0.5)
        
        if decision_option not in self.patterns_learned:
            self.patterns_learned[decision_option] = {
                'total_uses': 0,
                'total_score': 0.0,
                'success_rate': 0.5
            }
        
        pattern = self.patterns_learned[decision_option]
        pattern['total_uses'] += 1
        pattern['total_score'] += feedback_score
        pattern['success_rate'] = pattern['total_score'] / pattern['total_uses']