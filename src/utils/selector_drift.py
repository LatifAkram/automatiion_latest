"""
Selector Drift Detection
=======================

ML-based selector drift detection and self-healing capabilities for
maintaining robust web automation in the face of UI changes.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import re
import io
from pathlib import Path

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    import cv2
    from PIL import Image
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class SelectorDriftDetector:
    """ML-based selector drift detection and self-healing."""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ML components
        self.vectorizer = None
        self.drift_patterns = {}
        self.selector_history = {}
        
        # Drift detection thresholds
        self.similarity_threshold = 0.8
        self.confidence_threshold = 0.7
        
        # Alternative selector strategies
        self.selector_strategies = [
            "css_selector",
            "xpath",
            "text_content",
            "attributes",
            "position",
            "visual_similarity"
        ]
        
    async def initialize(self):
        """Initialize selector drift detector."""
        try:
            if ML_AVAILABLE:
                # Initialize TF-IDF vectorizer for text similarity
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
            self.logger.info("Selector drift detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize selector drift detector: {e}", exc_info=True)
            raise
            
    async def detect_drift(self, page, original_selector: str, element_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect if a selector has drifted and find alternatives.
        
        Args:
            page: Playwright page object
            original_selector: Original CSS selector
            element_context: Context about the element (text, attributes, etc.)
            
        Returns:
            Drift detection results
        """
        try:
            self.logger.info(f"Detecting drift for selector: {original_selector}")
            
            # Check if original selector still works
            try:
                element = page.locator(original_selector)
                await element.wait_for(timeout=2000)
                is_present = await element.count() > 0
            except Exception:
                is_present = False
                
            if is_present:
                return {
                    "drift_detected": False,
                    "original_selector": original_selector,
                    "confidence": 1.0,
                    "message": "Original selector still works"
                }
                
            # Element not found - drift detected
            self.logger.warning(f"Drift detected for selector: {original_selector}")
            
            # Find alternative selectors
            alternatives = await self._find_alternative_selectors(page, original_selector, element_context)
            
            # Rank alternatives by confidence
            ranked_alternatives = await self._rank_alternatives(page, alternatives, element_context)
            
            # Store drift pattern for learning
            await self._store_drift_pattern(original_selector, alternatives, element_context)
            
            return {
                "drift_detected": True,
                "original_selector": original_selector,
                "alternatives": ranked_alternatives,
                "best_alternative": ranked_alternatives[0] if ranked_alternatives else None,
                "confidence": ranked_alternatives[0]["confidence"] if ranked_alternatives else 0.0,
                "message": f"Found {len(ranked_alternatives)} alternative selectors"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect drift: {e}", exc_info=True)
            return {
                "drift_detected": False,
                "error": str(e),
                "confidence": 0.0
            }
            
    async def _find_alternative_selectors(self, page, original_selector: str, 
                                        element_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find alternative selectors using multiple strategies."""
        alternatives = []
        
        try:
            # Strategy 1: Text-based selectors
            if element_context and element_context.get("text"):
                text_alternatives = await self._find_text_based_selectors(page, element_context["text"])
                alternatives.extend(text_alternatives)
                
            # Strategy 2: Attribute-based selectors
            if element_context and element_context.get("attributes"):
                attr_alternatives = await self._find_attribute_based_selectors(page, element_context["attributes"])
                alternatives.extend(attr_alternatives)
                
            # Strategy 3: Position-based selectors
            if element_context and element_context.get("position"):
                pos_alternatives = await self._find_position_based_selectors(page, element_context["position"])
                alternatives.extend(pos_alternatives)
                
            # Strategy 4: Visual similarity
            if element_context and element_context.get("visual_hash"):
                visual_alternatives = await self._find_visually_similar_elements(page, element_context["visual_hash"])
                alternatives.extend(visual_alternatives)
                
            # Strategy 5: Semantic similarity
            semantic_alternatives = await self._find_semantically_similar_selectors(page, original_selector)
            alternatives.extend(semantic_alternatives)
            
            # Strategy 6: XPath alternatives
            xpath_alternatives = await self._find_xpath_alternatives(page, original_selector, element_context)
            alternatives.extend(xpath_alternatives)
            
            return alternatives
            
        except Exception as e:
            self.logger.error(f"Failed to find alternative selectors: {e}", exc_info=True)
            return []
            
    async def _find_text_based_selectors(self, page, text: str) -> List[Dict[str, Any]]:
        """Find selectors based on text content."""
        alternatives = []
        
        try:
            # Find elements containing the text
            elements = page.locator(f"text={text}")
            count = await elements.count()
            
            for i in range(min(count, 5)):  # Limit to 5 alternatives
                element = elements.nth(i)
                
                # Generate CSS selector
                css_selector = await self._generate_css_selector(page, element)
                if css_selector:
                    alternatives.append({
                        "selector": css_selector,
                        "type": "text_based",
                        "strategy": "text_content",
                        "text": text,
                        "index": i
                    })
                    
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"Text-based selector search failed: {e}")
            return []
            
    async def _find_attribute_based_selectors(self, page, attributes: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find selectors based on element attributes."""
        alternatives = []
        
        try:
            for attr_name, attr_value in attributes.items():
                if attr_value:
                    # Create attribute selector
                    selector = f"[{attr_name}='{attr_value}']"
                    
                    try:
                        elements = page.locator(selector)
                        count = await elements.count()
                        
                        if count > 0:
                            alternatives.append({
                                "selector": selector,
                                "type": "attribute_based",
                                "strategy": "attributes",
                                "attribute": attr_name,
                                "value": attr_value
                            })
                    except Exception:
                        continue
                        
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"Attribute-based selector search failed: {e}")
            return []
            
    async def _find_position_based_selectors(self, page, position: Dict[str, int]) -> List[Dict[str, Any]]:
        """Find selectors based on element position."""
        alternatives = []
        
        try:
            # Find elements at similar position
            x, y = position.get("x", 0), position.get("y", 0)
            
            # Get all elements and check their positions
            all_elements = page.locator("*")
            count = await all_elements.count()
            
            for i in range(min(count, 100)):  # Limit search
                try:
                    element = all_elements.nth(i)
                    bounding_box = await element.bounding_box()
                    
                    if bounding_box:
                        # Check if position is close
                        distance = ((bounding_box["x"] - x) ** 2 + (bounding_box["y"] - y) ** 2) ** 0.5
                        
                        if distance < 50:  # Within 50 pixels
                            css_selector = await self._generate_css_selector(page, element)
                            if css_selector:
                                alternatives.append({
                                    "selector": css_selector,
                                    "type": "position_based",
                                    "strategy": "position",
                                    "distance": distance,
                                    "position": bounding_box
                                })
                except Exception:
                    continue
                    
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"Position-based selector search failed: {e}")
            return []
            
    async def _find_visually_similar_elements(self, page, visual_hash: str) -> List[Dict[str, Any]]:
        """Find elements with similar visual appearance."""
        alternatives = []
        
        try:
            if not ML_AVAILABLE:
                return alternatives
                
            # Get all images and elements with background images
            image_elements = page.locator("img, [style*='background']")
            count = await image_elements.count()
            
            for i in range(min(count, 20)):  # Limit search
                try:
                    element = image_elements.nth(i)
                    
                    # Calculate visual hash
                    element_hash = await self._calculate_visual_hash(page, element)
                    
                    if element_hash:
                        # Calculate similarity
                        similarity = self._calculate_hash_similarity(visual_hash, element_hash)
                        
                        if similarity > 0.8:  # High similarity threshold
                            css_selector = await self._generate_css_selector(page, element)
                            if css_selector:
                                alternatives.append({
                                    "selector": css_selector,
                                    "type": "visual_similarity",
                                    "strategy": "visual_similarity",
                                    "similarity": similarity,
                                    "visual_hash": element_hash
                                })
                except Exception:
                    continue
                    
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"Visual similarity search failed: {e}")
            return []
            
    async def _find_semantically_similar_selectors(self, page, original_selector: str) -> List[Dict[str, Any]]:
        """Find selectors with similar semantic meaning."""
        alternatives = []
        
        try:
            if not ML_AVAILABLE or not self.vectorizer:
                return alternatives
                
            # Extract semantic features from original selector
            original_features = self._extract_selector_features(original_selector)
            
            # Get all elements and compare
            all_elements = page.locator("*")
            count = await all_elements.count()
            
            for i in range(min(count, 50)):  # Limit search
                try:
                    element = all_elements.nth(i)
                    css_selector = await self._generate_css_selector(page, element)
                    
                    if css_selector:
                        # Extract features from this selector
                        element_features = self._extract_selector_features(css_selector)
                        
                        # Calculate semantic similarity
                        similarity = self._calculate_semantic_similarity(original_features, element_features)
                        
                        if similarity > self.similarity_threshold:
                            alternatives.append({
                                "selector": css_selector,
                                "type": "semantic_similarity",
                                "strategy": "semantic_similarity",
                                "similarity": similarity
                            })
                except Exception:
                    continue
                    
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity search failed: {e}")
            return []
            
    async def _find_xpath_alternatives(self, page, original_selector: str, 
                                     element_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find XPath alternatives to CSS selectors."""
        alternatives = []
        
        try:
            # Convert CSS selector to XPath
            xpath_selector = self._css_to_xpath(original_selector)
            
            if xpath_selector:
                try:
                    elements = page.locator(f"xpath={xpath_selector}")
                    count = await elements.count()
                    
                    if count > 0:
                        alternatives.append({
                            "selector": f"xpath={xpath_selector}",
                            "type": "xpath",
                            "strategy": "xpath",
                            "original_css": original_selector
                        })
                except Exception:
                    pass
                    
            # Generate XPath based on context
            if element_context:
                context_xpath = self._generate_context_xpath(element_context)
                if context_xpath:
                    try:
                        elements = page.locator(f"xpath={context_xpath}")
                        count = await elements.count()
                        
                        if count > 0:
                            alternatives.append({
                                "selector": f"xpath={context_xpath}",
                                "type": "xpath",
                                "strategy": "context_xpath",
                                "context": element_context
                            })
                    except Exception:
                        pass
                        
            return alternatives
            
        except Exception as e:
            self.logger.warning(f"XPath alternative search failed: {e}")
            return []
            
    async def _generate_css_selector(self, page, element) -> Optional[str]:
        """Generate a unique CSS selector for an element."""
        try:
            # Use Playwright's built-in selector generation
            selector = await element.evaluate("""
                (element) => {
                    if (element.id) {
                        return '#' + element.id;
                    }
                    
                    if (element.className) {
                        const classes = element.className.split(' ').filter(c => c);
                        if (classes.length > 0) {
                            return '.' + classes.join('.');
                        }
                    }
                    
                    let path = [];
                    while (element && element.nodeType === Node.ELEMENT_NODE) {
                        let selector = element.tagName.toLowerCase();
                        
                        if (element.id) {
                            selector += '#' + element.id;
                            path.unshift(selector);
                            break;
                        }
                        
                        if (element.className) {
                            const classes = element.className.split(' ').filter(c => c);
                            if (classes.length > 0) {
                                selector += '.' + classes.join('.');
                            }
                        }
                        
                        let index = 1;
                        let sibling = element.previousElementSibling;
                        while (sibling) {
                            if (sibling.tagName === element.tagName) {
                                index++;
                            }
                            sibling = sibling.previousElementSibling;
                        }
                        
                        if (index > 1) {
                            selector += `:nth-of-type(${index})`;
                        }
                        
                        path.unshift(selector);
                        element = element.parentElement;
                    }
                    
                    return path.join(' > ');
                }
            """)
            
            return selector
            
        except Exception as e:
            self.logger.warning(f"Failed to generate CSS selector: {e}")
            return None
            
    async def _rank_alternatives(self, page, alternatives: List[Dict[str, Any]], 
                               element_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank alternative selectors by confidence."""
        try:
            ranked_alternatives = []
            
            for alt in alternatives:
                confidence = await self._calculate_selector_confidence(page, alt, element_context)
                alt["confidence"] = confidence
                ranked_alternatives.append(alt)
                
            # Sort by confidence (highest first)
            ranked_alternatives.sort(key=lambda x: x["confidence"], reverse=True)
            
            return ranked_alternatives
            
        except Exception as e:
            self.logger.error(f"Failed to rank alternatives: {e}", exc_info=True)
            return alternatives
            
    async def _calculate_selector_confidence(self, page, alternative: Dict[str, Any], 
                                           element_context: Dict[str, Any]) -> float:
        """Calculate confidence score for an alternative selector."""
        try:
            confidence = 0.0
            
            # Test if selector works
            try:
                elements = page.locator(alternative["selector"])
                count = await elements.count()
                
                if count == 0:
                    return 0.0
                elif count == 1:
                    confidence += 0.3  # Unique element
                else:
                    confidence += 0.1  # Multiple elements
                    
            except Exception:
                return 0.0
                
            # Strategy-based confidence
            strategy_weights = {
                "text_content": 0.4,
                "attributes": 0.3,
                "position": 0.2,
                "visual_similarity": 0.35,
                "semantic_similarity": 0.25,
                "xpath": 0.3,
                "context_xpath": 0.25
            }
            
            strategy = alternative.get("strategy", "")
            confidence += strategy_weights.get(strategy, 0.1)
            
            # Similarity-based confidence
            if "similarity" in alternative:
                confidence += alternative["similarity"] * 0.3
                
            # Context matching
            if element_context and self._matches_context(alternative, element_context):
                confidence += 0.2
                
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence: {e}")
            return 0.0
            
    def _matches_context(self, alternative: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if alternative matches the original context."""
        try:
            # Check text matching
            if context.get("text") and alternative.get("text"):
                if context["text"].lower() in alternative["text"].lower():
                    return True
                    
            # Check attribute matching
            if context.get("attributes") and alternative.get("attribute"):
                context_attrs = context["attributes"]
                alt_attr = alternative.get("attribute")
                alt_value = alternative.get("value")
                
                if alt_attr in context_attrs and context_attrs[alt_attr] == alt_value:
                    return True
                    
            return False
            
        except Exception:
            return False
            
    async def _calculate_visual_hash(self, page, element) -> Optional[str]:
        """Calculate visual hash for an element."""
        try:
            if not ML_AVAILABLE:
                return None
                
            # Take screenshot of element
            screenshot_bytes = await element.screenshot()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Resize for consistent hashing
            image = image.resize((8, 8), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Calculate average pixel value
            pixels = list(image.getdata())
            avg_pixel = sum(pixels) / len(pixels)
            
            # Create hash
            hash_bits = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
            return hash_bits
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate visual hash: {e}")
            return None
            
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two visual hashes."""
        try:
            if len(hash1) != len(hash2):
                return 0.0
                
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            
            # Convert to similarity (0-1)
            similarity = 1.0 - (distance / len(hash1))
            return similarity
            
        except Exception:
            return 0.0
            
    def _extract_selector_features(self, selector: str) -> str:
        """Extract semantic features from a selector."""
        try:
            # Extract tag names, classes, IDs, attributes
            features = []
            
            # Tag names
            tag_match = re.search(r'^([a-zA-Z]+)', selector)
            if tag_match:
                features.append(tag_match.group(1))
                
            # Classes
            class_matches = re.findall(r'\.([a-zA-Z0-9_-]+)', selector)
            features.extend(class_matches)
            
            # IDs
            id_matches = re.findall(r'#([a-zA-Z0-9_-]+)', selector)
            features.extend(id_matches)
            
            # Attributes
            attr_matches = re.findall(r'\[([a-zA-Z0-9_-]+)', selector)
            features.extend(attr_matches)
            
            return ' '.join(features)
            
        except Exception:
            return selector
            
    def _calculate_semantic_similarity(self, features1: str, features2: str) -> float:
        """Calculate semantic similarity between feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
                
            # Use TF-IDF vectorization
            vectors = self.vectorizer.fit_transform([features1, features2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return similarity
            
        except Exception:
            return 0.0
            
    def _css_to_xpath(self, css_selector: str) -> Optional[str]:
        """Convert CSS selector to XPath."""
        try:
            # Simple CSS to XPath conversion
            xpath = css_selector
            
            # Replace common patterns
            xpath = re.sub(r'#([a-zA-Z0-9_-]+)', r'[@id="\1"]', xpath)
            xpath = re.sub(r'\.([a-zA-Z0-9_-]+)', r'[contains(@class, "\1")]', xpath)
            xpath = re.sub(r'\[([a-zA-Z0-9_-]+)="([^"]*)"\]', r'[@\1="\2"]', xpath)
            
            # Add element tag if not present
            if not re.match(r'^[a-zA-Z]', xpath):
                xpath = f"//*{xpath}"
            else:
                xpath = f"//{xpath}"
                
            return xpath
            
        except Exception:
            return None
            
    def _generate_context_xpath(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate XPath based on element context."""
        try:
            xpath_parts = []
            
            # Text-based XPath
            if context.get("text"):
                xpath_parts.append(f'contains(text(), "{context["text"]}")')
                
            # Attribute-based XPath
            if context.get("attributes"):
                for attr, value in context["attributes"].items():
                    if value:
                        xpath_parts.append(f'@{attr}="{value}"')
                        
            if xpath_parts:
                return f"//*[{' and '.join(xpath_parts)}]"
                
            return None
            
        except Exception:
            return None
            
    async def _store_drift_pattern(self, original_selector: str, alternatives: List[Dict[str, Any]], 
                                 element_context: Dict[str, Any]):
        """Store drift pattern for learning."""
        try:
            pattern_id = hashlib.md5(original_selector.encode()).hexdigest()
            
            self.drift_patterns[pattern_id] = {
                "original_selector": original_selector,
                "alternatives": alternatives,
                "context": element_context,
                "timestamp": datetime.utcnow().isoformat(),
                "success_count": 0,
                "failure_count": 0
            }
            
            # Store in selector history
            if original_selector not in self.selector_history:
                self.selector_history[original_selector] = []
                
            self.selector_history[original_selector].append({
                "timestamp": datetime.utcnow().isoformat(),
                "drift_detected": True,
                "alternatives_found": len(alternatives),
                "best_alternative": alternatives[0] if alternatives else None
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to store drift pattern: {e}")
            
    async def suggest_alternative(self, original_selector: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Suggest alternative selector based on learned patterns."""
        try:
            # Check if we have learned patterns for this selector
            if original_selector in self.selector_history:
                history = self.selector_history[original_selector]
                
                # Find most recent successful alternative
                for entry in reversed(history):
                    if entry.get("best_alternative"):
                        return entry["best_alternative"]["selector"]
                        
            # Check drift patterns
            pattern_id = hashlib.md5(original_selector.encode()).hexdigest()
            if pattern_id in self.drift_patterns:
                pattern = self.drift_patterns[pattern_id]
                
                # Return best alternative from pattern
                if pattern["alternatives"]:
                    return pattern["alternatives"][0]["selector"]
                    
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to suggest alternative: {e}")
            return None
            
    async def get_drift_statistics(self) -> Dict[str, Any]:
        """Get statistics about drift detection."""
        try:
            stats = {
                "total_patterns": len(self.drift_patterns),
                "total_selectors": len(self.selector_history),
                "total_drifts": sum(len(history) for history in self.selector_history.values()),
                "successful_fixes": sum(
                    sum(1 for entry in history if entry.get("best_alternative"))
                    for history in self.selector_history.values()
                )
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get drift statistics: {e}", exc_info=True)
            return {}
            
    async def shutdown(self):
        """Shutdown selector drift detector."""
        try:
            # Save learned patterns to file
            await self._save_patterns()
            
            self.logger.info("Selector drift detector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during selector drift detector shutdown: {e}", exc_info=True)
            
    async def _save_patterns(self):
        """Save learned patterns to file."""
        try:
            patterns_file = Path(self.config.data_path) / "drift_patterns.json"
            
            # Convert patterns to serializable format
            serializable_patterns = {}
            for pattern_id, pattern in self.drift_patterns.items():
                serializable_patterns[pattern_id] = {
                    "original_selector": pattern["original_selector"],
                    "alternatives": pattern["alternatives"],
                    "context": pattern["context"],
                    "timestamp": pattern["timestamp"],
                    "success_count": pattern["success_count"],
                    "failure_count": pattern["failure_count"]
                }
                
            with open(patterns_file, "w") as f:
                json.dump(serializable_patterns, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save patterns: {e}")