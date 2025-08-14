"""
Self-Healing Locator Stack
==========================

Selector resilience with multiple fallback strategies:
1. Role+Accessible Name query
2. CSS/XPath canonical selector  
3. Semantic text embedding nearest-neighbor
4. Visual template (node screenshot) similarity
5. Context re-rank (near label/anchor; same parent lineage)

Healing Algorithm:
- If exact selectors work, return element
- Get candidates by role/name, semantic text, visual similarity
- Re-rank with context
- Simulate each candidate to verify postconditions
- Persist alternative selector if successful
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import json

try:
    from playwright.async_api import Page, ElementHandle, Locator
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .semantic_dom_graph import SemanticDOMGraph, DOMNode, BoundingBox
from ..models.contracts import TargetSelector, StepContract


@dataclass
class LocatorStrategy:
    """A locator strategy with priority and confidence."""
    name: str
    selector: str
    priority: int
    confidence: float
    last_success: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class ElementCandidate:
    """Element candidate with scoring information."""
    element: ElementHandle
    node_id: str
    confidence: float
    strategy: str
    metadata: Dict[str, Any]


class SelfHealingLocatorStack:
    """
    Self-healing locator system with multiple fallback strategies.
    
    Priority order:
    1. Role+Accessible Name query
    2. CSS/XPath canonical selector
    3. Semantic text embedding nearest-neighbor
    4. Visual template similarity
    5. Context re-ranking
    """
    
    def __init__(self, semantic_graph: SemanticDOMGraph, config: Any = None):
        self.semantic_graph = semantic_graph
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Locator persistence
        self.locator_cache: Dict[str, List[LocatorStrategy]] = {}
        self.healing_stats = {
            'total_resolves': 0,
            'healed_resolves': 0,
            'heal_time_total': 0.0,
            'strategy_stats': {}
        }
        
        # Configuration
        self.max_candidates = 10
        self.confidence_threshold = 0.7
        self.visual_similarity_threshold = 0.8
        self.semantic_similarity_threshold = 0.75
        
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using simple metrics."""
        if not text1 or not text2:
            return 0.0
        
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_embedding_similarity(self, embed1: List[float], embed2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        if not embed1 or not embed2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            a = np.array(embed1)
            b = np.array(embed2)
            
            # Compute cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except Exception as e:
            self.logger.warning(f"Failed to compute embedding similarity: {e}")
            return 0.0
    
    def _compute_visual_similarity(self, hash1: str, hash2: str) -> float:
        """Compute visual similarity from hashes."""
        if not hash1 or not hash2:
            return 0.0
        
        if hash1 == hash2:
            return 1.0
        
        # Simple hamming distance for hashes
        try:
            # Convert hex to binary
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            
            # Calculate hamming distance
            hamming = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            max_distance = len(bin1)
            
            return 1.0 - (hamming / max_distance) if max_distance > 0 else 0.0
        except Exception:
            return 0.0
    
    async def _find_by_role_name(self, page: Page, target: TargetSelector) -> List[ElementCandidate]:
        """Find elements by role and accessible name."""
        candidates = []
        
        try:
            if target.role and target.name:
                # Use Playwright's role selector
                locator = page.get_by_role(target.role, name=target.name)
                elements = await locator.all()
                
                for element in elements:
                    candidates.append(ElementCandidate(
                        element=element,
                        node_id="",  # Will be filled later
                        confidence=0.95,
                        strategy="role_name",
                        metadata={"role": target.role, "name": target.name}
                    ))
            
            elif target.role:
                # Find by role only
                locator = page.get_by_role(target.role)
                elements = await locator.all()
                
                for element in elements:
                    candidates.append(ElementCandidate(
                        element=element,
                        node_id="",
                        confidence=0.8,
                        strategy="role_only",
                        metadata={"role": target.role}
                    ))
            
            elif target.name:
                # Find by accessible name
                locator = page.get_by_label(target.name)
                elements = await locator.all()
                
                for element in elements:
                    candidates.append(ElementCandidate(
                        element=element,
                        node_id="",
                        confidence=0.7,
                        strategy="name_only",
                        metadata={"name": target.name}
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Failed to find by role/name: {e}")
        
        return candidates
    
    async def _find_by_css_xpath(self, page: Page, target: TargetSelector) -> List[ElementCandidate]:
        """Find elements by CSS or XPath selectors."""
        candidates = []
        
        try:
            if target.css:
                elements = await page.query_selector_all(target.css)
                for element in elements:
                    candidates.append(ElementCandidate(
                        element=element,
                        node_id="",
                        confidence=0.9,
                        strategy="css",
                        metadata={"css": target.css}
                    ))
            
            if target.xpath:
                elements = await page.query_selector_all(f"xpath={target.xpath}")
                for element in elements:
                    candidates.append(ElementCandidate(
                        element=element,
                        node_id="",
                        confidence=0.9,
                        strategy="xpath",
                        metadata={"xpath": target.xpath}
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Failed to find by CSS/XPath: {e}")
        
        return candidates
    
    def _find_by_semantic_text(self, target: TargetSelector) -> List[str]:
        """Find nodes by semantic text similarity."""
        candidates = []
        
        if not target.text:
            return candidates
        
        # Search in semantic graph
        for node_id, node in self.semantic_graph.nodes.items():
            if not node.text_embed:
                continue
            
            # Compare text directly
            text_sim = self._compute_text_similarity(target.text, node.text_norm or "")
            
            # Compare embeddings if available
            embed_sim = 0.0
            if target.semantic_embedding and node.text_embed:
                embed_sim = self._compute_embedding_similarity(target.semantic_embedding, node.text_embed)
            
            # Combined similarity
            similarity = max(text_sim, embed_sim)
            
            if similarity >= self.semantic_similarity_threshold:
                candidates.append((node_id, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:self.max_candidates]]
    
    def _find_by_visual_template(self, target: TargetSelector) -> List[str]:
        """Find nodes by visual template similarity."""
        candidates = []
        
        if not target.visual_template:
            return candidates
        
        # Load target visual hash (simplified - in practice would load from file)
        target_hash = target.visual_template
        
        for node_id, node in self.semantic_graph.nodes.items():
            if not node.visual_hash:
                continue
            
            similarity = self._compute_visual_similarity(target_hash, node.visual_hash)
            
            if similarity >= self.visual_similarity_threshold:
                candidates.append((node_id, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:self.max_candidates]]
    
    async def _get_element_from_node(self, page: Page, node: DOMNode) -> Optional[ElementHandle]:
        """Get Playwright element from semantic node."""
        try:
            # Try various selectors from the node
            selectors_to_try = []
            
            if node.xpath:
                selectors_to_try.append(f"xpath={node.xpath}")
            
            if node.attributes and 'id' in node.attributes:
                selectors_to_try.append(f"#{node.attributes['id']}")
            
            if node.css_properties:
                # Build CSS selector from properties
                css_parts = [node.tag_name]
                if node.attributes:
                    if 'class' in node.attributes:
                        classes = node.attributes['class'].split()
                        css_parts.extend([f".{cls}" for cls in classes])
                    if 'id' in node.attributes:
                        css_parts.append(f"#{node.attributes['id']}")
                
                css_selector = "".join(css_parts)
                selectors_to_try.append(css_selector)
            
            # Try selectors in order
            for selector in selectors_to_try:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        return element
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get element from node: {e}")
            return None
    
    def _rerank_with_context(self, candidates: List[ElementCandidate], target: TargetSelector) -> List[ElementCandidate]:
        """Re-rank candidates using context information."""
        if not target.context:
            return candidates
        
        # Simple context scoring based on proximity to context elements
        for candidate in candidates:
            context_score = 0.0
            
            # Find context elements and compute proximity
            # This is simplified - in practice would use semantic graph relationships
            if "near" in target.context.lower():
                context_score += 0.2
            
            if "button" in target.context.lower() and candidate.metadata.get("role") == "button":
                context_score += 0.3
            
            # Adjust confidence based on context
            candidate.confidence = min(1.0, candidate.confidence + context_score)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates
    
    async def _simulate_action(self, page: Page, element: ElementHandle, action_type: str) -> bool:
        """
        Simulate action on element to verify it's the correct target.
        This is a simplified version - full implementation would use shadow DOM simulator.
        """
        try:
            # Check if element is actionable
            is_visible = await element.is_visible()
            is_enabled = await element.is_enabled()
            
            if not is_visible or not is_enabled:
                return False
            
            # For now, just return True if element is actionable
            # Full implementation would simulate the action and check postconditions
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to simulate action: {e}")
            return False
    
    def _persist_alternative_selector(self, target_key: str, strategy: LocatorStrategy):
        """Persist successful alternative selector for future use."""
        if target_key not in self.locator_cache:
            self.locator_cache[target_key] = []
        
        # Add or update strategy
        existing = None
        for i, existing_strategy in enumerate(self.locator_cache[target_key]):
            if existing_strategy.name == strategy.name:
                existing = i
                break
        
        if existing is not None:
            self.locator_cache[target_key][existing] = strategy
        else:
            self.locator_cache[target_key].append(strategy)
        
        # Sort by success rate and priority
        self.locator_cache[target_key].sort(
            key=lambda s: (s.success_rate(), s.priority), 
            reverse=True
        )
    
    async def resolve(self, page: Page, target: TargetSelector, action_type: str = "click") -> Optional[ElementHandle]:
        """
        Main healing algorithm implementation.
        
        Healing Algorithm (pseudocode):
        resolve(target):
          if exact_selectors_work(): return elem
          C = candidates_by_role_name() ∪ by_semantic_text() ∪ by_visual()
          C = rerank_with_context(C)
          for c in C:
            if simulate(c, action).post_ok: persist_alt_selector(c); return c
          raise NotFound
        """
        start_time = datetime.utcnow()
        self.healing_stats['total_resolves'] += 1
        
        target_key = f"{target.role}:{target.name}:{target.css}:{target.xpath}"
        
        try:
            # Step 1: Try exact selectors first
            exact_candidates = await self._find_by_css_xpath(page, target)
            
            for candidate in exact_candidates:
                if await self._simulate_action(page, candidate.element, action_type):
                    return candidate.element
            
            # Step 2: If exact selectors don't work, start healing
            self.logger.info(f"Starting healing process for target: {target_key}")
            self.healing_stats['healed_resolves'] += 1
            
            # Collect candidates from all strategies
            all_candidates = []
            
            # Role+Name candidates
            role_name_candidates = await self._find_by_role_name(page, target)
            all_candidates.extend(role_name_candidates)
            
            # Semantic text candidates
            semantic_node_ids = self._find_by_semantic_text(target)
            for node_id in semantic_node_ids:
                if node_id in self.semantic_graph.nodes:
                    node = self.semantic_graph.nodes[node_id]
                    element = await self._get_element_from_node(page, node)
                    if element:
                        all_candidates.append(ElementCandidate(
                            element=element,
                            node_id=node_id,
                            confidence=0.75,
                            strategy="semantic_text",
                            metadata={"node_id": node_id}
                        ))
            
            # Visual template candidates
            visual_node_ids = self._find_by_visual_template(target)
            for node_id in visual_node_ids:
                if node_id in self.semantic_graph.nodes:
                    node = self.semantic_graph.nodes[node_id]
                    element = await self._get_element_from_node(page, node)
                    if element:
                        all_candidates.append(ElementCandidate(
                            element=element,
                            node_id=node_id,
                            confidence=0.7,
                            strategy="visual_template",
                            metadata={"node_id": node_id}
                        ))
            
            # Step 3: Re-rank with context
            all_candidates = self._rerank_with_context(all_candidates, target)
            
            # Step 4: Test candidates and persist successful ones
            for candidate in all_candidates:
                if candidate.confidence >= self.confidence_threshold:
                    if await self._simulate_action(page, candidate.element, action_type):
                        # Success! Persist this strategy
                        strategy = LocatorStrategy(
                            name=candidate.strategy,
                            selector=str(candidate.metadata),
                            priority=1,
                            confidence=candidate.confidence,
                            last_success=datetime.utcnow(),
                            success_count=1,
                            failure_count=0
                        )
                        
                        self._persist_alternative_selector(target_key, strategy)
                        
                        # Update stats
                        heal_time = (datetime.utcnow() - start_time).total_seconds()
                        self.healing_stats['heal_time_total'] += heal_time
                        
                        strategy_name = candidate.strategy
                        if strategy_name not in self.healing_stats['strategy_stats']:
                            self.healing_stats['strategy_stats'][strategy_name] = {'success': 0, 'failure': 0}
                        self.healing_stats['strategy_stats'][strategy_name]['success'] += 1
                        
                        self.logger.info(f"Healed selector in {heal_time:.2f}s using {strategy_name}")
                        return candidate.element
            
            # No candidates worked
            self.logger.error(f"Failed to heal selector for target: {target_key}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in healing algorithm: {e}")
            return None
    
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get healing statistics."""
        stats = self.healing_stats.copy()
        
        if stats['healed_resolves'] > 0:
            stats['average_heal_time'] = stats['heal_time_total'] / stats['healed_resolves']
            stats['heal_rate'] = stats['healed_resolves'] / stats['total_resolves']
        else:
            stats['average_heal_time'] = 0.0
            stats['heal_rate'] = 0.0
        
        return stats
    
    def get_cached_strategies(self, target_key: str) -> List[LocatorStrategy]:
        """Get cached strategies for a target."""
        return self.locator_cache.get(target_key, [])
    
    def clear_cache(self):
        """Clear the locator cache."""
        self.locator_cache.clear()
        self.healing_stats = {
            'total_resolves': 0,
            'healed_resolves': 0,
            'heal_time_total': 0.0,
            'strategy_stats': {}
        }