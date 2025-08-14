"""
Semantic DOM Graph
==================

The universalizer that normalizes any UI by building a semantic graph
from AccTree + HTML + CSS + screenshot crop per node.

Features:
- Role, text normalization, attributes, bbox, CSS, XPath, ARIA
- Vision embeddings and text embeddings per node
- Fingerprinting for drift detection
- Delta snapshots for time-machine capabilities
- Node matching and similarity scoring
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import io
import base64

try:
    from playwright.async_api import Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class BoundingBox:
    """Element bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def area(self) -> float:
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class DOMNode:
    """Semantic DOM node with all required properties."""
    id: str
    tag_name: str
    role: Optional[str] = None
    text_content: Optional[str] = None
    text_norm: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    bbox: Optional[BoundingBox] = None
    css_properties: Optional[Dict[str, str]] = None
    xpath: Optional[str] = None
    aria_label: Optional[str] = None
    aria_role: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    # Embeddings and fingerprints
    vision_embed: Optional[List[float]] = None
    text_embed: Optional[List[float]] = None
    fingerprint: Optional[str] = None
    
    # Visual data
    screenshot_crop: Optional[str] = None  # Base64 encoded image
    visual_hash: Optional[str] = None
    
    # Metadata
    timestamp: datetime = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.attributes is None:
            self.attributes = {}
        if self.css_properties is None:
            self.css_properties = {}


class SemanticDOMGraph:
    """
    Semantic DOM Graph builder and manager.
    
    Builds graph from AccTree + HTML + CSS + screenshot crop per node.
    Maintains delta snapshots for drift/time-machine capabilities.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Graph storage
        self.nodes: Dict[str, DOMNode] = {}
        self.snapshots: List[Dict[str, Any]] = []
        self.current_snapshot_id: Optional[str] = None
        
        # Embeddings model
        self.text_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.logger.warning(f"Failed to load text embedding model: {e}")
        
        # Node ID counter
        self._node_id_counter = 0
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self._node_id_counter += 1
        return f"node_{self._node_id_counter}"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text content for consistent matching."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        normalized = " ".join(text.strip().split())
        
        # Convert to lowercase for comparison
        normalized = normalized.lower()
        
        return normalized
    
    def _compute_text_embedding(self, text: str) -> Optional[List[float]]:
        """Compute text embedding using sentence transformer."""
        if not self.text_model or not text:
            return None
        
        try:
            embedding = self.text_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            self.logger.warning(f"Failed to compute text embedding: {e}")
            return None
    
    def _compute_visual_hash(self, image_data: bytes) -> str:
        """Compute perceptual hash of image for visual similarity."""
        if not CV2_AVAILABLE:
            # Fallback to simple hash
            return hashlib.md5(image_data).hexdigest()
        
        try:
            # Convert to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize to standard size for comparison
            img_resized = cv2.resize(img, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Compute hash
            return hashlib.md5(gray.tobytes()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to compute visual hash: {e}")
            return hashlib.md5(image_data).hexdigest()
    
    def _compute_fingerprint(self, node: DOMNode) -> str:
        """
        Compute node fingerprint: hash(role|text_norm|top-k(embed)|bbox_q).
        """
        fingerprint_parts = []
        
        # Role
        if node.role:
            fingerprint_parts.append(f"role:{node.role}")
        
        # Normalized text
        if node.text_norm:
            fingerprint_parts.append(f"text:{node.text_norm}")
        
        # Top-k embedding values (first 10 dimensions)
        if node.text_embed:
            top_k = node.text_embed[:10]
            embed_str = ",".join([f"{x:.3f}" for x in top_k])
            fingerprint_parts.append(f"embed:{embed_str}")
        
        # Quantized bounding box
        if node.bbox:
            bbox_q = f"{int(node.bbox.x/10)*10},{int(node.bbox.y/10)*10},{int(node.bbox.width/10)*10},{int(node.bbox.height/10)*10}"
            fingerprint_parts.append(f"bbox:{bbox_q}")
        
        # Tag name
        fingerprint_parts.append(f"tag:{node.tag_name}")
        
        # Combine and hash
        fingerprint_data = "|".join(fingerprint_parts)
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    async def _capture_element_screenshot(self, page: Page, element: ElementHandle) -> Optional[str]:
        """Capture screenshot crop of element."""
        try:
            # Get element screenshot
            screenshot_bytes = await element.screenshot()
            
            # Convert to base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            
            return screenshot_b64
        except Exception as e:
            self.logger.warning(f"Failed to capture element screenshot: {e}")
            return None
    
    async def _extract_css_properties(self, page: Page, element: ElementHandle) -> Dict[str, str]:
        """Extract relevant CSS properties from element."""
        try:
            css_props = await page.evaluate("""
                (element) => {
                    const style = window.getComputedStyle(element);
                    return {
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        position: style.position,
                        zIndex: style.zIndex,
                        backgroundColor: style.backgroundColor,
                        color: style.color,
                        fontSize: style.fontSize,
                        fontFamily: style.fontFamily,
                        border: style.border,
                        padding: style.padding,
                        margin: style.margin
                    };
                }
            """, element)
            
            return css_props
        except Exception as e:
            self.logger.warning(f"Failed to extract CSS properties: {e}")
            return {}
    
    async def _extract_xpath(self, page: Page, element: ElementHandle) -> Optional[str]:
        """Extract XPath for element."""
        try:
            xpath = await page.evaluate("""
                (element) => {
                    function getXPath(node) {
                        if (node.id !== '') {
                            return `//*[@id="${node.id}"]`;
                        }
                        if (node === document.body) {
                            return '/html/body';
                        }
                        
                        let ix = 0;
                        const siblings = node.parentNode.childNodes;
                        for (let i = 0; i < siblings.length; i++) {
                            const sibling = siblings[i];
                            if (sibling === node) {
                                return getXPath(node.parentNode) + '/' + node.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === node.tagName) {
                                ix++;
                            }
                        }
                    }
                    return getXPath(element);
                }
            """, element)
            
            return xpath
        except Exception as e:
            self.logger.warning(f"Failed to extract XPath: {e}")
            return None
    
    async def build_from_page(self, page: Page, capture_screenshots: bool = True) -> str:
        """
        Build semantic DOM graph from Playwright page.
        
        Args:
            page: Playwright page object
            capture_screenshots: Whether to capture element screenshots
            
        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot_{datetime.utcnow().isoformat()}"
        self.current_snapshot_id = snapshot_id
        
        try:
            # Get all elements with accessibility tree information
            elements = await page.query_selector_all('*')
            
            # Clear current nodes
            self.nodes.clear()
            
            for element in elements:
                try:
                    node_id = self._generate_node_id()
                    
                    # Get basic element info
                    tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                    text_content = await element.text_content()
                    
                    # Get attributes
                    attributes = await element.evaluate("""
                        el => {
                            const attrs = {};
                            for (let attr of el.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return attrs;
                        }
                    """)
                    
                    # Get bounding box
                    bbox_data = await element.bounding_box()
                    bbox = None
                    if bbox_data:
                        bbox = BoundingBox(
                            x=bbox_data['x'],
                            y=bbox_data['y'],
                            width=bbox_data['width'],
                            height=bbox_data['height']
                        )
                    
                    # Get accessibility info
                    accessibility_info = await element.evaluate("""
                        el => ({
                            role: el.getAttribute('role') || el.ariaRole,
                            label: el.getAttribute('aria-label') || el.ariaLabel,
                            name: el.getAttribute('aria-labelledby') || el.textContent?.trim()
                        })
                    """)
                    
                    # Normalize text
                    text_norm = self._normalize_text(text_content or "")
                    
                    # Compute text embedding
                    text_embed = self._compute_text_embedding(text_norm)
                    
                    # Get CSS properties
                    css_properties = await self._extract_css_properties(page, element)
                    
                    # Get XPath
                    xpath = await self._extract_xpath(page, element)
                    
                    # Capture screenshot if requested
                    screenshot_crop = None
                    visual_hash = None
                    if capture_screenshots and bbox and bbox.area() > 0:
                        screenshot_crop = await self._capture_element_screenshot(page, element)
                        if screenshot_crop:
                            screenshot_bytes = base64.b64decode(screenshot_crop)
                            visual_hash = self._compute_visual_hash(screenshot_bytes)
                    
                    # Create node
                    node = DOMNode(
                        id=node_id,
                        tag_name=tag_name,
                        role=accessibility_info.get('role'),
                        text_content=text_content,
                        text_norm=text_norm,
                        attributes=attributes,
                        bbox=bbox,
                        css_properties=css_properties,
                        xpath=xpath,
                        aria_label=accessibility_info.get('label'),
                        aria_role=accessibility_info.get('role'),
                        text_embed=text_embed,
                        screenshot_crop=screenshot_crop,
                        visual_hash=visual_hash
                    )
                    
                    # Compute fingerprint
                    node.fingerprint = self._compute_fingerprint(node)
                    
                    # Store node
                    self.nodes[node_id] = node
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process element: {e}")
                    continue
            
            # Build parent-child relationships
            await self._build_relationships(page)
            
            # Save snapshot
            snapshot_data = {
                'id': snapshot_id,
                'timestamp': datetime.utcnow().isoformat(),
                'node_count': len(self.nodes),
                'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()}
            }
            self.snapshots.append(snapshot_data)
            
            self.logger.info(f"Built semantic DOM graph with {len(self.nodes)} nodes")
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Failed to build semantic DOM graph: {e}")
            raise
    
    async def _build_relationships(self, page: Page):
        """Build parent-child relationships between nodes."""
        try:
            # Get DOM tree structure
            tree_data = await page.evaluate("""
                () => {
                    function buildTree(element, nodeMap = new Map()) {
                        const children = Array.from(element.children);
                        const childIds = [];
                        
                        children.forEach((child, index) => {
                            const childId = `node_${nodeMap.size + 1}`;
                            nodeMap.set(child, childId);
                            childIds.push(childId);
                            buildTree(child, nodeMap);
                        });
                        
                        return { nodeMap, childIds };
                    }
                    
                    const result = buildTree(document.body);
                    return Array.from(result.nodeMap.entries()).map(([el, id]) => ({
                        id,
                        parentId: el.parentElement ? 'parent_id' : null,
                        children: Array.from(el.children).map(() => 'child_id')
                    }));
                }
            """)
            
            # Update node relationships (simplified for now)
            # In a full implementation, we'd need to map elements to our node IDs
            
        except Exception as e:
            self.logger.warning(f"Failed to build relationships: {e}")
    
    def get_semantic_graph(self) -> Dict[str, DOMNode]:
        """Get current semantic graph."""
        return self.nodes.copy()
    
    def match_nodes(self, graph_a: Dict[str, DOMNode], graph_b: Dict[str, DOMNode]) -> Tuple[Dict[str, str], float]:
        """
        Match nodes between two graphs and compute similarity.
        
        Returns:
            Tuple of (node_mapping, overall_similarity)
        """
        mapping = {}
        total_matches = 0
        total_nodes = max(len(graph_a), len(graph_b))
        
        if total_nodes == 0:
            return mapping, 1.0
        
        # Create fingerprint maps
        fingerprints_a = {node.fingerprint: node_id for node_id, node in graph_a.items() if node.fingerprint}
        fingerprints_b = {node.fingerprint: node_id for node_id, node in graph_b.items() if node.fingerprint}
        
        # Match by fingerprint first
        for fingerprint, node_id_a in fingerprints_a.items():
            if fingerprint in fingerprints_b:
                node_id_b = fingerprints_b[fingerprint]
                mapping[node_id_a] = node_id_b
                total_matches += 1
        
        # Calculate similarity
        similarity = total_matches / total_nodes if total_nodes > 0 else 0.0
        
        return mapping, similarity
    
    def query(self, role: Optional[str] = None, name: Optional[str] = None, 
              text: Optional[str] = None, near: Optional[str] = None, k: int = 10) -> List[str]:
        """
        Query nodes by various criteria.
        
        Args:
            role: ARIA role to match
            name: Accessible name to match
            text: Text content to match
            near: Node ID to find nodes near
            k: Maximum number of results
            
        Returns:
            List of matching node IDs
        """
        results = []
        
        for node_id, node in self.nodes.items():
            score = 0.0
            
            # Role matching
            if role and node.role:
                if node.role.lower() == role.lower():
                    score += 1.0
                elif role.lower() in node.role.lower():
                    score += 0.5
            
            # Name matching (aria-label or text content)
            if name:
                if node.aria_label and name.lower() in node.aria_label.lower():
                    score += 1.0
                elif node.text_norm and name.lower() in node.text_norm:
                    score += 0.8
            
            # Text matching
            if text and node.text_norm:
                if text.lower() in node.text_norm:
                    score += 1.0
            
            # Proximity matching (simplified)
            if near and near in self.nodes:
                near_node = self.nodes[near]
                if node.bbox and near_node.bbox:
                    # Calculate distance
                    center1 = node.bbox.center()
                    center2 = near_node.bbox.center()
                    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                    
                    # Closer nodes get higher scores
                    if distance < 100:  # Within 100 pixels
                        score += 1.0 - (distance / 100)
            
            if score > 0:
                results.append((node_id, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, score in results[:k]]
    
    def get_node_by_fingerprint(self, fingerprint: str) -> Optional[DOMNode]:
        """Get node by fingerprint."""
        for node in self.nodes.values():
            if node.fingerprint == fingerprint:
                return node
        return None
    
    def compute_drift(self, previous_snapshot_id: str) -> Dict[str, Any]:
        """
        Compute drift between current graph and previous snapshot.
        
        Returns:
            Drift analysis with added, removed, and modified nodes
        """
        if not self.snapshots:
            return {"error": "No previous snapshots available"}
        
        # Find previous snapshot
        previous_snapshot = None
        for snapshot in self.snapshots:
            if snapshot['id'] == previous_snapshot_id:
                previous_snapshot = snapshot
                break
        
        if not previous_snapshot:
            return {"error": f"Snapshot {previous_snapshot_id} not found"}
        
        # Reconstruct previous nodes
        previous_nodes = {}
        for node_id, node_data in previous_snapshot['nodes'].items():
            # Convert dict back to DOMNode (simplified)
            previous_nodes[node_id] = DOMNode(**node_data)
        
        # Compare graphs
        mapping, similarity = self.match_nodes(previous_nodes, self.nodes)
        
        # Find changes
        current_fingerprints = {node.fingerprint for node in self.nodes.values() if node.fingerprint}
        previous_fingerprints = {node.fingerprint for node in previous_nodes.values() if node.fingerprint}
        
        added_fingerprints = current_fingerprints - previous_fingerprints
        removed_fingerprints = previous_fingerprints - current_fingerprints
        
        return {
            "similarity": similarity,
            "mapping": mapping,
            "added_nodes": len(added_fingerprints),
            "removed_nodes": len(removed_fingerprints),
            "total_nodes_current": len(self.nodes),
            "total_nodes_previous": len(previous_nodes),
            "drift_score": 1.0 - similarity
        }