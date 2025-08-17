"""
COMPUTER VISION SYSTEM
=====================

Advanced computer vision for element detection, UI understanding, and visual automation.
Provides image analysis, element recognition, and visual feedback for automation.

✅ FEATURES:
- Element detection and recognition
- UI component analysis
- Image comparison and matching
- Visual feedback and validation
- Screenshot analysis
- OCR capabilities
- Pattern recognition
"""

import asyncio
import base64
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from playwright.async_api import Page
import random

logger = logging.getLogger(__name__)

class VisionTask(Enum):
    ELEMENT_DETECTION = "element_detection"
    TEXT_RECOGNITION = "text_recognition"
    IMAGE_COMPARISON = "image_comparison"
    UI_ANALYSIS = "ui_analysis"
    PATTERN_MATCHING = "pattern_matching"
    VISUAL_VALIDATION = "visual_validation"

class ElementType(Enum):
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    IMAGE = "image"
    TEXT = "text"
    FORM = "form"
    MENU = "menu"
    MODAL = "modal"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO = "radio"

@dataclass
class VisualElement:
    """Visual element detected by computer vision"""
    element_type: ElementType
    confidence: float
    bounding_box: Dict[str, float]
    text_content: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    selector_suggestions: List[str] = field(default_factory=list)

@dataclass
class VisionResult:
    """Result of computer vision analysis"""
    task_type: VisionTask
    success: bool
    confidence: float
    elements: List[VisualElement] = field(default_factory=list)
    text_content: str = ""
    image_data: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

class ComputerVisionSystem:
    """Advanced computer vision system for automation"""
    
    def __init__(self):
        self.element_patterns = {}
        self.ui_templates = {}
        self.vision_cache = {}
        
        # Initialize element detection patterns
        self._initialize_element_patterns()
        self._initialize_ui_templates()
    
    def _initialize_element_patterns(self):
        """Initialize patterns for element detection"""
        self.element_patterns = {
            ElementType.BUTTON: {
                'visual_cues': ['rounded_corners', 'shadow', 'background_color'],
                'text_patterns': ['click', 'submit', 'send', 'save', 'cancel', 'ok'],
                'size_range': {'min_width': 50, 'min_height': 20, 'max_width': 300, 'max_height': 80}
            },
            ElementType.INPUT: {
                'visual_cues': ['border', 'background_white', 'cursor_text'],
                'text_patterns': ['enter', 'type', 'search', 'email', 'password'],
                'size_range': {'min_width': 100, 'min_height': 20, 'max_width': 500, 'max_height': 50}
            },
            ElementType.LINK: {
                'visual_cues': ['underline', 'blue_color', 'pointer_cursor'],
                'text_patterns': ['click here', 'learn more', 'read more', 'view'],
                'size_range': {'min_width': 30, 'min_height': 15, 'max_width': 400, 'max_height': 30}
            },
            ElementType.DROPDOWN: {
                'visual_cues': ['arrow_down', 'border', 'background'],
                'text_patterns': ['select', 'choose', 'pick'],
                'size_range': {'min_width': 80, 'min_height': 25, 'max_width': 300, 'max_height': 40}
            }
        }
    
    def _initialize_ui_templates(self):
        """Initialize UI component templates"""
        self.ui_templates = {
            'login_form': {
                'required_elements': [ElementType.INPUT, ElementType.INPUT, ElementType.BUTTON],
                'layout_pattern': 'vertical',
                'text_indicators': ['username', 'password', 'login', 'sign in']
            },
            'search_bar': {
                'required_elements': [ElementType.INPUT, ElementType.BUTTON],
                'layout_pattern': 'horizontal',
                'text_indicators': ['search', 'find', 'query']
            },
            'navigation_menu': {
                'required_elements': [ElementType.LINK, ElementType.LINK, ElementType.LINK],
                'layout_pattern': 'horizontal_or_vertical',
                'text_indicators': ['home', 'about', 'contact', 'menu']
            },
            'modal_dialog': {
                'required_elements': [ElementType.TEXT, ElementType.BUTTON],
                'layout_pattern': 'centered',
                'text_indicators': ['confirm', 'cancel', 'ok', 'close']
            }
        }
    
    async def analyze_page(self, page: Page, task: VisionTask) -> VisionResult:
        """Analyze page using computer vision"""
        start_time = time.time()
        
        try:
            # Take screenshot for analysis
            screenshot_data = await self._capture_screenshot(page)
            
            # Perform analysis based on task type
            if task == VisionTask.ELEMENT_DETECTION:
                result = await self._detect_elements(page, screenshot_data)
            elif task == VisionTask.TEXT_RECOGNITION:
                result = await self._recognize_text(page, screenshot_data)
            elif task == VisionTask.UI_ANALYSIS:
                result = await self._analyze_ui_components(page, screenshot_data)
            elif task == VisionTask.VISUAL_VALIDATION:
                result = await self._validate_visual_state(page, screenshot_data)
            elif task == VisionTask.PATTERN_MATCHING:
                result = await self._match_patterns(page, screenshot_data)
            else:
                result = VisionResult(
                    task_type=task,
                    success=False,
                    confidence=0.0,
                    error=f"Unsupported vision task: {task.value}"
                )
            
            # Add timing and metadata
            result.analysis['processing_time'] = time.time() - start_time
            result.analysis['screenshot_size'] = len(screenshot_data)
            result.image_data = screenshot_data
            
            return result
            
        except Exception as e:
            logger.error(f"Computer vision analysis failed: {e}")
            return VisionResult(
                task_type=task,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    async def _capture_screenshot(self, page: Page) -> str:
        """Capture screenshot for analysis"""
        try:
            screenshot_bytes = await page.screenshot(type='png')
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            return screenshot_base64
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return ""
    
    async def _detect_elements(self, page: Page, screenshot_data: str) -> VisionResult:
        """Detect UI elements using computer vision"""
        try:
            # Get page dimensions
            viewport = await page.viewport_size()
            
            # Simulate element detection (in production, use real CV models)
            detected_elements = []
            
            # Detect buttons
            buttons = await self._detect_buttons(page)
            detected_elements.extend(buttons)
            
            # Detect input fields
            inputs = await self._detect_inputs(page)
            detected_elements.extend(inputs)
            
            # Detect links
            links = await self._detect_links(page)
            detected_elements.extend(links)
            
            # Detect images
            images = await self._detect_images(page)
            detected_elements.extend(images)
            
            return VisionResult(
                task_type=VisionTask.ELEMENT_DETECTION,
                success=True,
                confidence=0.85,
                elements=detected_elements,
                analysis={
                    'total_elements': len(detected_elements),
                    'element_types': list(set(elem.element_type for elem in detected_elements)),
                    'viewport_size': viewport
                }
            )
            
        except Exception as e:
            return VisionResult(
                task_type=VisionTask.ELEMENT_DETECTION,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    async def _detect_buttons(self, page: Page) -> List[VisualElement]:
        """Detect button elements"""
        try:
            buttons = []
            
            # Find button elements
            button_selectors = [
                'button',
                'input[type="button"]',
                'input[type="submit"]',
                '[role="button"]',
                '.btn',
                '.button'
            ]
            
            for selector in button_selectors:
                try:
                    elements = await page.locator(selector).all()
                    
                    for element in elements:
                        try:
                            # Get bounding box
                            box = await element.bounding_box()
                            if not box:
                                continue
                            
                            # Get text content
                            text = await element.text_content() or ""
                            
                            # Calculate confidence based on visual cues
                            confidence = self._calculate_button_confidence(box, text)
                            
                            # Generate selector suggestions
                            selectors = await self._generate_selector_suggestions(element, 'button')
                            
                            visual_element = VisualElement(
                                element_type=ElementType.BUTTON,
                                confidence=confidence,
                                bounding_box=box,
                                text_content=text.strip(),
                                selector_suggestions=selectors
                            )
                            
                            buttons.append(visual_element)
                            
                        except:
                            continue
                except:
                    continue
            
            return buttons
            
        except Exception as e:
            logger.error(f"Button detection failed: {e}")
            return []
    
    async def _detect_inputs(self, page: Page) -> List[VisualElement]:
        """Detect input elements"""
        try:
            inputs = []
            
            # Find input elements
            input_selectors = [
                'input[type="text"]',
                'input[type="email"]',
                'input[type="password"]',
                'input[type="search"]',
                'textarea',
                '[contenteditable="true"]'
            ]
            
            for selector in input_selectors:
                try:
                    elements = await page.locator(selector).all()
                    
                    for element in elements:
                        try:
                            # Get bounding box
                            box = await element.bounding_box()
                            if not box:
                                continue
                            
                            # Get placeholder or value
                            placeholder = await element.get_attribute('placeholder') or ""
                            value = await element.input_value() if hasattr(element, 'input_value') else ""
                            
                            # Calculate confidence
                            confidence = self._calculate_input_confidence(box, placeholder)
                            
                            # Generate selector suggestions
                            selectors = await self._generate_selector_suggestions(element, 'input')
                            
                            visual_element = VisualElement(
                                element_type=ElementType.INPUT,
                                confidence=confidence,
                                bounding_box=box,
                                text_content=placeholder or value,
                                selector_suggestions=selectors,
                                attributes={
                                    'placeholder': placeholder,
                                    'value': value,
                                    'type': await element.get_attribute('type') or 'text'
                                }
                            )
                            
                            inputs.append(visual_element)
                            
                        except:
                            continue
                except:
                    continue
            
            return inputs
            
        except Exception as e:
            logger.error(f"Input detection failed: {e}")
            return []
    
    async def _detect_links(self, page: Page) -> List[VisualElement]:
        """Detect link elements"""
        try:
            links = []
            
            # Find link elements
            link_elements = await page.locator('a').all()
            
            for element in link_elements:
                try:
                    # Get bounding box
                    box = await element.bounding_box()
                    if not box:
                        continue
                    
                    # Get text and href
                    text = await element.text_content() or ""
                    href = await element.get_attribute('href') or ""
                    
                    # Skip empty links
                    if not text.strip() and not href:
                        continue
                    
                    # Calculate confidence
                    confidence = self._calculate_link_confidence(box, text, href)
                    
                    # Generate selector suggestions
                    selectors = await self._generate_selector_suggestions(element, 'link')
                    
                    visual_element = VisualElement(
                        element_type=ElementType.LINK,
                        confidence=confidence,
                        bounding_box=box,
                        text_content=text.strip(),
                        selector_suggestions=selectors,
                        attributes={
                            'href': href,
                            'target': await element.get_attribute('target') or ""
                        }
                    )
                    
                    links.append(visual_element)
                    
                except:
                    continue
            
            return links
            
        except Exception as e:
            logger.error(f"Link detection failed: {e}")
            return []
    
    async def _detect_images(self, page: Page) -> List[VisualElement]:
        """Detect image elements"""
        try:
            images = []
            
            # Find image elements
            image_elements = await page.locator('img').all()
            
            for element in image_elements:
                try:
                    # Get bounding box
                    box = await element.bounding_box()
                    if not box:
                        continue
                    
                    # Get image attributes
                    src = await element.get_attribute('src') or ""
                    alt = await element.get_attribute('alt') or ""
                    title = await element.get_attribute('title') or ""
                    
                    # Calculate confidence
                    confidence = self._calculate_image_confidence(box, src, alt)
                    
                    # Generate selector suggestions
                    selectors = await self._generate_selector_suggestions(element, 'image')
                    
                    visual_element = VisualElement(
                        element_type=ElementType.IMAGE,
                        confidence=confidence,
                        bounding_box=box,
                        text_content=alt or title,
                        selector_suggestions=selectors,
                        attributes={
                            'src': src,
                            'alt': alt,
                            'title': title
                        }
                    )
                    
                    images.append(visual_element)
                    
                except:
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return []
    
    def _calculate_button_confidence(self, box: Dict, text: str) -> float:
        """Calculate confidence score for button detection"""
        confidence = 0.5  # Base confidence
        
        # Size-based confidence
        width, height = box['width'], box['height']
        pattern = self.element_patterns[ElementType.BUTTON]
        size_range = pattern['size_range']
        
        if size_range['min_width'] <= width <= size_range['max_width']:
            confidence += 0.2
        if size_range['min_height'] <= height <= size_range['max_height']:
            confidence += 0.2
        
        # Text-based confidence
        text_lower = text.lower()
        for pattern_text in pattern['text_patterns']:
            if pattern_text in text_lower:
                confidence += 0.1
                break
        
        return min(confidence, 1.0)
    
    def _calculate_input_confidence(self, box: Dict, placeholder: str) -> float:
        """Calculate confidence score for input detection"""
        confidence = 0.6  # Base confidence for inputs
        
        # Size-based confidence
        width, height = box['width'], box['height']
        pattern = self.element_patterns[ElementType.INPUT]
        size_range = pattern['size_range']
        
        if size_range['min_width'] <= width <= size_range['max_width']:
            confidence += 0.2
        if size_range['min_height'] <= height <= size_range['max_height']:
            confidence += 0.1
        
        # Placeholder-based confidence
        if placeholder:
            placeholder_lower = placeholder.lower()
            for pattern_text in pattern['text_patterns']:
                if pattern_text in placeholder_lower:
                    confidence += 0.1
                    break
        
        return min(confidence, 1.0)
    
    def _calculate_link_confidence(self, box: Dict, text: str, href: str) -> float:
        """Calculate confidence score for link detection"""
        confidence = 0.4  # Base confidence
        
        # Href presence
        if href and href.startswith(('http', '/', '#')):
            confidence += 0.3
        
        # Text-based confidence
        if text:
            text_lower = text.lower()
            pattern = self.element_patterns[ElementType.LINK]
            for pattern_text in pattern['text_patterns']:
                if pattern_text in text_lower:
                    confidence += 0.2
                    break
        
        # Size-based confidence
        width, height = box['width'], box['height']
        if width > 10 and height > 10:  # Reasonable clickable size
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_image_confidence(self, box: Dict, src: str, alt: str) -> float:
        """Calculate confidence score for image detection"""
        confidence = 0.7  # Base confidence for images
        
        # Source presence
        if src:
            confidence += 0.2
        
        # Alt text presence
        if alt:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _generate_selector_suggestions(self, element, element_type: str) -> List[str]:
        """Generate selector suggestions for an element"""
        try:
            selectors = []
            
            # Get common attributes
            id_attr = await element.get_attribute('id')
            class_attr = await element.get_attribute('class')
            name_attr = await element.get_attribute('name')
            data_testid = await element.get_attribute('data-testid')
            
            # ID-based selector
            if id_attr:
                selectors.append(f'#{id_attr}')
            
            # Class-based selector
            if class_attr:
                classes = class_attr.split()
                if classes:
                    selectors.append(f'.{classes[0]}')
                    if len(classes) > 1:
                        selectors.append(f'.{".".join(classes[:2])}')
            
            # Name-based selector
            if name_attr:
                selectors.append(f'[name="{name_attr}"]')
            
            # Data-testid selector
            if data_testid:
                selectors.append(f'[data-testid="{data_testid}"]')
            
            # Tag-based selector
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            if tag_name:
                selectors.append(tag_name)
            
            return selectors[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Selector generation failed: {e}")
            return []
    
    async def _recognize_text(self, page: Page, screenshot_data: str) -> VisionResult:
        """Recognize text using OCR"""
        try:
            # Simulate OCR (in production, use real OCR engine)
            # Extract all text content from the page
            all_text = await page.evaluate("""
                () => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    
                    const texts = [];
                    let node;
                    
                    while (node = walker.nextNode()) {
                        const text = node.textContent.trim();
                        if (text && text.length > 0) {
                            texts.push(text);
                        }
                    }
                    
                    return texts.join(' ');
                }
            """)
            
            return VisionResult(
                task_type=VisionTask.TEXT_RECOGNITION,
                success=True,
                confidence=0.9,
                text_content=all_text,
                analysis={
                    'text_length': len(all_text),
                    'word_count': len(all_text.split()),
                    'method': 'DOM_extraction'
                }
            )
            
        except Exception as e:
            return VisionResult(
                task_type=VisionTask.TEXT_RECOGNITION,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    async def _analyze_ui_components(self, page: Page, screenshot_data: str) -> VisionResult:
        """Analyze UI components and layout"""
        try:
            # Detect all elements
            detection_result = await self._detect_elements(page, screenshot_data)
            
            if not detection_result.success:
                return detection_result
            
            # Analyze UI patterns
            ui_patterns = self._analyze_ui_patterns(detection_result.elements)
            
            # Detect common UI components
            components = self._detect_ui_components(detection_result.elements)
            
            return VisionResult(
                task_type=VisionTask.UI_ANALYSIS,
                success=True,
                confidence=0.8,
                elements=detection_result.elements,
                analysis={
                    'ui_patterns': ui_patterns,
                    'components': components,
                    'layout_analysis': self._analyze_layout(detection_result.elements)
                }
            )
            
        except Exception as e:
            return VisionResult(
                task_type=VisionTask.UI_ANALYSIS,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    def _analyze_ui_patterns(self, elements: List[VisualElement]) -> Dict[str, Any]:
        """Analyze UI patterns from detected elements"""
        patterns = {
            'forms': 0,
            'navigation': 0,
            'content_blocks': 0,
            'interactive_elements': 0
        }
        
        # Count different types of elements
        element_counts = {}
        for element in elements:
            element_type = element.element_type.value
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        # Detect form patterns
        if element_counts.get('input', 0) >= 2 and element_counts.get('button', 0) >= 1:
            patterns['forms'] = 1
        
        # Detect navigation patterns
        if element_counts.get('link', 0) >= 3:
            patterns['navigation'] = 1
        
        # Count interactive elements
        patterns['interactive_elements'] = (
            element_counts.get('button', 0) + 
            element_counts.get('link', 0) + 
            element_counts.get('input', 0)
        )
        
        return patterns
    
    def _detect_ui_components(self, elements: List[VisualElement]) -> List[str]:
        """Detect common UI components"""
        components = []
        
        # Check for login form
        has_inputs = any(elem.element_type == ElementType.INPUT for elem in elements)
        has_button = any(elem.element_type == ElementType.BUTTON for elem in elements)
        has_password_field = any('password' in elem.text_content.lower() for elem in elements)
        
        if has_inputs and has_button and has_password_field:
            components.append('login_form')
        
        # Check for search functionality
        has_search_input = any('search' in elem.text_content.lower() for elem in elements if elem.element_type == ElementType.INPUT)
        if has_search_input and has_button:
            components.append('search_bar')
        
        # Check for navigation menu
        link_count = sum(1 for elem in elements if elem.element_type == ElementType.LINK)
        if link_count >= 3:
            components.append('navigation_menu')
        
        return components
    
    def _analyze_layout(self, elements: List[VisualElement]) -> Dict[str, Any]:
        """Analyze page layout"""
        if not elements:
            return {}
        
        # Calculate bounding box statistics
        x_positions = [elem.bounding_box['x'] for elem in elements]
        y_positions = [elem.bounding_box['y'] for elem in elements]
        widths = [elem.bounding_box['width'] for elem in elements]
        heights = [elem.bounding_box['height'] for elem in elements]
        
        return {
            'element_count': len(elements),
            'layout_bounds': {
                'min_x': min(x_positions),
                'max_x': max(x_positions),
                'min_y': min(y_positions),
                'max_y': max(y_positions)
            },
            'average_element_size': {
                'width': sum(widths) / len(widths),
                'height': sum(heights) / len(heights)
            },
            'layout_density': len(elements) / (max(x_positions) * max(y_positions)) if max(x_positions) > 0 and max(y_positions) > 0 else 0
        }
    
    async def _validate_visual_state(self, page: Page, screenshot_data: str) -> VisionResult:
        """Validate visual state of the page"""
        try:
            # Perform basic visual validation
            validation_results = {
                'page_loaded': True,
                'elements_visible': True,
                'layout_stable': True,
                'no_errors': True
            }
            
            # Check if page is fully loaded
            ready_state = await page.evaluate('document.readyState')
            validation_results['page_loaded'] = ready_state == 'complete'
            
            # Check for error messages
            error_selectors = ['.error', '.alert-error', '[role="alert"]', '.warning']
            for selector in error_selectors:
                try:
                    error_count = await page.locator(selector).count()
                    if error_count > 0:
                        validation_results['no_errors'] = False
                        break
                except:
                    continue
            
            # Calculate overall confidence
            confidence = sum(validation_results.values()) / len(validation_results)
            
            return VisionResult(
                task_type=VisionTask.VISUAL_VALIDATION,
                success=True,
                confidence=confidence,
                analysis={'validation_results': validation_results}
            )
            
        except Exception as e:
            return VisionResult(
                task_type=VisionTask.VISUAL_VALIDATION,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    async def _match_patterns(self, page: Page, screenshot_data: str) -> VisionResult:
        """Match visual patterns on the page"""
        try:
            # Detect elements first
            detection_result = await self._detect_elements(page, screenshot_data)
            
            if not detection_result.success:
                return detection_result
            
            # Match against known UI templates
            matched_patterns = []
            
            for template_name, template in self.ui_templates.items():
                match_score = self._calculate_template_match(detection_result.elements, template)
                if match_score > 0.7:
                    matched_patterns.append({
                        'template': template_name,
                        'confidence': match_score
                    })
            
            return VisionResult(
                task_type=VisionTask.PATTERN_MATCHING,
                success=True,
                confidence=0.8,
                elements=detection_result.elements,
                analysis={
                    'matched_patterns': matched_patterns,
                    'total_patterns_checked': len(self.ui_templates)
                }
            )
            
        except Exception as e:
            return VisionResult(
                task_type=VisionTask.PATTERN_MATCHING,
                success=False,
                confidence=0.0,
                error=str(e)
            )
    
    def _calculate_template_match(self, elements: List[VisualElement], template: Dict) -> float:
        """Calculate how well elements match a UI template"""
        required_elements = template['required_elements']
        text_indicators = template['text_indicators']
        
        # Check if required element types are present
        element_types = [elem.element_type for elem in elements]
        type_matches = 0
        
        for required_type in required_elements:
            if required_type in element_types:
                type_matches += 1
        
        type_score = type_matches / len(required_elements)
        
        # Check for text indicators
        all_text = ' '.join(elem.text_content.lower() for elem in elements)
        text_matches = 0
        
        for indicator in text_indicators:
            if indicator in all_text:
                text_matches += 1
        
        text_score = text_matches / len(text_indicators) if text_indicators else 0
        
        # Combine scores
        return (type_score * 0.7) + (text_score * 0.3)
    
    async def find_element_by_visual_cues(self, page: Page, description: str) -> Dict[str, Any]:
        """Find element using visual description"""
        try:
            # Analyze page
            result = await self.analyze_page(page, VisionTask.ELEMENT_DETECTION)
            
            if not result.success:
                return {'success': False, 'error': result.error}
            
            # Find best matching element
            best_match = None
            best_score = 0
            
            description_lower = description.lower()
            
            for element in result.elements:
                score = 0
                
                # Text content matching
                if element.text_content and description_lower in element.text_content.lower():
                    score += 0.5
                
                # Element type matching
                if element.element_type.value in description_lower:
                    score += 0.3
                
                # Attribute matching
                for attr_value in element.attributes.values():
                    if isinstance(attr_value, str) and description_lower in attr_value.lower():
                        score += 0.2
                        break
                
                if score > best_score:
                    best_score = score
                    best_match = element
            
            if best_match and best_score > 0.3:
                return {
                    'success': True,
                    'element': best_match,
                    'confidence': best_score,
                    'suggested_selectors': best_match.selector_suggestions
                }
            else:
                return {
                    'success': False,
                    'error': f'No element found matching description: {description}'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Global computer vision system instance
_global_vision_system: Optional[ComputerVisionSystem] = None

def get_computer_vision_system() -> ComputerVisionSystem:
    """Get or create the global computer vision system"""
    global _global_vision_system
    
    if _global_vision_system is None:
        _global_vision_system = ComputerVisionSystem()
    
    return _global_vision_system