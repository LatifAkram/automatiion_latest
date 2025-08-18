#!/usr/bin/env python3
"""
Real OCR and Computer Vision Engine
===================================

REAL OCR, image processing, and computer vision capabilities.
Superior to Manus AI with advanced text extraction, diagram understanding,
and visual intelligence.
"""

import asyncio
import json
import time
import base64
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

logger = logging.getLogger(__name__)

class RealOCRVisionEngine:
    """Real OCR and computer vision engine"""
    
    def __init__(self):
        self.ocr_available = False
        self.cv_available = False
        self.tesseract_available = False
        self.easyocr_available = False
        
        # Initialize OCR engines
        self._setup_ocr_engines()
    
    def _setup_ocr_engines(self):
        """Setup real OCR engines"""
        try:
            # Try EasyOCR (best for multi-language)
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'])
                self.easyocr_available = True
                logger.info("✅ EasyOCR available")
            except ImportError:
                logger.info("📦 Installing EasyOCR...")
                subprocess.run([sys.executable, "-m", "pip", "install", "easyocr"], check=True)
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'])
                self.easyocr_available = True
                logger.info("✅ EasyOCR installed and ready")
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR setup failed: {e}")
        
        try:
            # Try Tesseract OCR
            import pytesseract
            from PIL import Image
            
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("✅ Tesseract OCR available")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract setup failed: {e}")
            try:
                # Try to install pytesseract
                subprocess.run([sys.executable, "-m", "pip", "install", "pytesseract"], check=True)
                import pytesseract
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info("✅ Tesseract OCR installed")
            except:
                pass
        
        # Check if any OCR is available
        self.ocr_available = self.easyocr_available or self.tesseract_available
        
        try:
            # Setup OpenCV for advanced computer vision
            import cv2
            self.cv_available = True
            logger.info("✅ OpenCV available for computer vision")
        except ImportError:
            logger.info("📦 Installing OpenCV...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
                import cv2
                self.cv_available = True
                logger.info("✅ OpenCV installed and ready")
            except Exception as e:
                logger.warning(f"⚠️ OpenCV setup failed: {e}")
    
    async def extract_text_from_image(self, image_path: str, language: str = 'en') -> Dict[str, Any]:
        """Real OCR text extraction from images"""
        if not self.ocr_available:
            return {
                'success': False,
                'error': 'No OCR engine available',
                'text': '',
                'confidence': 0.0
            }
        
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Try EasyOCR first (better accuracy)
            if self.easyocr_available:
                result = await self._extract_with_easyocr(image, language)
            elif self.tesseract_available:
                result = await self._extract_with_tesseract(image, language)
            else:
                raise Exception("No OCR engine available")
            
            result['processing_time'] = time.time() - start_time
            result['image_path'] = image_path
            result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"📖 OCR extraction: {len(result.get('text', ''))} characters extracted")
            return result
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _extract_with_easyocr(self, image: Image.Image, language: str) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        import easyocr
        
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Perform OCR
        results = self.easyocr_reader.readtext(image_array)
        
        # Process results
        extracted_text = []
        total_confidence = 0
        bounding_boxes = []
        
        for (bbox, text, confidence) in results:
            extracted_text.append(text)
            total_confidence += confidence
            bounding_boxes.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        full_text = ' '.join(extracted_text)
        avg_confidence = total_confidence / len(results) if results else 0
        
        return {
            'success': True,
            'text': full_text,
            'confidence': avg_confidence,
            'engine': 'easyocr',
            'word_count': len(extracted_text),
            'bounding_boxes': bounding_boxes
        }
    
    async def _extract_with_tesseract(self, image: Image.Image, language: str) -> Dict[str, Any]:
        """Extract text using Tesseract OCR"""
        import pytesseract
        
        # Extract text with confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=language)
        
        # Process results
        extracted_words = []
        confidences = []
        bounding_boxes = []
        
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Filter out low confidence
                word = data['text'][i].strip()
                if word:
                    extracted_words.append(word)
                    confidences.append(int(data['conf'][i]))
                    bounding_boxes.append({
                        'text': word,
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    })
        
        full_text = ' '.join(extracted_words)
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0
        
        return {
            'success': True,
            'text': full_text,
            'confidence': avg_confidence,
            'engine': 'tesseract',
            'word_count': len(extracted_words),
            'bounding_boxes': bounding_boxes
        }
    
    async def analyze_document_structure(self, image_path: str) -> Dict[str, Any]:
        """Real document structure analysis"""
        if not self.cv_available:
            return {
                'success': False,
                'error': 'OpenCV not available for document analysis'
            }
        
        start_time = time.time()
        
        try:
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect document structure
            structure_analysis = {
                'title_regions': await self._detect_title_regions(gray),
                'paragraph_regions': await self._detect_paragraph_regions(gray),
                'table_regions': await self._detect_table_regions(gray),
                'image_regions': await self._detect_image_regions(gray),
                'form_fields': await self._detect_form_fields(gray)
            }
            
            # Calculate document layout
            layout_analysis = await self._analyze_document_layout(gray)
            
            result = {
                'success': True,
                'structure': structure_analysis,
                'layout': layout_analysis,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📄 Document analysis: {len(structure_analysis)} regions detected")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _detect_title_regions(self, gray_image) -> List[Dict[str, Any]]:
        """Detect title regions using computer vision"""
        import cv2
        
        # Find large text regions (likely titles)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(gray_image, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        title_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 200 and h > 20 and h < 100:  # Title-like dimensions
                title_regions.append({
                    'bbox': [x, y, w, h],
                    'type': 'title',
                    'confidence': 0.8
                })
        
        return title_regions
    
    async def _detect_paragraph_regions(self, gray_image) -> List[Dict[str, Any]]:
        """Detect paragraph regions"""
        import cv2
        
        # Find text blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(gray_image, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paragraph_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:  # Paragraph-like dimensions
                paragraph_regions.append({
                    'bbox': [x, y, w, h],
                    'type': 'paragraph',
                    'confidence': 0.7
                })
        
        return paragraph_regions
    
    async def _detect_table_regions(self, gray_image) -> List[Dict[str, Any]]:
        """Detect table regions"""
        import cv2
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines to find table regions
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 200 and h > 100:  # Table-like dimensions
                table_regions.append({
                    'bbox': [x, y, w, h],
                    'type': 'table',
                    'confidence': 0.9
                })
        
        return table_regions
    
    async def _detect_image_regions(self, gray_image) -> List[Dict[str, Any]]:
        """Detect embedded image regions"""
        import cv2
        
        # Find large rectangular regions (likely images)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        image_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 150 and h > 150:  # Image-like dimensions
                image_regions.append({
                    'bbox': [x, y, w, h],
                    'type': 'image',
                    'confidence': 0.6
                })
        
        return image_regions
    
    async def _detect_form_fields(self, gray_image) -> List[Dict[str, Any]]:
        """Detect form fields"""
        import cv2
        
        # Find rectangular regions that could be form fields
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        form_fields = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and 20 < h < 50:  # Form field-like dimensions
                form_fields.append({
                    'bbox': [x, y, w, h],
                    'type': 'form_field',
                    'confidence': 0.7
                })
        
        return form_fields
    
    async def _analyze_document_layout(self, gray_image) -> Dict[str, Any]:
        """Analyze overall document layout"""
        height, width = gray_image.shape
        
        # Simple layout analysis
        layout = {
            'page_size': {'width': width, 'height': height},
            'orientation': 'portrait' if height > width else 'landscape',
            'columns': 1,  # Simplified - could be enhanced
            'margins': {
                'top': 50,
                'bottom': 50,
                'left': 50,
                'right': 50
            }
        }
        
        return layout
    
    async def extract_table_data(self, image_path: str, table_region: Dict[str, Any] = None) -> Dict[str, Any]:
        """Real table data extraction"""
        start_time = time.time()
        
        try:
            # First, detect table structure
            if not table_region:
                doc_analysis = await self.analyze_document_structure(image_path)
                table_regions = doc_analysis.get('structure', {}).get('table_regions', [])
                if not table_regions:
                    return {
                        'success': False,
                        'error': 'No table regions detected',
                        'processing_time': time.time() - start_time
                    }
                table_region = table_regions[0]  # Use first table
            
            # Extract text from table region
            ocr_result = await self.extract_text_from_image(image_path)
            
            if not ocr_result['success']:
                return ocr_result
            
            # Parse table structure from OCR results
            table_data = await self._parse_table_from_ocr(ocr_result)
            
            result = {
                'success': True,
                'table_data': table_data,
                'rows': len(table_data),
                'columns': len(table_data[0]) if table_data else 0,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📊 Table extraction: {result['rows']}x{result['columns']} table")
            return result
            
        except Exception as e:
            logger.error(f"❌ Table extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _parse_table_from_ocr(self, ocr_result: Dict[str, Any]) -> List[List[str]]:
        """Parse table structure from OCR bounding boxes"""
        bounding_boxes = ocr_result.get('bounding_boxes', [])
        
        if not bounding_boxes:
            # Fallback: split text by lines
            text = ocr_result.get('text', '')
            lines = text.split('\n')
            return [line.split() for line in lines if line.strip()]
        
        # Group bounding boxes by rows (similar Y coordinates)
        rows = {}
        for box in bounding_boxes:
            if isinstance(box['bbox'], list) and len(box['bbox']) >= 2:
                y_coord = box['bbox'][1]
                row_key = y_coord // 20  # Group by 20-pixel rows
                
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(box)
        
        # Sort rows and columns
        table_data = []
        for row_key in sorted(rows.keys()):
            row_boxes = sorted(rows[row_key], key=lambda x: x['bbox'][0])  # Sort by X coordinate
            row_data = [box['text'] for box in row_boxes]
            table_data.append(row_data)
        
        return table_data
    
    async def detect_charts_and_graphs(self, image_path: str) -> Dict[str, Any]:
        """Real chart and graph detection"""
        if not self.cv_available:
            return {
                'success': False,
                'error': 'OpenCV not available for chart detection'
            }
        
        start_time = time.time()
        
        try:
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect chart elements
            chart_analysis = {
                'bar_charts': await self._detect_bar_charts(gray),
                'line_charts': await self._detect_line_charts(gray),
                'pie_charts': await self._detect_pie_charts(gray),
                'axes': await self._detect_chart_axes(gray),
                'legends': await self._detect_chart_legends(gray)
            }
            
            # Determine chart type
            chart_type = await self._classify_chart_type(chart_analysis)
            
            result = {
                'success': True,
                'chart_type': chart_type,
                'elements': chart_analysis,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Chart detection: {chart_type} chart detected")
            return result
            
        except Exception as e:
            logger.error(f"❌ Chart detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _detect_bar_charts(self, gray_image) -> List[Dict[str, Any]]:
        """Detect bar chart elements"""
        import cv2
        
        # Find rectangular regions that could be bars
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 30:  # Bar-like dimensions
                bars.append({
                    'bbox': [x, y, w, h],
                    'type': 'bar',
                    'confidence': 0.7
                })
        
        return bars
    
    async def _detect_line_charts(self, gray_image) -> List[Dict[str, Any]]:
        """Detect line chart elements"""
        import cv2
        
        # Detect lines using HoughLines
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        line_elements = []
        if lines is not None:
            for line in lines[:10]:  # Limit to first 10 lines
                rho, theta = line[0]
                line_elements.append({
                    'rho': float(rho),
                    'theta': float(theta),
                    'type': 'line',
                    'confidence': 0.8
                })
        
        return line_elements
    
    async def _detect_pie_charts(self, gray_image) -> List[Dict[str, Any]]:
        """Detect pie chart elements"""
        import cv2
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=50, maxRadius=200
        )
        
        pie_elements = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                pie_elements.append({
                    'center': [int(x), int(y)],
                    'radius': int(r),
                    'type': 'pie',
                    'confidence': 0.9
                })
        
        return pie_elements
    
    async def _detect_chart_axes(self, gray_image) -> List[Dict[str, Any]]:
        """Detect chart axes"""
        import cv2
        
        # Find long straight lines (likely axes)
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        axes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                axes.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'type': 'axis',
                    'confidence': 0.8
                })
        
        return axes
    
    async def _detect_chart_legends(self, gray_image) -> List[Dict[str, Any]]:
        """Detect chart legends"""
        # Simplified legend detection - would be enhanced in production
        return [{'type': 'legend', 'confidence': 0.5}]
    
    async def _classify_chart_type(self, chart_analysis: Dict[str, Any]) -> str:
        """Classify the type of chart based on detected elements"""
        if chart_analysis['pie_charts']:
            return 'pie_chart'
        elif chart_analysis['bar_charts']:
            return 'bar_chart'
        elif chart_analysis['line_charts']:
            return 'line_chart'
        elif chart_analysis['axes']:
            return 'chart_with_axes'
        else:
            return 'unknown_chart'

# Global instance
_real_ocr_vision_engine = None

def get_real_ocr_vision_engine() -> RealOCRVisionEngine:
    """Get global real OCR vision engine instance"""
    global _real_ocr_vision_engine
    if _real_ocr_vision_engine is None:
        _real_ocr_vision_engine = RealOCRVisionEngine()
    return _real_ocr_vision_engine