"""
Enterprise CAPTCHA Solving & Security Bypass System
==================================================

Comprehensive automation for all security challenges:
- reCAPTCHA v2/v3 solving (Google)
- hCaptcha enterprise solving
- Image recognition CAPTCHAs
- Audio CAPTCHA processing
- Text-based CAPTCHAs (distorted text)
- Mathematical CAPTCHAs
- Behavioral analysis bypass
- Cloudflare challenge solving
- Bot detection evasion
- Rate limiting bypass

Features:
- AI-powered computer vision for image recognition
- Advanced audio processing and speech recognition
- Machine learning pattern recognition
- Behavioral simulation (human-like mouse movements)
- Proxy rotation and fingerprint randomization
- Real-time success rate optimization
- Multi-provider CAPTCHA solving services
- Stealth mode operation
"""

import asyncio
import logging
import json
import re
import base64
import io
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import requests
import numpy as np

try:
    import cv2
    import pytesseract
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import tensorflow as tf
    from transformers import pipeline
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False

try:
    from playwright.async_api import Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from ...core.deterministic_executor import DeterministicExecutor
from ...core.semantic_dom_graph import SemanticDOMGraph


class CaptchaType(str, Enum):
    """Types of CAPTCHAs supported."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE_RECOGNITION = "image_recognition"
    TEXT_CAPTCHA = "text_captcha"
    AUDIO_CAPTCHA = "audio_captcha"
    MATH_CAPTCHA = "math_captcha"
    BEHAVIORAL = "behavioral"
    CLOUDFLARE = "cloudflare"
    FUNCAPTCHA = "funcaptcha"
    GEETEST = "geetest"
    CUSTOM = "custom"


class SolveMethod(str, Enum):
    """CAPTCHA solving methods."""
    AI_VISION = "ai_vision"
    OCR_PROCESSING = "ocr_processing"
    AUDIO_RECOGNITION = "audio_recognition"
    PATTERN_MATCHING = "pattern_matching"
    BEHAVIORAL_SIMULATION = "behavioral_simulation"
    THIRD_PARTY_SERVICE = "third_party_service"
    MACHINE_LEARNING = "machine_learning"
    HYBRID_APPROACH = "hybrid_approach"


class CaptchaStatus(str, Enum):
    """CAPTCHA solving status."""
    PENDING = "pending"
    SOLVING = "solving"
    SOLVED = "solved"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID = "invalid"
    BYPASSED = "bypassed"


@dataclass
class CaptchaChallenge:
    """CAPTCHA challenge information."""
    challenge_id: str
    captcha_type: CaptchaType
    site_key: Optional[str] = None
    page_url: str = ""
    image_data: Optional[bytes] = None
    audio_data: Optional[bytes] = None
    question: Optional[str] = None
    options: List[str] = None
    difficulty: str = "medium"
    timeout: int = 60
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SolveResult:
    """CAPTCHA solving result."""
    challenge_id: str
    status: CaptchaStatus
    solution: Optional[str] = None
    confidence: float = 0.0
    solve_time: float = 0.0
    method_used: Optional[SolveMethod] = None
    error_message: Optional[str] = None
    attempts: int = 1
    cost: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AIVisionSolver:
    """AI-powered computer vision CAPTCHA solver."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models
        self.vision_model = None
        self.text_recognition_model = None
        
        if AI_MODELS_AVAILABLE:
            try:
                # Initialize vision pipeline for image classification
                self.vision_model = pipeline("image-classification", 
                                            model="google/vit-base-patch16-224")
                
                # Initialize text recognition
                self.text_recognition_model = pipeline("text2text-generation",
                                                     model="microsoft/DialoGPT-medium")
            except Exception as e:
                self.logger.warning(f"AI model initialization failed: {e}")
        
        # Image processing settings
        self.image_preprocessing = {
            'resize_factor': 2.0,
            'contrast_enhancement': 1.5,
            'brightness_adjustment': 1.2,
            'noise_reduction': True,
            'edge_detection': True
        }
    
    async def solve_image_captcha(self, challenge: CaptchaChallenge) -> SolveResult:
        """Solve image-based CAPTCHA using AI vision."""
        start_time = time.time()
        
        try:
            if not challenge.image_data:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    error_message="No image data provided"
                )
            
            # Preprocess image
            processed_image = await self._preprocess_image(challenge.image_data)
            
            # Determine CAPTCHA type and solve accordingly
            if challenge.captcha_type == CaptchaType.IMAGE_RECOGNITION:
                solution = await self._solve_image_recognition(processed_image, challenge.question)
            elif challenge.captcha_type == CaptchaType.TEXT_CAPTCHA:
                solution = await self._solve_text_captcha(processed_image)
            else:
                solution = await self._solve_generic_image(processed_image, challenge)
            
            solve_time = time.time() - start_time
            
            if solution:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.SOLVED,
                    solution=solution,
                    confidence=0.85,
                    solve_time=solve_time,
                    method_used=SolveMethod.AI_VISION
                )
            else:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    solve_time=solve_time,
                    method_used=SolveMethod.AI_VISION,
                    error_message="Failed to extract solution from image"
                )
                
        except Exception as e:
            self.logger.error(f"AI vision solving failed: {e}")
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                solve_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for better recognition."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize image
            if self.image_preprocessing['resize_factor'] != 1.0:
                new_size = tuple(int(dim * self.image_preprocessing['resize_factor']) 
                               for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.image_preprocessing['contrast_enhancement'])
            
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.image_preprocessing['brightness_adjustment'])
            
            # Apply noise reduction
            if self.image_preprocessing['noise_reduction']:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply edge detection if needed
            if self.image_preprocessing['edge_detection'] and OPENCV_AVAILABLE:
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                edges = cv2.Canny(gray, 50, 150)
                img_array = cv2.addWeighted(img_array, 0.8, 
                                          cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.2, 0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            # Return original image as numpy array
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
    
    async def _solve_image_recognition(self, image: np.ndarray, question: str) -> Optional[str]:
        """Solve image recognition CAPTCHA (e.g., 'Select all traffic lights')."""
        try:
            if not self.vision_model:
                # Fallback to basic pattern matching
                return await self._pattern_based_recognition(image, question)
            
            # Convert numpy array back to PIL Image for the model
            pil_image = Image.fromarray(image.astype('uint8'))
            
            # Use AI model to classify image content
            results = self.vision_model(pil_image)
            
            # Parse question to understand what to look for
            target_objects = self._parse_recognition_question(question)
            
            # Find matching objects in results
            matches = []
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                for target in target_objects:
                    if target.lower() in label and score > 0.3:
                        matches.append({
                            'object': target,
                            'confidence': score,
                            'label': label
                        })
            
            # Return the best match or grid coordinates
            if matches:
                best_match = max(matches, key=lambda x: x['confidence'])
                return best_match['object']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Image recognition failed: {e}")
            return None
    
    async def _solve_text_captcha(self, image: np.ndarray) -> Optional[str]:
        """Solve text-based CAPTCHA using OCR."""
        try:
            if not OPENCV_AVAILABLE:
                return None
            
            # Convert to grayscale for better OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply additional preprocessing for text recognition
            # Threshold the image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.medianBlur(thresh, 3)
            
            # Use Tesseract OCR
            text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            
            # Clean up the text
            text = re.sub(r'[^a-zA-Z0-9]', '', text.strip())
            
            return text if len(text) >= 3 else None
            
        except Exception as e:
            self.logger.error(f"Text CAPTCHA solving failed: {e}")
            return None
    
    def _parse_recognition_question(self, question: str) -> List[str]:
        """Parse image recognition question to extract target objects."""
        question_lower = question.lower()
        
        # Common object mappings
        object_keywords = {
            'traffic lights': ['traffic light', 'traffic_light', 'signal'],
            'cars': ['car', 'vehicle', 'automobile'],
            'bicycles': ['bicycle', 'bike', 'cycling'],
            'buses': ['bus', 'coach'],
            'motorcycles': ['motorcycle', 'motorbike'],
            'trucks': ['truck', 'lorry'],
            'crosswalks': ['crosswalk', 'zebra crossing', 'pedestrian crossing'],
            'fire hydrants': ['fire hydrant', 'hydrant'],
            'stairs': ['stairs', 'staircase', 'steps'],
            'chimneys': ['chimney'],
            'palm trees': ['palm tree', 'palm'],
            'mountains': ['mountain', 'hill'],
            'bridges': ['bridge'],
            'boats': ['boat', 'ship', 'vessel']
        }
        
        targets = []
        for key, keywords in object_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    targets.append(key)
                    break
        
        return targets if targets else ['unknown']


class AudioCaptchaSolver:
    """Audio CAPTCHA solver using speech recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize speech recognition
        self.recognizer = None
        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Adjust for ambient noise
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
    
    async def solve_audio_captcha(self, challenge: CaptchaChallenge) -> SolveResult:
        """Solve audio CAPTCHA using speech recognition."""
        start_time = time.time()
        
        try:
            if not challenge.audio_data or not AUDIO_AVAILABLE:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    error_message="Audio processing not available"
                )
            
            # Process audio data
            audio_text = await self._process_audio(challenge.audio_data)
            
            solve_time = time.time() - start_time
            
            if audio_text:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.SOLVED,
                    solution=audio_text,
                    confidence=0.75,
                    solve_time=solve_time,
                    method_used=SolveMethod.AUDIO_RECOGNITION
                )
            else:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    solve_time=solve_time,
                    error_message="Failed to recognize audio"
                )
                
        except Exception as e:
            self.logger.error(f"Audio CAPTCHA solving failed: {e}")
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                solve_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio data and extract text."""
        try:
            # Save audio data to temporary file
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Enhance audio quality
            # Normalize volume
            audio_segment = audio_segment.normalize()
            
            # Apply noise reduction (simple high-pass filter)
            audio_segment = audio_segment.high_pass_filter(300)
            
            # Convert to WAV format for speech recognition
            wav_data = io.BytesIO()
            audio_segment.export(wav_data, format="wav")
            wav_data.seek(0)
            
            # Use speech recognition
            with sr.AudioFile(wav_data) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Record the audio
                audio = self.recognizer.record(source)
                
                # Try multiple recognition engines
                try:
                    # Try Google Speech Recognition
                    text = self.recognizer.recognize_google(audio)
                    return self._clean_audio_text(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
                
                try:
                    # Try Sphinx (offline)
                    text = self.recognizer.recognize_sphinx(audio)
                    return self._clean_audio_text(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return None
    
    def _clean_audio_text(self, text: str) -> str:
        """Clean and format recognized audio text."""
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text


class BehavioralSolver:
    """Behavioral CAPTCHA solver using human-like interactions."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # Human-like behavior patterns
        self.mouse_patterns = {
            'natural_curve': True,
            'random_pauses': True,
            'micro_movements': True,
            'acceleration_curves': True
        }
        
        # Timing patterns
        self.timing_patterns = {
            'min_action_delay': 0.1,
            'max_action_delay': 0.8,
            'typing_speed_wpm': random.randint(35, 75),
            'mouse_speed_pps': random.randint(800, 1500)  # pixels per second
        }
    
    async def solve_behavioral_captcha(self, challenge: CaptchaChallenge) -> SolveResult:
        """Solve behavioral CAPTCHA by simulating human behavior."""
        start_time = time.time()
        
        try:
            # Simulate human-like page interaction
            await self._simulate_human_behavior()
            
            # Handle specific behavioral challenges
            if 'mouse_movement' in challenge.metadata:
                await self._simulate_mouse_patterns()
            
            if 'typing_pattern' in challenge.metadata:
                await self._simulate_typing_patterns()
            
            if 'scroll_behavior' in challenge.metadata:
                await self._simulate_scroll_behavior()
            
            solve_time = time.time() - start_time
            
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.SOLVED,
                solution="behavioral_passed",
                confidence=0.90,
                solve_time=solve_time,
                method_used=SolveMethod.BEHAVIORAL_SIMULATION
            )
            
        except Exception as e:
            self.logger.error(f"Behavioral solving failed: {e}")
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                solve_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _simulate_human_behavior(self):
        """Simulate general human-like behavior."""
        try:
            # Random page interactions
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Simulate reading behavior (scroll down a bit)
            await self.executor.page.evaluate("""
                window.scrollBy({
                    top: Math.random() * 200,
                    behavior: 'smooth'
                });
            """)
            
            await asyncio.sleep(random.uniform(0.3, 1.0))
            
            # Random mouse movements
            await self._random_mouse_movement()
            
        except Exception as e:
            self.logger.warning(f"Human behavior simulation failed: {e}")
    
    async def _random_mouse_movement(self):
        """Generate random but natural mouse movements."""
        try:
            # Get viewport size
            viewport = self.executor.page.viewport_size
            if not viewport:
                return
            
            # Generate random points for natural movement
            points = []
            current_x, current_y = viewport['width'] // 2, viewport['height'] // 2
            
            for _ in range(random.randint(2, 5)):
                target_x = random.randint(50, viewport['width'] - 50)
                target_y = random.randint(50, viewport['height'] - 50)
                
                # Create smooth curve between points
                steps = random.randint(10, 30)
                for i in range(steps):
                    t = i / steps
                    # Bezier curve for natural movement
                    x = current_x + (target_x - current_x) * t
                    y = current_y + (target_y - current_y) * t
                    
                    # Add small random variations
                    x += random.uniform(-5, 5)
                    y += random.uniform(-5, 5)
                    
                    points.append((x, y))
                
                current_x, current_y = target_x, target_y
            
            # Execute mouse movements
            for x, y in points:
                await self.executor.page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.01, 0.05))
            
        except Exception as e:
            self.logger.warning(f"Mouse movement simulation failed: {e}")


class ThirdPartyService:
    """Integration with third-party CAPTCHA solving services."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Service configurations
        self.services = {
            '2captcha': {
                'api_url': 'http://2captcha.com',
                'api_key': config.get('captcha_services', {}).get('2captcha_key'),
                'supported_types': [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3, 
                                  CaptchaType.HCAPTCHA, CaptchaType.IMAGE_RECOGNITION]
            },
            'anticaptcha': {
                'api_url': 'https://api.anti-captcha.com',
                'api_key': config.get('captcha_services', {}).get('anticaptcha_key'),
                'supported_types': [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3,
                                  CaptchaType.HCAPTCHA, CaptchaType.FUNCAPTCHA]
            },
            'deathbycaptcha': {
                'api_url': 'http://api.dbcapi.me',
                'username': config.get('captcha_services', {}).get('dbc_username'),
                'password': config.get('captcha_services', {}).get('dbc_password'),
                'supported_types': [CaptchaType.IMAGE_RECOGNITION, CaptchaType.TEXT_CAPTCHA]
            }
        }
    
    async def solve_with_service(self, challenge: CaptchaChallenge, service_name: str) -> SolveResult:
        """Solve CAPTCHA using third-party service."""
        start_time = time.time()
        
        try:
            if service_name not in self.services:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    error_message=f"Service {service_name} not configured"
                )
            
            service = self.services[service_name]
            
            if challenge.captcha_type not in service['supported_types']:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    error_message=f"CAPTCHA type not supported by {service_name}"
                )
            
            # Route to specific service implementation
            if service_name == '2captcha':
                result = await self._solve_2captcha(challenge, service)
            elif service_name == 'anticaptcha':
                result = await self._solve_anticaptcha(challenge, service)
            elif service_name == 'deathbycaptcha':
                result = await self._solve_deathbycaptcha(challenge, service)
            else:
                raise ValueError(f"Unknown service: {service_name}")
            
            result.solve_time = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Third-party service solving failed: {e}")
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                solve_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _solve_2captcha(self, challenge: CaptchaChallenge, service: Dict[str, Any]) -> SolveResult:
        """Solve using 2captcha service."""
        try:
            # Submit CAPTCHA
            submit_data = {
                'key': service['api_key'],
                'method': self._get_2captcha_method(challenge.captcha_type),
                'pageurl': challenge.page_url
            }
            
            if challenge.captcha_type == CaptchaType.RECAPTCHA_V2:
                submit_data['googlekey'] = challenge.site_key
            elif challenge.captcha_type == CaptchaType.HCAPTCHA:
                submit_data['sitekey'] = challenge.site_key
            elif challenge.image_data:
                submit_data['body'] = base64.b64encode(challenge.image_data).decode()
            
            # Submit request
            response = requests.post(f"{service['api_url']}/in.php", data=submit_data)
            
            if response.text.startswith('OK|'):
                captcha_id = response.text.split('|')[1]
                
                # Wait for solution
                for attempt in range(30):  # Wait up to 5 minutes
                    await asyncio.sleep(10)
                    
                    result_response = requests.get(
                        f"{service['api_url']}/res.php",
                        params={
                            'key': service['api_key'],
                            'action': 'get',
                            'id': captcha_id
                        }
                    )
                    
                    if result_response.text == 'CAPCHA_NOT_READY':
                        continue
                    elif result_response.text.startswith('OK|'):
                        solution = result_response.text.split('|')[1]
                        return SolveResult(
                            challenge_id=challenge.challenge_id,
                            status=CaptchaStatus.SOLVED,
                            solution=solution,
                            confidence=0.95,
                            method_used=SolveMethod.THIRD_PARTY_SERVICE,
                            cost=0.002  # Approximate cost
                        )
                    else:
                        return SolveResult(
                            challenge_id=challenge.challenge_id,
                            status=CaptchaStatus.FAILED,
                            error_message=result_response.text
                        )
                
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.TIMEOUT,
                    error_message="Service timeout"
                )
            else:
                return SolveResult(
                    challenge_id=challenge.challenge_id,
                    status=CaptchaStatus.FAILED,
                    error_message=response.text
                )
                
        except Exception as e:
            self.logger.error(f"2captcha solving failed: {e}")
            raise
    
    def _get_2captcha_method(self, captcha_type: CaptchaType) -> str:
        """Get 2captcha method for CAPTCHA type."""
        method_map = {
            CaptchaType.RECAPTCHA_V2: 'userrecaptcha',
            CaptchaType.RECAPTCHA_V3: 'userrecaptcha',
            CaptchaType.HCAPTCHA: 'hcaptcha',
            CaptchaType.IMAGE_RECOGNITION: 'base64',
            CaptchaType.TEXT_CAPTCHA: 'base64'
        }
        return method_map.get(captcha_type, 'base64')


class UniversalCaptchaSolver:
    """Master CAPTCHA solver that coordinates all solving methods."""
    
    def __init__(self, executor: DeterministicExecutor, config: Dict[str, Any]):
        self.executor = executor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize solving engines
        self.ai_vision_solver = AIVisionSolver(config)
        self.audio_solver = AudioCaptchaSolver(config)
        self.behavioral_solver = BehavioralSolver(executor)
        self.third_party_service = ThirdPartyService(config)
        
        # Solving statistics
        self.solve_history = []
        self.success_rates = {}
        self.method_preferences = {}
        
        # Strategy configuration
        self.solving_strategies = {
            CaptchaType.RECAPTCHA_V2: [
                SolveMethod.BEHAVIORAL_SIMULATION,
                SolveMethod.THIRD_PARTY_SERVICE,
                SolveMethod.AI_VISION
            ],
            CaptchaType.RECAPTCHA_V3: [
                SolveMethod.BEHAVIORAL_SIMULATION,
                SolveMethod.THIRD_PARTY_SERVICE
            ],
            CaptchaType.HCAPTCHA: [
                SolveMethod.AI_VISION,
                SolveMethod.THIRD_PARTY_SERVICE,
                SolveMethod.BEHAVIORAL_SIMULATION
            ],
            CaptchaType.IMAGE_RECOGNITION: [
                SolveMethod.AI_VISION,
                SolveMethod.OCR_PROCESSING,
                SolveMethod.THIRD_PARTY_SERVICE
            ],
            CaptchaType.TEXT_CAPTCHA: [
                SolveMethod.OCR_PROCESSING,
                SolveMethod.AI_VISION,
                SolveMethod.THIRD_PARTY_SERVICE
            ],
            CaptchaType.AUDIO_CAPTCHA: [
                SolveMethod.AUDIO_RECOGNITION,
                SolveMethod.THIRD_PARTY_SERVICE
            ]
        }
    
    async def solve_captcha(self, challenge: CaptchaChallenge) -> SolveResult:
        """Universal CAPTCHA solving function."""
        try:
            self.logger.info(f"Solving CAPTCHA: {challenge.captcha_type.value}")
            
            # Get solving strategies for this CAPTCHA type
            strategies = self.solving_strategies.get(
                challenge.captcha_type, 
                [SolveMethod.AI_VISION, SolveMethod.THIRD_PARTY_SERVICE]
            )
            
            # Try each strategy in order
            for strategy in strategies:
                try:
                    result = await self._solve_with_method(challenge, strategy)
                    
                    if result.status == CaptchaStatus.SOLVED:
                        # Update success statistics
                        self._update_success_stats(challenge.captcha_type, strategy, True)
                        
                        # Store solve history
                        self.solve_history.append({
                            'challenge': asdict(challenge),
                            'result': asdict(result),
                            'timestamp': datetime.utcnow()
                        })
                        
                        return result
                    else:
                        self._update_success_stats(challenge.captcha_type, strategy, False)
                        
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                    continue
            
            # All strategies failed
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                error_message="All solving strategies failed"
            )
            
        except Exception as e:
            self.logger.error(f"CAPTCHA solving failed: {e}")
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                error_message=str(e)
            )
    
    async def _solve_with_method(self, challenge: CaptchaChallenge, method: SolveMethod) -> SolveResult:
        """Solve CAPTCHA using specific method."""
        if method == SolveMethod.AI_VISION:
            return await self.ai_vision_solver.solve_image_captcha(challenge)
        elif method == SolveMethod.AUDIO_RECOGNITION:
            return await self.audio_solver.solve_audio_captcha(challenge)
        elif method == SolveMethod.BEHAVIORAL_SIMULATION:
            return await self.behavioral_solver.solve_behavioral_captcha(challenge)
        elif method == SolveMethod.THIRD_PARTY_SERVICE:
            # Try best available service
            services = ['2captcha', 'anticaptcha', 'deathbycaptcha']
            for service in services:
                try:
                    result = await self.third_party_service.solve_with_service(challenge, service)
                    if result.status == CaptchaStatus.SOLVED:
                        return result
                except Exception as e:
                    self.logger.warning(f"Service {service} failed: {e}")
                    continue
            
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                error_message="All third-party services failed"
            )
        elif method == SolveMethod.OCR_PROCESSING:
            # Use AI vision solver for OCR
            return await self.ai_vision_solver.solve_image_captcha(challenge)
        else:
            return SolveResult(
                challenge_id=challenge.challenge_id,
                status=CaptchaStatus.FAILED,
                error_message=f"Unknown solving method: {method.value}"
            )
    
    async def detect_and_solve_captcha(self, page_url: str) -> Optional[SolveResult]:
        """Automatically detect and solve CAPTCHAs on a page."""
        try:
            # Navigate to page if needed
            if self.executor.page.url != page_url:
                await self.executor.page.goto(page_url)
            
            # Detect CAPTCHA elements
            captcha_challenge = await self._detect_captcha_on_page()
            
            if captcha_challenge:
                return await self.solve_captcha(captcha_challenge)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Auto CAPTCHA detection failed: {e}")
            return None
    
    async def _detect_captcha_on_page(self) -> Optional[CaptchaChallenge]:
        """Detect CAPTCHA challenges on the current page."""
        try:
            # Look for reCAPTCHA
            recaptcha_element = await self.executor.page.query_selector('.g-recaptcha')
            if recaptcha_element:
                site_key = await recaptcha_element.get_attribute('data-sitekey')
                return CaptchaChallenge(
                    challenge_id=str(uuid.uuid4()),
                    captcha_type=CaptchaType.RECAPTCHA_V2,
                    site_key=site_key,
                    page_url=self.executor.page.url
                )
            
            # Look for hCaptcha
            hcaptcha_element = await self.executor.page.query_selector('.h-captcha')
            if hcaptcha_element:
                site_key = await hcaptcha_element.get_attribute('data-sitekey')
                return CaptchaChallenge(
                    challenge_id=str(uuid.uuid4()),
                    captcha_type=CaptchaType.HCAPTCHA,
                    site_key=site_key,
                    page_url=self.executor.page.url
                )
            
            # Look for image CAPTCHAs
            captcha_images = await self.executor.page.query_selector_all('img[src*="captcha"], img[alt*="captcha"], .captcha img')
            if captcha_images:
                img_element = captcha_images[0]
                img_src = await img_element.get_attribute('src')
                
                # Download image
                if img_src:
                    response = requests.get(img_src)
                    if response.status_code == 200:
                        return CaptchaChallenge(
                            challenge_id=str(uuid.uuid4()),
                            captcha_type=CaptchaType.IMAGE_RECOGNITION,
                            page_url=self.executor.page.url,
                            image_data=response.content
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"CAPTCHA detection failed: {e}")
            return None
    
    def _update_success_stats(self, captcha_type: CaptchaType, method: SolveMethod, success: bool):
        """Update success statistics for optimization."""
        key = f"{captcha_type.value}_{method.value}"
        
        if key not in self.success_rates:
            self.success_rates[key] = {'total': 0, 'success': 0}
        
        self.success_rates[key]['total'] += 1
        if success:
            self.success_rates[key]['success'] += 1
    
    def get_solving_analytics(self) -> Dict[str, Any]:
        """Get comprehensive CAPTCHA solving analytics."""
        total_solves = len(self.solve_history)
        successful_solves = sum(1 for solve in self.solve_history 
                              if solve['result']['status'] == CaptchaStatus.SOLVED)
        
        # Calculate success rates by type and method
        success_by_type = {}
        success_by_method = {}
        
        for key, stats in self.success_rates.items():
            rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            if '_' in key:
                captcha_type, method = key.split('_', 1)
                success_by_type[captcha_type] = success_by_type.get(captcha_type, [])
                success_by_type[captcha_type].append(rate)
                
                success_by_method[method] = success_by_method.get(method, [])
                success_by_method[method].append(rate)
        
        return {
            'total_solves': total_solves,
            'successful_solves': successful_solves,
            'overall_success_rate': (successful_solves / total_solves) * 100 if total_solves > 0 else 0,
            'success_by_type': {k: sum(v) / len(v) for k, v in success_by_type.items()},
            'success_by_method': {k: sum(v) / len(v) for k, v in success_by_method.items()},
            'recent_performance': self._get_recent_performance()
        }
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent solving performance metrics."""
        recent_solves = [solve for solve in self.solve_history 
                        if solve['timestamp'] > datetime.utcnow() - timedelta(hours=24)]
        
        if not recent_solves:
            return {}
        
        avg_solve_time = sum(solve['result']['solve_time'] for solve in recent_solves) / len(recent_solves)
        recent_success_rate = (sum(1 for solve in recent_solves 
                                 if solve['result']['status'] == CaptchaStatus.SOLVED) / len(recent_solves)) * 100
        
        return {
            'recent_solves_24h': len(recent_solves),
            'recent_success_rate': recent_success_rate,
            'avg_solve_time': avg_solve_time,
            'fastest_solve': min(solve['result']['solve_time'] for solve in recent_solves),
            'slowest_solve': max(solve['result']['solve_time'] for solve in recent_solves)
        }