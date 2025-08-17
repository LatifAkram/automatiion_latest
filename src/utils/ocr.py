#!/usr/bin/env python3
"""
Dependency-light OCR utility
- Uses system 'tesseract' if available (via subprocess)
- Returns structured result with text and metadata
- If tesseract is missing, returns an error status
"""

import io
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any


def _tesseract_available() -> bool:
    return shutil.which('tesseract') is not None


def extract_text_from_image_bytes(image_bytes: bytes, lang: str = 'eng') -> Dict[str, Any]:
    """Extract text from image bytes using system tesseract if available.
    Returns {'success': bool, 'text': str, 'error': Optional[str], 'engine': 'tesseract'|'none'}
    """
    if not image_bytes:
        return {'success': False, 'text': '', 'error': 'empty image', 'engine': 'none'}
    if not _tesseract_available():
        return {'success': False, 'text': '', 'error': 'tesseract not installed', 'engine': 'none'}
    tmpdir = tempfile.mkdtemp(prefix='ocr_')
    img_path = os.path.join(tmpdir, 'image.png')
    try:
        with open(img_path, 'wb') as f:
            f.write(image_bytes)
        # Run tesseract: output to stdout
        # Some versions support: tesseract image.png stdout -l eng
        cmd = ['tesseract', img_path, 'stdout', '-l', lang]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode != 0:
            return {'success': False, 'text': '', 'error': proc.stderr.decode('utf-8', errors='ignore'), 'engine': 'tesseract'}
        text = proc.stdout.decode('utf-8', errors='ignore')
        return {'success': True, 'text': text, 'error': None, 'engine': 'tesseract'}
    except Exception as e:
        return {'success': False, 'text': '', 'error': str(e), 'engine': 'tesseract'}
    finally:
        try:
            os.remove(img_path)
            os.rmdir(tmpdir)
        except Exception:
            pass