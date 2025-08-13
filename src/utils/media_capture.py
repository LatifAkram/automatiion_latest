"""
Media Capture
============

Utilities for capturing screenshots and videos during automation with
advanced features like selective capture, compression, and storage management.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

try:
    from PIL import Image
    import cv2
    from moviepy.editor import VideoFileClip, ImageSequenceClip
    MEDIA_AVAILABLE = True
except ImportError:
    MEDIA_AVAILABLE = False


class MediaCapture:
    """Media capture utilities for automation."""
    
    def __init__(self, media_path: str):
        self.media_path = media_path
        self.logger = logging.getLogger(__name__)
        
        # Media storage structure
        self.screenshots_path = Path(media_path) / "screenshots"
        self.videos_path = Path(media_path) / "videos"
        self.recordings_path = Path(media_path) / "recordings"
        
        # Video recording state
        self.active_recordings: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize media capture directories."""
        try:
            # Create media directories
            self.screenshots_path.mkdir(parents=True, exist_ok=True)
            self.videos_path.mkdir(parents=True, exist_ok=True)
            self.recordings_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Media capture initialized: {self.media_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize media capture: {e}", exc_info=True)
            raise
            
    async def capture_screenshot(self, page, task_id: str, name: str, 
                               quality: str = "high", format: str = "png") -> str:
        """
        Capture screenshot from browser page.
        
        Args:
            page: Playwright page object
            task_id: Task identifier
            name: Screenshot name
            quality: Screenshot quality (low, medium, high)
            format: Image format (png, jpg, webp)
            
        Returns:
            Path to saved screenshot
        """
        try:
            if not MEDIA_AVAILABLE:
                self.logger.warning("Media libraries not available, using basic capture")
                return await self._basic_screenshot(page, task_id, name)
                
            # Generate filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_id}_{name}_{timestamp}.{format}"
            filepath = self.screenshots_path / filename
            
            # Capture screenshot using Playwright
            screenshot_bytes = await page.screenshot(
                full_page=True,
                type=format,
                quality=95 if quality == "high" else 80 if quality == "medium" else 60
            )
            
            # Save screenshot
            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)
                
            # Optimize image if needed
            if quality != "high":
                await self._optimize_image(filepath, quality)
                
            self.logger.info(f"Captured screenshot: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}", exc_info=True)
            return ""
            
    async def _basic_screenshot(self, page, task_id: str, name: str) -> str:
        """Basic screenshot capture without optimization."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_id}_{name}_{timestamp}.png"
            filepath = self.screenshots_path / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            screenshot_bytes = await page.screenshot(full_page=True)
            
            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)
                
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Basic screenshot failed: {e}", exc_info=True)
            return ""
            
    async def _optimize_image(self, filepath: Path, quality: str):
        """Optimize image file size and quality."""
        try:
            if not MEDIA_AVAILABLE:
                return
                
            # Open image
            with Image.open(filepath) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                    
                # Resize if too large
                max_size = 1920 if quality == "medium" else 1280 if quality == "low" else 3840
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                # Save optimized image
                save_kwargs = {}
                if filepath.suffix.lower() == '.jpg':
                    save_kwargs['quality'] = 85 if quality == "medium" else 70 if quality == "low" else 95
                    save_kwargs['optimize'] = True
                elif filepath.suffix.lower() == '.webp':
                    save_kwargs['quality'] = 85 if quality == "medium" else 70 if quality == "low" else 95
                    
                img.save(filepath, **save_kwargs)
                
        except Exception as e:
            self.logger.warning(f"Image optimization failed: {e}")
            
    async def start_video_recording(self, page, task_id: str, name: str) -> str:
        """
        Start video recording of browser session.
        
        Args:
            page: Playwright page object
            task_id: Task identifier
            name: Recording name
            
        Returns:
            Recording ID
        """
        try:
            recording_id = f"{task_id}_{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Start video recording
            video_path = self.recordings_path / f"{recording_id}.webm"
            
            # Start recording using Playwright
            await page.video.start(path=str(video_path))
            
            # Store recording info
            self.active_recordings[recording_id] = {
                "page": page,
                "video_path": video_path,
                "start_time": datetime.utcnow(),
                "task_id": task_id,
                "name": name
            }
            
            self.logger.info(f"Started video recording: {recording_id}")
            return recording_id
            
        except Exception as e:
            self.logger.error(f"Failed to start video recording: {e}", exc_info=True)
            return ""
            
    async def stop_video_recording(self, recording_id: str) -> str:
        """
        Stop video recording and save the file.
        
        Args:
            recording_id: Recording identifier
            
        Returns:
            Path to saved video file
        """
        try:
            if recording_id not in self.active_recordings:
                self.logger.warning(f"Recording {recording_id} not found")
                return ""
                
            recording_info = self.active_recordings[recording_id]
            page = recording_info["page"]
            video_path = recording_info["video_path"]
            
            # Stop recording
            await page.video.stop()
            
            # Move to videos directory
            final_path = self.videos_path / video_path.name
            if video_path.exists():
                video_path.rename(final_path)
                
            # Clean up recording info
            del self.active_recordings[recording_id]
            
            self.logger.info(f"Stopped video recording: {final_path}")
            return str(final_path)
            
        except Exception as e:
            self.logger.error(f"Failed to stop video recording: {e}", exc_info=True)
            return ""
            
    async def capture_element_screenshot(self, page, element_selector: str, 
                                       task_id: str, name: str) -> str:
        """
        Capture screenshot of specific element.
        
        Args:
            page: Playwright page object
            element_selector: CSS selector for element
            task_id: Task identifier
            name: Screenshot name
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Wait for element to be visible
            await page.wait_for_selector(element_selector, timeout=5000)
            
            # Capture element screenshot
            element = page.locator(element_selector)
            screenshot_bytes = await element.screenshot()
            
            # Save screenshot
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_id}_{name}_{timestamp}.png"
            filepath = self.screenshots_path / filename
            
            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)
                
            self.logger.info(f"Captured element screenshot: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to capture element screenshot: {e}", exc_info=True)
            return ""
            
    async def capture_full_page_screenshot(self, page, task_id: str, name: str) -> str:
        """
        Capture full page screenshot including scrollable content.
        
        Args:
            page: Playwright page object
            task_id: Task identifier
            name: Screenshot name
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Get page dimensions
            viewport = await page.evaluate("""
                () => {
                    return {
                        width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                        height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                    }
                }
            """)
            
            # Set viewport to full page size
            await page.set_viewport_size({
                "width": viewport["width"],
                "height": viewport["height"]
            })
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # Save screenshot
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_id}_{name}_{timestamp}.png"
            filepath = self.screenshots_path / filename
            
            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)
                
            self.logger.info(f"Captured full page screenshot: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to capture full page screenshot: {e}", exc_info=True)
            return ""
            
    async def create_screenshot_sequence(self, page, task_id: str, name: str, 
                                       interval: float = 1.0, duration: float = 10.0) -> List[str]:
        """
        Create a sequence of screenshots over time.
        
        Args:
            page: Playwright page object
            task_id: Task identifier
            name: Sequence name
            interval: Time between screenshots (seconds)
            duration: Total duration (seconds)
            
        Returns:
            List of screenshot file paths
        """
        try:
            screenshots = []
            start_time = datetime.utcnow()
            screenshot_count = 0
            
            while (datetime.utcnow() - start_time).total_seconds() < duration:
                # Capture screenshot
                screenshot_path = await self.capture_screenshot(
                    page, task_id, f"{name}_seq_{screenshot_count:03d}"
                )
                
                if screenshot_path:
                    screenshots.append(screenshot_path)
                    screenshot_count += 1
                    
                # Wait for next interval
                await asyncio.sleep(interval)
                
            self.logger.info(f"Created screenshot sequence: {len(screenshots)} screenshots")
            return screenshots
            
        except Exception as e:
            self.logger.error(f"Failed to create screenshot sequence: {e}", exc_info=True)
            return []
            
    async def create_video_from_screenshots(self, screenshot_paths: List[str], 
                                          task_id: str, name: str, fps: int = 2) -> str:
        """
        Create video from sequence of screenshots.
        
        Args:
            screenshot_paths: List of screenshot file paths
            task_id: Task identifier
            name: Video name
            fps: Frames per second
            
        Returns:
            Path to created video file
        """
        try:
            if not MEDIA_AVAILABLE or not screenshot_paths:
                return ""
                
            # Load images
            images = []
            for path in screenshot_paths:
                if Path(path).exists():
                    img = Image.open(path)
                    images.append(img)
                    
            if not images:
                return ""
                
            # Create video
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{task_id}_{name}_{timestamp}.mp4"
            video_path = self.videos_path / video_filename
            
            # Create video clip
            clip = ImageSequenceClip(images, fps=fps)
            clip.write_videofile(str(video_path), codec='libx264')
            
            self.logger.info(f"Created video from screenshots: {video_path}")
            return str(video_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create video from screenshots: {e}", exc_info=True)
            return ""
            
    async def compress_video(self, video_path: str, quality: str = "medium") -> str:
        """
        Compress video file to reduce size.
        
        Args:
            video_path: Path to video file
            quality: Compression quality (low, medium, high)
            
        Returns:
            Path to compressed video file
        """
        try:
            if not MEDIA_AVAILABLE:
                return video_path
                
            # Load video
            clip = VideoFileClip(video_path)
            
            # Determine compression settings
            if quality == "low":
                bitrate = "500k"
                resolution = (640, 480)
            elif quality == "medium":
                bitrate = "1000k"
                resolution = (1280, 720)
            else:  # high
                bitrate = "2000k"
                resolution = (1920, 1080)
                
            # Resize if needed
            if clip.size[0] > resolution[0] or clip.size[1] > resolution[1]:
                clip = clip.resize(resolution)
                
            # Create compressed file path
            compressed_path = video_path.replace(".mp4", f"_compressed_{quality}.mp4")
            
            # Write compressed video
            clip.write_videofile(
                compressed_path,
                codec='libx264',
                bitrate=bitrate,
                audio_codec='aac'
            )
            
            clip.close()
            
            self.logger.info(f"Compressed video: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            self.logger.error(f"Failed to compress video: {e}", exc_info=True)
            return video_path
            
    async def cleanup_old_media(self, max_age_days: int = 30):
        """
        Clean up old media files to save storage space.
        
        Args:
            max_age_days: Maximum age of files to keep (days)
        """
        try:
            cutoff_time = datetime.utcnow().timestamp() - (max_age_days * 24 * 60 * 60)
            
            # Clean up screenshots
            for filepath in self.screenshots_path.glob("*"):
                if filepath.stat().st_mtime < cutoff_time:
                    filepath.unlink()
                    self.logger.info(f"Deleted old screenshot: {filepath}")
                    
            # Clean up videos
            for filepath in self.videos_path.glob("*"):
                if filepath.stat().st_mtime < cutoff_time:
                    filepath.unlink()
                    self.logger.info(f"Deleted old video: {filepath}")
                    
            # Clean up recordings
            for filepath in self.recordings_path.glob("*"):
                if filepath.stat().st_mtime < cutoff_time:
                    filepath.unlink()
                    self.logger.info(f"Deleted old recording: {filepath}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old media: {e}", exc_info=True)
            
    async def get_media_statistics(self) -> Dict[str, Any]:
        """Get statistics about captured media."""
        try:
            stats = {
                "screenshots": {
                    "count": len(list(self.screenshots_path.glob("*"))),
                    "total_size_mb": sum(f.stat().st_size for f in self.screenshots_path.glob("*")) / (1024 * 1024)
                },
                "videos": {
                    "count": len(list(self.videos_path.glob("*"))),
                    "total_size_mb": sum(f.stat().st_size for f in self.videos_path.glob("*")) / (1024 * 1024)
                },
                "recordings": {
                    "count": len(list(self.recordings_path.glob("*"))),
                    "total_size_mb": sum(f.stat().st_size for f in self.recordings_path.glob("*")) / (1024 * 1024)
                },
                "active_recordings": len(self.active_recordings)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get media statistics: {e}", exc_info=True)
            return {}
            
    async def shutdown(self):
        """Shutdown media capture and cleanup."""
        try:
            # Stop all active recordings
            for recording_id in list(self.active_recordings.keys()):
                await self.stop_video_recording(recording_id)
                
            self.logger.info("Media capture shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during media capture shutdown: {e}", exc_info=True)