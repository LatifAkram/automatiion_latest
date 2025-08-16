#!/usr/bin/env python3
"""
SIMPLE RELIABLE AUTOMATION
==========================
A no-nonsense automation system that actually works for simple tasks.
Less sophistication, more reliability.
"""

import asyncio
import time
import os
from playwright.async_api import async_playwright

class SimpleReliableAutomation:
    """Simple automation that prioritizes working over being sophisticated"""
    
    def __init__(self):
        self.browser = None
        self.page = None
    
    async def execute(self, instruction: str) -> dict:
        """Execute automation with focus on reliability"""
        instruction_lower = instruction.lower()
        
        try:
            async with async_playwright() as p:
                # Launch browser with realistic settings
                self.browser = await p.chromium.launch(
                    headless=False,
                    args=['--no-sandbox', '--disable-web-security']
                )
                context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                self.page = await context.new_page()
                
                # Determine what to do based on instruction
                if 'youtube' in instruction_lower:
                    return await self._handle_youtube(instruction)
                elif 'google' in instruction_lower:
                    return await self._handle_google(instruction)
                else:
                    return await self._handle_generic(instruction)
                    
        except Exception as e:
            return {
                'success': False,
                'error': f'Automation failed: {str(e)}',
                'message': 'Simple automation encountered an error'
            }
        finally:
            if self.browser:
                await self.browser.close()
    
    async def _handle_youtube(self, instruction: str) -> dict:
        """Handle YouTube automation with multiple fallback strategies"""
        try:
            # Step 1: Navigate to YouTube
            await self.page.goto('https://www.youtube.com', wait_until='networkidle')
            await asyncio.sleep(3)  # Give it time to load
            
            actions = ['Opened YouTube']
            
            # Step 2: Handle search if needed
            if any(word in instruction.lower() for word in ['search', 'find', 'play', 'trending', 'songs']):
                search_success = await self._youtube_search(instruction)
                if search_success:
                    actions.append('Performed search successfully')
                else:
                    actions.append('Search failed - YouTube DOM may have changed')
            
            # Step 3: Take screenshot
            screenshot_path = f"screenshots/simple_automation_{int(time.time())}.png"
            os.makedirs("screenshots", exist_ok=True)
            await self.page.screenshot(path=screenshot_path)
            
            return {
                'success': True,
                'message': f'YouTube automation completed with {len(actions)} actions',
                'url': self.page.url,
                'actions_performed': actions,
                'screenshot': screenshot_path,
                'page_title': await self.page.title()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'YouTube automation failed'
            }
    
    async def _youtube_search(self, instruction: str) -> bool:
        """Robust YouTube search with multiple strategies"""
        
        # Extract search terms
        search_terms = "trending songs 2025"  # Default
        if 'trending' in instruction.lower():
            search_terms = "trending songs 2025"
        elif 'music' in instruction.lower():
            search_terms = "music"
        
        # Strategy 1: Try to find and use search box
        search_strategies = [
            # Most specific first
            "input[name='search_query']",
            "input#search",
            "[placeholder*='Search']",
            "input[aria-label*='Search']",
            "input[type='text']",
            # Fallback to any input that might be search
            "input"
        ]
        
        for i, selector in enumerate(search_strategies):
            try:
                print(f"🔍 Trying search strategy {i+1}: {selector}")
                
                # Wait for element to be available
                await self.page.wait_for_selector(selector, timeout=5000)
                
                # Get the element
                search_box = self.page.locator(selector).first()
                
                # Check if it's visible and enabled
                if await search_box.is_visible() and await search_box.is_enabled():
                    # Click to focus
                    await search_box.click()
                    await asyncio.sleep(0.5)
                    
                    # Clear and type
                    await search_box.fill('')
                    await asyncio.sleep(0.5)
                    await search_box.type(search_terms, delay=100)
                    await asyncio.sleep(1)
                    
                    # Try to submit
                    await self.page.keyboard.press('Enter')
                    await asyncio.sleep(2)
                    
                    print(f"✅ Search strategy {i+1} succeeded!")
                    return True
                    
            except Exception as e:
                print(f"❌ Search strategy {i+1} failed: {e}")
                continue
        
        # Strategy 2: Try clicking trending directly
        trending_selectors = [
            "text=Trending",
            "[aria-label*='Trending']",
            "a[href*='trending']",
            "yt-formatted-string:has-text('Trending')"
        ]
        
        for selector in trending_selectors:
            try:
                element = self.page.locator(selector).first()
                if await element.is_visible():
                    await element.click()
                    await asyncio.sleep(2)
                    print("✅ Clicked Trending section!")
                    return True
            except:
                continue
        
        print("❌ All search strategies failed")
        return False
    
    async def _handle_google(self, instruction: str) -> dict:
        """Handle Google automation"""
        try:
            await self.page.goto('https://www.google.com', wait_until='networkidle')
            await asyncio.sleep(2)
            
            screenshot_path = f"screenshots/simple_automation_{int(time.time())}.png"
            os.makedirs("screenshots", exist_ok=True)
            await self.page.screenshot(path=screenshot_path)
            
            return {
                'success': True,
                'message': 'Google opened successfully',
                'url': self.page.url,
                'actions_performed': ['Opened Google'],
                'screenshot': screenshot_path,
                'page_title': await self.page.title()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Google automation failed'
            }
    
    async def _handle_generic(self, instruction: str) -> dict:
        """Handle generic instructions"""
        return {
            'success': False,
            'message': 'Generic automation not implemented yet',
            'suggestion': 'Try instructions with "youtube" or "google"'
        }

# Integration function for the existing system
async def execute_simple_reliable_automation(instruction: str) -> dict:
    """Execute simple reliable automation"""
    automation = SimpleReliableAutomation()
    return await automation.execute(instruction)