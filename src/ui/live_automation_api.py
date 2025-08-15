#!/usr/bin/env python3
"""
Live Automation API - Connect UI to Real Playwright Automation
===============================================================

REAL-TIME API for UI-triggered live automation:
✅ UI instruction → Real browser automation
✅ Live website interaction from frontend commands
✅ Real-time status updates and screenshots
✅ Actual performance metrics and healing
✅ WebSocket streaming for live updates

100% REAL AUTOMATION FROM UI INSTRUCTIONS!
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

# Import the real live automation
try:
    from testing.live_playwright_automation import get_live_playwright_automation, PLAYWRIGHT_AVAILABLE
    LIVE_AUTOMATION_AVAILABLE = True
except ImportError:
    LIVE_AUTOMATION_AVAILABLE = False
    print("⚠️ Live automation not available")

logger = logging.getLogger(__name__)

class LiveAutomationAPI:
    """
    API Bridge: UI Instructions → REAL Live Automation
    
    🎯 CAPABILITIES:
    ✅ Parse UI instructions into automation workflows
    ✅ Execute REAL browser automation on live websites
    ✅ Stream live updates back to UI (WebSocket)
    ✅ Capture real screenshots and performance data
    ✅ Handle errors with real healing and recovery
    """
    
    def __init__(self):
        self.automation = None
        self.active_sessions: Dict[str, Dict] = {}
        self.websocket_clients: List = []
        
        if LIVE_AUTOMATION_AVAILABLE:
            self.automation = get_live_playwright_automation({
                'browser': 'chromium',
                'mode': 'headful',  # Visible for demo
                'record_video': True,
                'screenshots_dir': 'data/live_screenshots',
                'videos_dir': 'data/live_videos'
            })
            logger.info("✅ Live Automation API initialized with REAL browser control")
        else:
            logger.warning("⚠️ Live Automation API initialized without Playwright")
    
    def add_websocket_client(self, websocket):
        """Add WebSocket client for live updates"""
        self.websocket_clients.append(websocket)
    
    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all WebSocket clients"""
        if self.websocket_clients:
            message_json = json.dumps(message)
            for client in self.websocket_clients[:]:  # Copy list to avoid modification during iteration
                try:
                    await client.send(message_json)
                except Exception as e:
                    logger.warning(f"⚠️ WebSocket client removed due to error: {e}")
                    self.websocket_clients.remove(client)
    
    async def execute_live_instruction(self, instruction: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute REAL live automation from UI instruction
        
        Examples:
        - "Search for 'AI automation' on Google"
        - "Navigate to GitHub and search for 'playwright'"
        - "Go to Stack Overflow and find Python questions"
        """
        if not LIVE_AUTOMATION_AVAILABLE or not self.automation:
            return {
                'success': False,
                'error': 'Live automation not available. Install Playwright: pip install playwright && playwright install',
                'instruction': instruction
            }
        
        session_id = f"live_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            logger.info(f"🚀 EXECUTING LIVE INSTRUCTION: {instruction}")
            
            # Broadcast start
            await self.broadcast_update({
                'type': 'automation_start',
                'session_id': session_id,
                'instruction': instruction,
                'timestamp': datetime.now().isoformat()
            })
            
            # Parse instruction into automation workflow
            workflow = await self._parse_instruction_to_workflow(instruction, context)
            
            if not workflow['success']:
                return workflow
            
            logger.info(f"📋 Parsed workflow: {len(workflow['steps'])} steps")
            
            # Execute the workflow with REAL automation
            execution_result = await self._execute_live_workflow(session_id, workflow['steps'])
            
            execution_time = time.time() - start_time
            
            # Broadcast completion
            await self.broadcast_update({
                'type': 'automation_complete',
                'session_id': session_id,
                'success': execution_result['success'],
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now().isoformat(),
                'results': execution_result
            })
            
            return {
                'success': execution_result['success'],
                'session_id': session_id,
                'instruction': instruction,
                'workflow_steps': len(workflow['steps']),
                'execution_time_seconds': execution_time,
                'results': execution_result,
                'live_automation': True,
                'playwright_used': True
            }
            
        except Exception as e:
            error_msg = f"Live automation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            await self.broadcast_update({
                'type': 'automation_error',
                'session_id': session_id,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': False,
                'error': error_msg,
                'instruction': instruction,
                'session_id': session_id
            }
    
    async def _parse_instruction_to_workflow(self, instruction: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse natural language instruction into automation workflow"""
        try:
            instruction_lower = instruction.lower()
            steps = []
            
            # Determine target website and action
            if 'google' in instruction_lower and 'search' in instruction_lower:
                # Google search workflow
                search_term = self._extract_search_term(instruction)
                steps = [
                    {'action': 'navigate', 'url': 'https://www.google.com', 'description': 'Navigate to Google'},
                    {'action': 'find_element', 'selector': 'input[name="q"]', 'description': 'Find search box'},
                    {'action': 'type', 'selector': 'input[name="q"]', 'text': search_term, 'description': f'Type search term: {search_term}'},
                    {'action': 'click', 'selector': 'input[value="Google Search"], button[type="submit"]', 'description': 'Click search button'},
                    {'action': 'wait_for_results', 'selector': '#search', 'description': 'Wait for search results'},
                    {'action': 'verify_results', 'expected': 'results', 'description': 'Verify search results appeared'}
                ]
                
            elif 'github' in instruction_lower:
                # GitHub workflow
                search_term = self._extract_search_term(instruction) or 'automation'
                steps = [
                    {'action': 'navigate', 'url': 'https://github.com', 'description': 'Navigate to GitHub'},
                    {'action': 'find_element', 'selector': 'input[placeholder*="Search"], .header-search-input', 'description': 'Find search box'},
                    {'action': 'type', 'selector': 'input[placeholder*="Search"], .header-search-input', 'text': search_term, 'description': f'Type search term: {search_term}'},
                    {'action': 'press_key', 'key': 'Enter', 'description': 'Press Enter to search'},
                    {'action': 'wait_for_results', 'selector': '[data-testid="results-list"], .repo-list', 'description': 'Wait for repository results'},
                    {'action': 'verify_results', 'expected': 'repositories', 'description': 'Verify repositories found'}
                ]
                
            elif 'stackoverflow' in instruction_lower or 'stack overflow' in instruction_lower:
                # Stack Overflow workflow
                search_term = self._extract_search_term(instruction) or 'python'
                steps = [
                    {'action': 'navigate', 'url': 'https://stackoverflow.com', 'description': 'Navigate to Stack Overflow'},
                    {'action': 'find_element', 'selector': 'input[name="q"], .s-input', 'description': 'Find search box'},
                    {'action': 'type', 'selector': 'input[name="q"], .s-input', 'text': search_term, 'description': f'Type search term: {search_term}'},
                    {'action': 'press_key', 'key': 'Enter', 'description': 'Press Enter to search'},
                    {'action': 'wait_for_results', 'selector': '#questions, .question-summary', 'description': 'Wait for question results'},
                    {'action': 'verify_results', 'expected': 'questions', 'description': 'Verify questions found'}
                ]
                
            elif 'wikipedia' in instruction_lower:
                # Wikipedia workflow
                search_term = self._extract_search_term(instruction) or 'artificial intelligence'
                steps = [
                    {'action': 'navigate', 'url': 'https://en.wikipedia.org', 'description': 'Navigate to Wikipedia'},
                    {'action': 'find_element', 'selector': '#searchInput', 'description': 'Find search box'},
                    {'action': 'type', 'selector': '#searchInput', 'text': search_term, 'description': f'Type search term: {search_term}'},
                    {'action': 'click', 'selector': '#searchButton, button[type="submit"]', 'description': 'Click search button'},
                    {'action': 'wait_for_results', 'selector': '#mw-content-text, .mw-parser-output', 'description': 'Wait for article content'},
                    {'action': 'verify_results', 'expected': 'article', 'description': 'Verify article content loaded'}
                ]
                
            else:
                # Generic workflow - try to extract URL and action
                if 'navigate' in instruction_lower or 'go to' in instruction_lower:
                    url = self._extract_url(instruction)
                    if url:
                        steps = [
                            {'action': 'navigate', 'url': url, 'description': f'Navigate to {url}'},
                            {'action': 'verify_results', 'expected': 'page loaded', 'description': 'Verify page loaded'}
                        ]
                    else:
                        return {
                            'success': False,
                            'error': f"Could not parse URL from instruction: {instruction}"
                        }
                else:
                    return {
                        'success': False,
                        'error': f"Could not parse instruction: {instruction}. Try 'Search for X on Google' or 'Navigate to URL'"
                    }
            
            return {
                'success': True,
                'steps': steps,
                'instruction': instruction,
                'workflow_type': 'live_automation'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Instruction parsing failed: {str(e)}"
            }
    
    def _extract_search_term(self, instruction: str) -> str:
        """Extract search term from instruction"""
        # Look for quoted strings first
        import re
        quoted_match = re.search(r"['\"]([^'\"]+)['\"]", instruction)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for "search for X" patterns
        search_patterns = [
            r"search for (.+?)(?:\s+on\s+|\s*$)",
            r"find (.+?)(?:\s+on\s+|\s*$)",
            r"look for (.+?)(?:\s+on\s+|\s*$)"
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, instruction.lower())
            if match:
                return match.group(1).strip()
        
        # Default search terms
        if 'automation' in instruction.lower():
            return 'automation'
        elif 'python' in instruction.lower():
            return 'python'
        elif 'ai' in instruction.lower():
            return 'artificial intelligence'
        
        return 'programming'
    
    def _extract_url(self, instruction: str) -> Optional[str]:
        """Extract URL from instruction"""
        import re
        
        # Look for full URLs
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, instruction)
        if match:
            return match.group(0)
        
        # Look for domain names
        domain_patterns = [
            r'(?:go to|navigate to|visit)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'([a-zA-Z0-9.-]+\.com)',
            r'([a-zA-Z0-9.-]+\.org)'
        ]
        
        for pattern in domain_patterns:
            match = re.search(pattern, instruction.lower())
            if match:
                domain = match.group(1)
                return f'https://{domain}' if not domain.startswith('http') else domain
        
        return None
    
    async def _execute_live_workflow(self, session_id: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute workflow with REAL live automation"""
        try:
            # Create live browser session
            session_result = await self.automation.create_live_session(session_id, 'about:blank')
            
            if not session_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to create live session: {session_result.get('error', 'Unknown error')}"
                }
            
            # Store session info
            self.active_sessions[session_id] = {
                'start_time': datetime.now(),
                'steps_completed': 0,
                'total_steps': len(steps),
                'screenshots': [],
                'status': 'running'
            }
            
            logger.info(f"✅ Created REAL live session: {session_id}")
            
            # Execute each step with real automation
            step_results = []
            
            for i, step in enumerate(steps):
                step_start_time = time.time()
                
                # Broadcast step start
                await self.broadcast_update({
                    'type': 'step_start',
                    'session_id': session_id,
                    'step_index': i,
                    'step': step,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"🔄 Step {i+1}/{len(steps)}: {step.get('description', step['action'])}")
                
                # Execute the step
                step_result = await self._execute_live_step(session_id, step)
                
                step_execution_time = time.time() - step_start_time
                step_result['execution_time_seconds'] = step_execution_time
                step_results.append(step_result)
                
                # Update session
                self.active_sessions[session_id]['steps_completed'] = i + 1
                if 'screenshot' in step_result:
                    self.active_sessions[session_id]['screenshots'].append(step_result['screenshot'])
                
                # Broadcast step completion
                await self.broadcast_update({
                    'type': 'step_complete',
                    'session_id': session_id,
                    'step_index': i,
                    'step_result': step_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"{'✅' if step_result['success'] else '❌'} Step {i+1}: {step_result.get('error', 'Success')}")
                
                # Stop if step failed and it's critical
                if not step_result['success'] and step['action'] in ['navigate', 'find_element']:
                    logger.error(f"❌ Critical step failed: {step['action']}")
                    break
                
                # Small delay between steps for realistic behavior
                await asyncio.sleep(0.5)
            
            # Close the session and get final statistics
            close_result = await self.automation.close_session(session_id)
            
            # Update session status
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['close_result'] = close_result
            
            # Calculate overall success
            successful_steps = sum(1 for result in step_results if result['success'])
            success_rate = (successful_steps / len(steps)) * 100 if steps else 0
            overall_success = success_rate >= 70  # 70% success threshold
            
            logger.info(f"🏁 Live workflow completed: {successful_steps}/{len(steps)} steps successful ({success_rate:.1f}%)")
            
            return {
                'success': overall_success,
                'steps_executed': len(step_results),
                'successful_steps': successful_steps,
                'success_rate_percent': success_rate,
                'step_results': step_results,
                'session_info': self.active_sessions[session_id],
                'close_result': close_result,
                'live_automation': True
            }
            
        except Exception as e:
            error_msg = f"Live workflow execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Clean up session
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = error_msg
            
            try:
                await self.automation.close_session(session_id)
            except:
                pass  # Session might not exist
            
            return {
                'success': False,
                'error': error_msg,
                'live_automation': True
            }
    
    async def _execute_live_step(self, session_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single automation step with REAL browser control"""
        action = step['action']
        
        try:
            if action == 'navigate':
                return await self.automation.live_navigate(session_id, step['url'])
                
            elif action == 'find_element':
                return await self.automation.live_find_element(session_id, step['selector'])
                
            elif action == 'type':
                return await self.automation.live_type(session_id, step['selector'], step['text'])
                
            elif action == 'click':
                return await self.automation.live_click(session_id, step['selector'])
                
            elif action == 'wait_for_results':
                return await self.automation.live_wait_for_results(session_id, step['selector'])
                
            elif action == 'verify_results':
                return await self.automation.live_verify_results(session_id, step['expected'])
                
            elif action == 'press_key':
                # Simulate key press by finding active element and sending key
                session = self.automation.sessions.get(session_id)
                if session and session.page:
                    await session.page.keyboard.press(step['key'])
                    return {
                        'success': True,
                        'action': 'press_key',
                        'key': step['key']
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Session not found for key press'
                    }
                    
            else:
                return {
                    'success': False,
                    'error': f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Step execution failed: {str(e)}",
                'action': action
            }
    
    async def get_live_status(self) -> Dict[str, Any]:
        """Get current live automation status"""
        if not LIVE_AUTOMATION_AVAILABLE or not self.automation:
            return {
                'available': False,
                'error': 'Live automation not available'
            }
        
        stats = await self.automation.get_live_statistics()
        
        return {
            'available': True,
            'playwright_available': PLAYWRIGHT_AVAILABLE,
            'active_sessions': len(self.active_sessions),
            'websocket_clients': len(self.websocket_clients),
            'session_details': {
                session_id: {
                    'steps_completed': info['steps_completed'],
                    'total_steps': info['total_steps'],
                    'status': info['status'],
                    'screenshots_count': len(info['screenshots'])
                }
                for session_id, info in self.active_sessions.items()
            },
            'automation_stats': stats
        }
    
    async def get_session_screenshots(self, session_id: str) -> Dict[str, Any]:
        """Get screenshots from a live session"""
        if session_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found'
            }
        
        session_info = self.active_sessions[session_id]
        screenshots = session_info.get('screenshots', [])
        
        return {
            'success': True,
            'session_id': session_id,
            'screenshots': screenshots,
            'count': len(screenshots)
        }

# Global API instance
_live_automation_api = None

def get_live_automation_api() -> LiveAutomationAPI:
    """Get global live automation API instance"""
    global _live_automation_api
    
    if _live_automation_api is None:
        _live_automation_api = LiveAutomationAPI()
    
    return _live_automation_api

# Example usage and testing
async def test_live_automation_api():
    """Test the live automation API"""
    print("🚀 TESTING LIVE AUTOMATION API")
    print("=" * 40)
    
    api = get_live_automation_api()
    
    # Test status
    status = await api.get_live_status()
    print(f"📊 API Available: {status['available']}")
    print(f"🎭 Playwright Available: {status.get('playwright_available', False)}")
    
    if status['available']:
        # Test live instruction execution
        instructions = [
            "Search for 'playwright automation' on Google",
            "Navigate to GitHub and search for 'automation'",
            "Go to Stack Overflow and find Python questions"
        ]
        
        for instruction in instructions:
            print(f"\n🎯 Testing: {instruction}")
            result = await api.execute_live_instruction(instruction)
            print(f"   {'✅' if result['success'] else '❌'} Result: {result.get('success_rate_percent', 0):.1f}% success")
            if result['success'] and 'results' in result:
                print(f"   📊 Steps: {result['results'].get('successful_steps', 0)}/{result['results'].get('steps_executed', 0)}")
                if 'close_result' in result['results'] and result['results']['close_result']['success']:
                    print(f"   📸 Screenshots: {result['results']['close_result']['screenshots_count']}")

if __name__ == "__main__":
    # Run API test
    asyncio.run(test_live_automation_api())