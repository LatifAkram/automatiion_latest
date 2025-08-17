"""
COMPREHENSIVE SESSION MANAGEMENT SYSTEM
=======================================

Enterprise-grade session management for complex automation workflows.
Handles authentication, session persistence, state management, and recovery.

✅ FEATURES:
- Multi-platform session management
- Authentication flow handling
- Session persistence and recovery
- Cookie and token management
- State synchronization
- Automatic session renewal
- Security and encryption
"""

import asyncio
import json
import time
import logging
import hashlib
import base64
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pickle
from playwright.async_api import Page, BrowserContext

logger = logging.getLogger(__name__)

class SessionState(Enum):
    INACTIVE = "inactive"
    AUTHENTICATING = "authenticating"
    ACTIVE = "active"
    EXPIRED = "expired"
    ERROR = "error"
    RENEWING = "renewing"

class AuthMethod(Enum):
    USERNAME_PASSWORD = "username_password"
    SSO = "sso"
    API_KEY = "api_key"
    OAUTH = "oauth"
    TWO_FACTOR = "two_factor"
    BIOMETRIC = "biometric"

@dataclass
class SessionCredentials:
    """Credentials for session authentication"""
    username: str = ""
    password: str = ""
    api_key: str = ""
    oauth_token: str = ""
    refresh_token: str = ""
    two_factor_code: str = ""
    additional_fields: Dict[str, str] = field(default_factory=dict)

@dataclass
class SessionInfo:
    """Information about an active session"""
    session_id: str
    platform: str
    user_id: str
    state: SessionState
    auth_method: AuthMethod
    created_at: float
    last_activity: float
    expires_at: float
    cookies: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, str] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    page_context: Optional[Page] = None

class SessionManager:
    """Enterprise session management system"""
    
    def __init__(self, storage_path: str = "sessions"):
        self.storage_path = storage_path
        self.sessions: Dict[str, SessionInfo] = {}
        self.platform_configs: Dict[str, Dict[str, Any]] = {}
        self.encryption_key = self._generate_encryption_key()
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Load platform configurations
        self._load_platform_configurations()
        
        # Load persisted sessions
        self._load_persisted_sessions()
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data"""
        key_file = os.path.join(self.storage_path, ".session_key")
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = os.urandom(32)  # 256-bit key
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _load_platform_configurations(self):
        """Load platform-specific authentication configurations"""
        self.platform_configs = {
            'amazon': {
                'login_url': 'https://www.amazon.com/ap/signin',
                'username_selector': 'input[name="email"]',
                'password_selector': 'input[name="password"]',
                'submit_selector': 'input[id="signInSubmit"]',
                'success_indicators': ['nav-user-name', 'nav-line-1'],
                'session_duration': 3600 * 24,  # 24 hours
                'supports_2fa': True
            },
            'flipkart': {
                'login_url': 'https://www.flipkart.com/account/login',
                'username_selector': 'input[class*="username"]',
                'password_selector': 'input[class*="password"]',
                'submit_selector': 'button[type="submit"]',
                'success_indicators': ['_2dxgBa', 'account-dropdown'],
                'session_duration': 3600 * 12,  # 12 hours
                'supports_2fa': False
            },
            'facebook': {
                'login_url': 'https://www.facebook.com/login',
                'username_selector': 'input[name="email"]',
                'password_selector': 'input[name="pass"]',
                'submit_selector': 'button[name="login"]',
                'success_indicators': ['[data-testid="nav-user-name"]', '[role="banner"]'],
                'session_duration': 3600 * 8,  # 8 hours
                'supports_2fa': True
            },
            'linkedin': {
                'login_url': 'https://www.linkedin.com/login',
                'username_selector': 'input[id="username"]',
                'password_selector': 'input[id="password"]',
                'submit_selector': 'button[type="submit"]',
                'success_indicators': ['global-nav', 'feed-container'],
                'session_duration': 3600 * 6,  # 6 hours
                'supports_2fa': True
            },
            'salesforce': {
                'login_url': 'https://login.salesforce.com',
                'username_selector': 'input[id="username"]',
                'password_selector': 'input[id="password"]',
                'submit_selector': 'input[id="Login"]',
                'success_indicators': ['oneHeader', 'slds-context-bar'],
                'session_duration': 3600 * 2,  # 2 hours
                'supports_2fa': True
            },
            'github': {
                'login_url': 'https://github.com/login',
                'username_selector': 'input[name="login"]',
                'password_selector': 'input[name="password"]',
                'submit_selector': 'input[type="submit"]',
                'success_indicators': ['Header-link--profile', 'user-nav'],
                'session_duration': 3600 * 24,  # 24 hours
                'supports_2fa': True
            }
        }
    
    async def create_session(self, platform: str, credentials: SessionCredentials, 
                           page: Page, session_id: str = None) -> SessionInfo:
        """Create a new authenticated session"""
        if session_id is None:
            session_id = self._generate_session_id(platform, credentials.username)
        
        session_info = SessionInfo(
            session_id=session_id,
            platform=platform,
            user_id=credentials.username,
            state=SessionState.AUTHENTICATING,
            auth_method=AuthMethod.USERNAME_PASSWORD,
            created_at=time.time(),
            last_activity=time.time(),
            expires_at=time.time() + self.platform_configs.get(platform, {}).get('session_duration', 3600),
            page_context=page
        )
        
        try:
            # Perform authentication
            auth_result = await self._authenticate_session(session_info, credentials)
            
            if auth_result['success']:
                session_info.state = SessionState.ACTIVE
                session_info.cookies = auth_result.get('cookies', {})
                session_info.tokens = auth_result.get('tokens', {})
                session_info.session_data = auth_result.get('session_data', {})
                
                # Store session
                self.sessions[session_id] = session_info
                
                # Persist session
                await self._persist_session(session_info)
                
                logger.info(f"Session created successfully for {platform}: {session_id}")
                return session_info
            else:
                session_info.state = SessionState.ERROR
                raise Exception(f"Authentication failed: {auth_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            session_info.state = SessionState.ERROR
            logger.error(f"Failed to create session for {platform}: {e}")
            raise
    
    async def _authenticate_session(self, session_info: SessionInfo, 
                                  credentials: SessionCredentials) -> Dict[str, Any]:
        """Perform platform-specific authentication"""
        platform = session_info.platform
        page = session_info.page_context
        config = self.platform_configs.get(platform, {})
        
        if not config:
            return {'success': False, 'error': f'Platform {platform} not configured'}
        
        try:
            # Navigate to login page
            await page.goto(config['login_url'])
            await page.wait_for_load_state('domcontentloaded')
            
            # Fill username
            username_selector = config['username_selector']
            await page.wait_for_selector(username_selector, timeout=10000)
            await page.fill(username_selector, credentials.username)
            
            # Fill password
            password_selector = config['password_selector']
            await page.fill(password_selector, credentials.password)
            
            # Submit form
            submit_selector = config['submit_selector']
            await page.click(submit_selector)
            
            # Wait for authentication to complete
            await self._wait_for_authentication_completion(page, config)
            
            # Handle 2FA if required
            if config.get('supports_2fa') and credentials.two_factor_code:
                await self._handle_two_factor_authentication(page, credentials.two_factor_code)
            
            # Verify successful login
            success = await self._verify_authentication_success(page, config)
            
            if success:
                # Extract session data
                cookies = await self._extract_cookies(page)
                tokens = await self._extract_tokens(page)
                session_data = await self._extract_session_data(page, platform)
                
                return {
                    'success': True,
                    'cookies': cookies,
                    'tokens': tokens,
                    'session_data': session_data
                }
            else:
                return {'success': False, 'error': 'Authentication verification failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _wait_for_authentication_completion(self, page: Page, config: Dict[str, Any]):
        """Wait for authentication to complete"""
        try:
            # Wait for either success indicators or error messages
            await page.wait_for_function(
                """() => {
                    // Check for success indicators
                    const successSelectors = arguments[0];
                    for (const selector of successSelectors) {
                        if (document.querySelector(selector)) return true;
                    }
                    
                    // Check for error messages
                    const errorSelectors = ['[data-testid="error"]', '.error', '.alert-error'];
                    for (const selector of errorSelectors) {
                        if (document.querySelector(selector)) return true;
                    }
                    
                    return false;
                }""",
                config.get('success_indicators', []),
                timeout=15000
            )
        except:
            # Timeout is acceptable, we'll verify success separately
            pass
    
    async def _handle_two_factor_authentication(self, page: Page, two_factor_code: str):
        """Handle two-factor authentication"""
        try:
            # Common 2FA selectors
            two_fa_selectors = [
                'input[name="code"]',
                'input[name="otp"]',
                'input[placeholder*="code"]',
                'input[aria-label*="code"]'
            ]
            
            for selector in two_fa_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.fill(selector, two_factor_code)
                    
                    # Find and click submit button
                    submit_selectors = [
                        'button[type="submit"]',
                        'input[type="submit"]',
                        'button:contains("Verify")',
                        'button:contains("Continue")'
                    ]
                    
                    for submit_selector in submit_selectors:
                        try:
                            await page.click(submit_selector)
                            break
                        except:
                            continue
                    
                    break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"2FA handling failed: {e}")
    
    async def _verify_authentication_success(self, page: Page, config: Dict[str, Any]) -> bool:
        """Verify that authentication was successful"""
        success_indicators = config.get('success_indicators', [])
        
        for indicator in success_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=5000)
                if element:
                    return True
            except:
                continue
        
        # Additional checks
        try:
            # Check if we're redirected away from login page
            current_url = page.url
            login_url = config['login_url']
            
            if login_url not in current_url:
                return True
                
            # Check for absence of error messages
            error_selectors = ['.error', '.alert-error', '[data-testid="error"]']
            for error_selector in error_selectors:
                try:
                    error_element = await page.wait_for_selector(error_selector, timeout=1000)
                    if error_element:
                        return False
                except:
                    continue
            
            return True
            
        except:
            return False
    
    async def _extract_cookies(self, page: Page) -> Dict[str, Any]:
        """Extract session cookies"""
        try:
            cookies = await page.context.cookies()
            return {cookie['name']: cookie['value'] for cookie in cookies}
        except:
            return {}
    
    async def _extract_tokens(self, page: Page) -> Dict[str, str]:
        """Extract authentication tokens from page"""
        try:
            # Extract common token types from localStorage and sessionStorage
            tokens = await page.evaluate("""() => {
                const tokens = {};
                
                // Check localStorage
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && (key.includes('token') || key.includes('auth'))) {
                        tokens[key] = localStorage.getItem(key);
                    }
                }
                
                // Check sessionStorage
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    if (key && (key.includes('token') || key.includes('auth'))) {
                        tokens[key] = sessionStorage.getItem(key);
                    }
                }
                
                return tokens;
            }""")
            
            return tokens or {}
            
        except:
            return {}
    
    async def _extract_session_data(self, page: Page, platform: str) -> Dict[str, Any]:
        """Extract platform-specific session data"""
        try:
            session_data = {
                'url': page.url,
                'title': await page.title(),
                'user_agent': await page.evaluate('navigator.userAgent'),
                'timestamp': time.time()
            }
            
            # Platform-specific data extraction
            if platform == 'amazon':
                try:
                    user_name = await page.text_content('#nav-link-accountList-nav-line-1')
                    session_data['user_name'] = user_name
                except:
                    pass
            
            elif platform == 'facebook':
                try:
                    user_name = await page.text_content('[data-testid="nav-user-name"]')
                    session_data['user_name'] = user_name
                except:
                    pass
            
            return session_data
            
        except:
            return {}
    
    async def restore_session(self, session_id: str, page: Page) -> bool:
        """Restore a persisted session"""
        if session_id not in self.sessions:
            # Try to load from persistence
            loaded = await self._load_session_from_persistence(session_id)
            if not loaded:
                return False
        
        session_info = self.sessions[session_id]
        
        try:
            # Check if session is still valid
            if not self._is_session_valid(session_info):
                # Try to renew session
                renewed = await self._renew_session(session_info)
                if not renewed:
                    return False
            
            # Restore cookies
            if session_info.cookies:
                await self._restore_cookies(page, session_info.cookies)
            
            # Restore tokens
            if session_info.tokens:
                await self._restore_tokens(page, session_info.tokens)
            
            # Update session info
            session_info.page_context = page
            session_info.last_activity = time.time()
            session_info.state = SessionState.ACTIVE
            
            logger.info(f"Session restored successfully: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return False
    
    async def _restore_cookies(self, page: Page, cookies: Dict[str, Any]):
        """Restore cookies to the page context"""
        try:
            cookie_list = []
            for name, value in cookies.items():
                cookie_list.append({
                    'name': name,
                    'value': str(value),
                    'domain': page.url.split('/')[2] if '/' in page.url else 'localhost',
                    'path': '/'
                })
            
            await page.context.add_cookies(cookie_list)
            
        except Exception as e:
            logger.warning(f"Failed to restore cookies: {e}")
    
    async def _restore_tokens(self, page: Page, tokens: Dict[str, str]):
        """Restore tokens to localStorage/sessionStorage"""
        try:
            await page.evaluate("""(tokens) => {
                for (const [key, value] of Object.entries(tokens)) {
                    try {
                        localStorage.setItem(key, value);
                    } catch (e) {
                        sessionStorage.setItem(key, value);
                    }
                }
            }""", tokens)
            
        except Exception as e:
            logger.warning(f"Failed to restore tokens: {e}")
    
    def _is_session_valid(self, session_info: SessionInfo) -> bool:
        """Check if a session is still valid"""
        current_time = time.time()
        
        # Check if session has expired
        if current_time > session_info.expires_at:
            return False
        
        # Check if session state is active
        if session_info.state != SessionState.ACTIVE:
            return False
        
        return True
    
    async def _renew_session(self, session_info: SessionInfo) -> bool:
        """Attempt to renew an expired session"""
        try:
            session_info.state = SessionState.RENEWING
            
            # Platform-specific renewal logic
            if session_info.platform in ['amazon', 'facebook']:
                # These platforms typically auto-renew on activity
                session_info.expires_at = time.time() + self.platform_configs[session_info.platform]['session_duration']
                session_info.state = SessionState.ACTIVE
                return True
            
            # For other platforms, renewal might require re-authentication
            return False
            
        except Exception as e:
            logger.error(f"Session renewal failed: {e}")
            session_info.state = SessionState.EXPIRED
            return False
    
    async def _persist_session(self, session_info: SessionInfo):
        """Persist session to storage"""
        try:
            # Create a serializable copy
            session_data = {
                'session_id': session_info.session_id,
                'platform': session_info.platform,
                'user_id': session_info.user_id,
                'state': session_info.state.value,
                'auth_method': session_info.auth_method.value,
                'created_at': session_info.created_at,
                'last_activity': session_info.last_activity,
                'expires_at': session_info.expires_at,
                'cookies': session_info.cookies,
                'tokens': session_info.tokens,
                'session_data': session_info.session_data
            }
            
            # Encrypt sensitive data
            encrypted_data = self._encrypt_data(session_data)
            
            # Save to file
            session_file = os.path.join(self.storage_path, f"{session_info.session_id}.session")
            with open(session_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")
    
    async def _load_session_from_persistence(self, session_id: str) -> bool:
        """Load session from persistent storage"""
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.session")
            
            if not os.path.exists(session_file):
                return False
            
            with open(session_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            session_data = self._decrypt_data(encrypted_data)
            
            # Recreate session info
            session_info = SessionInfo(
                session_id=session_data['session_id'],
                platform=session_data['platform'],
                user_id=session_data['user_id'],
                state=SessionState(session_data['state']),
                auth_method=AuthMethod(session_data['auth_method']),
                created_at=session_data['created_at'],
                last_activity=session_data['last_activity'],
                expires_at=session_data['expires_at'],
                cookies=session_data['cookies'],
                tokens=session_data['tokens'],
                session_data=session_data['session_data']
            )
            
            self.sessions[session_id] = session_info
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session from persistence: {e}")
            return False
    
    def _load_persisted_sessions(self):
        """Load all persisted sessions on startup"""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.session'):
                    session_id = filename[:-8]  # Remove .session extension
                    asyncio.create_task(self._load_session_from_persistence(session_id))
        except:
            pass
    
    def _encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt sensitive session data"""
        try:
            # Simple encryption using base64 and XOR (for demo purposes)
            # In production, use proper encryption like AES
            json_data = json.dumps(data).encode('utf-8')
            
            encrypted = bytearray()
            key_bytes = self.encryption_key
            
            for i, byte in enumerate(json_data):
                encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            return base64.b64encode(bytes(encrypted))
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return b''
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt session data"""
        try:
            decoded_data = base64.b64decode(encrypted_data)
            
            decrypted = bytearray()
            key_bytes = self.encryption_key
            
            for i, byte in enumerate(decoded_data):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            json_str = bytes(decrypted).decode('utf-8')
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return {}
    
    def _generate_session_id(self, platform: str, user_id: str) -> str:
        """Generate a unique session ID"""
        timestamp = str(time.time())
        data = f"{platform}:{user_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """Get all active sessions"""
        return [session for session in self.sessions.values() 
                if session.state == SessionState.ACTIVE]
    
    async def close_session(self, session_id: str):
        """Close and clean up a session"""
        if session_id in self.sessions:
            session_info = self.sessions[session_id]
            session_info.state = SessionState.INACTIVE
            
            # Clean up resources
            if session_info.page_context:
                try:
                    await session_info.page_context.close()
                except:
                    pass
            
            # Remove from memory
            del self.sessions[session_id]
            
            # Remove from persistence
            session_file = os.path.join(self.storage_path, f"{session_id}.session")
            try:
                os.remove(session_file)
            except:
                pass
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_info in self.sessions.items():
            if current_time > session_info.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id)
        
        return len(expired_sessions)

# Global session manager instance
_global_session_manager: Optional[SessionManager] = None

def get_session_manager(storage_path: str = "sessions") -> SessionManager:
    """Get or create the global session manager"""
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = SessionManager(storage_path)
    
    return _global_session_manager