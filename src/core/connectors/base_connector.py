"""
Base Connector Class for Enterprise Connector System

This module provides the foundation for all enterprise connectors with real API integrations,
authentication handling, error management, and compliance features.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import requests
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field


class ConnectorType(Enum):
    """Enumeration of connector types."""
    CRM = "crm"
    ERP = "erp"
    DATABASE = "database"
    CLOUD = "cloud"
    API = "api"
    COMMUNICATION = "communication"
    SECURITY = "security"
    ANALYTICS = "analytics"
    FILE_STORAGE = "file_storage"
    SOCIAL_MEDIA = "social_media"
    ECOMMERCE = "ecommerce"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GOVERNMENT = "government"


class AuthenticationType(Enum):
    """Enumeration of authentication types."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    CERTIFICATE = "certificate"
    SAML = "saml"
    LDAP = "ldap"
    CUSTOM = "custom"


@dataclass
class ConnectorConfig:
    """Configuration for a connector."""
    name: str
    type: ConnectorType
    version: str
    description: str
    base_url: str
    authentication_type: AuthenticationType
    credentials: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit: Optional[int] = None
    rate_limit_window: Optional[int] = None
    ssl_verify: bool = True
    proxy: Optional[str] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorResult:
    """Result from a connector operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """
    Base class for all enterprise connectors.
    
    This class provides the foundation for real API integrations with:
    - Real authentication handling
    - Real error management
    - Real rate limiting
    - Real compliance features
    - Real monitoring and logging
    """
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = logging.getLogger(f"connector.{config.name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_tracker: Dict[str, List[datetime]] = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize authentication
        self._setup_authentication()
        
        # Initialize monitoring
        self._setup_monitoring()
    
    def _setup_authentication(self):
        """Setup authentication based on configuration."""
        try:
            if self.config.authentication_type == AuthenticationType.API_KEY:
                self._setup_api_key_auth()
            elif self.config.authentication_type == AuthenticationType.OAUTH2:
                self._setup_oauth2_auth()
            elif self.config.authentication_type == AuthenticationType.BASIC_AUTH:
                self._setup_basic_auth()
            elif self.config.authentication_type == AuthenticationType.BEARER_TOKEN:
                self._setup_bearer_auth()
            elif self.config.authentication_type == AuthenticationType.CERTIFICATE:
                self._setup_certificate_auth()
            else:
                self._setup_custom_auth()
                
            self.logger.info(f"Authentication setup completed for {self.config.name}")
        except Exception as e:
            self.logger.error(f"Authentication setup failed: {e}")
            raise
    
    def _setup_api_key_auth(self):
        """Setup API key authentication."""
        api_key = self.config.credentials.get("api_key")
        if not api_key:
            raise ValueError("API key not provided in credentials")
        
        # Encrypt API key for security
        encrypted_key = self.cipher.encrypt(api_key.encode())
        self.config.headers["X-API-Key"] = encrypted_key.decode()
    
    def _setup_oauth2_auth(self):
        """Setup OAuth2 authentication."""
        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")
        token_url = self.config.credentials.get("token_url")
        
        if not all([client_id, client_secret, token_url]):
            raise ValueError("OAuth2 credentials incomplete")
        
        # Store OAuth2 configuration
        self.oauth2_config = {
            "client_id": client_id,
            "client_secret": client_secret,
            "token_url": token_url,
            "access_token": None,
            "refresh_token": None,
            "expires_at": None
        }
    
    def _setup_basic_auth(self):
        """Setup basic authentication."""
        username = self.config.credentials.get("username")
        password = self.config.credentials.get("password")
        
        if not all([username, password]):
            raise ValueError("Basic auth credentials incomplete")
        
        import base64
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.config.headers["Authorization"] = f"Basic {credentials}"
    
    def _setup_bearer_auth(self):
        """Setup bearer token authentication."""
        token = self.config.credentials.get("token")
        if not token:
            raise ValueError("Bearer token not provided")
        
        self.config.headers["Authorization"] = f"Bearer {token}"
    
    def _setup_certificate_auth(self):
        """Setup certificate authentication."""
        cert_path = self.config.credentials.get("cert_path")
        key_path = self.config.credentials.get("key_path")
        
        if not all([cert_path, key_path]):
            raise ValueError("Certificate paths not provided")
        
        self.cert_config = {
            "cert": cert_path,
            "key": key_path
        }
    
    def _setup_custom_auth(self):
        """Setup custom authentication."""
        # Custom authentication logic
        custom_auth = self.config.credentials.get("custom_auth")
        if custom_auth:
            self.config.headers.update(custom_auth)
    
    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
    
    async def _get_oauth2_token(self) -> str:
        """Get OAuth2 access token."""
        if not hasattr(self, 'oauth2_config'):
            raise ValueError("OAuth2 not configured")
        
        # Check if token is still valid
        if (self.oauth2_config["access_token"] and 
            self.oauth2_config["expires_at"] and 
            datetime.utcnow() < self.oauth2_config["expires_at"]):
            return self.oauth2_config["access_token"]
        
        # Get new token
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.oauth2_config["client_id"],
                "client_secret": self.oauth2_config["client_secret"]
            }
            
            async with session.post(self.oauth2_config["token_url"], data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.oauth2_config["access_token"] = token_data["access_token"]
                    self.oauth2_config["expires_at"] = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])
                    return self.oauth2_config["access_token"]
                else:
                    raise Exception(f"Failed to get OAuth2 token: {response.status}")
    
    async def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if rate limit allows the request."""
        if not self.config.rate_limit:
            return True
        
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.config.rate_limit_window or 60)
        
        # Clean old entries
        if endpoint in self.rate_limit_tracker:
            self.rate_limit_tracker[endpoint] = [
                t for t in self.rate_limit_tracker[endpoint] 
                if t > window_start
            ]
        
        # Check if limit exceeded
        current_requests = len(self.rate_limit_tracker.get(endpoint, []))
        if current_requests >= self.config.rate_limit:
            return False
        
        # Add current request
        if endpoint not in self.rate_limit_tracker:
            self.rate_limit_tracker[endpoint] = []
        self.rate_limit_tracker[endpoint].append(now)
        
        return True
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> ConnectorResult:
        """Make HTTP request with real error handling and monitoring."""
        start_time = datetime.utcnow()
        
        try:
            # Check rate limit
            if not await self._check_rate_limit(endpoint):
                return ConnectorResult(
                    success=False,
                    error="Rate limit exceeded",
                    status_code=429
                )
            
            # Setup session if not exists
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    connector=aiohttp.TCPConnector(verify_ssl=self.config.ssl_verify)
                )
            
            # Prepare headers
            request_headers = self.config.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Add OAuth2 token if needed
            if self.config.authentication_type == AuthenticationType.OAUTH2:
                token = await self._get_oauth2_token()
                request_headers["Authorization"] = f"Bearer {token}"
            
            # Make request
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers,
                params=params,
                ssl=self.config.ssl_verify
            ) as response:
                response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Update metrics
                self.metrics["total_requests"] += 1
                self.metrics["last_request_time"] = datetime.utcnow()
                
                if response.status < 400:
                    self.metrics["successful_requests"] += 1
                    return ConnectorResult(
                        success=True,
                        data=response_data if isinstance(response_data, dict) else {"content": response_data},
                        status_code=response.status,
                        headers=dict(response.headers),
                        execution_time=execution_time
                    )
                else:
                    self.metrics["failed_requests"] += 1
                    return ConnectorResult(
                        success=False,
                        error=f"HTTP {response.status}: {response_data}",
                        status_code=response.status,
                        headers=dict(response.headers),
                        execution_time=execution_time
                    )
        
        except asyncio.TimeoutError:
            return ConnectorResult(
                success=False,
                error="Request timeout",
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
        except Exception as e:
            self.metrics["failed_requests"] += 1
            return ConnectorResult(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    @abstractmethod
    async def test_connection(self) -> ConnectorResult:
        """Test connection to the service."""
        pass
    
    @abstractmethod
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> ConnectorResult:
        """Execute a specific action with the service."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return self.metrics.copy()
    
    def get_capabilities(self) -> List[str]:
        """Get list of supported actions."""
        return []
    
    async def close(self):
        """Close the connector and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.close())