"""
Configuration management for the automation platform.
Handles environment variables, AI model settings, and system configuration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AIConfig(BaseSettings):
    """Configuration for AI models and providers."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-2.0-flash-exp", env="GOOGLE_MODEL")
    
    # Local LLM Configuration
    local_llm_url: str = Field(default="http://127.0.0.1:1234", env="LOCAL_LLM_URL")
    local_llm_model: str = Field(default="deepseek-coder-v2-lite-instruct", env="LOCAL_LLM_MODEL")
    
    # Default AI provider (priority order)
    default_provider: str = Field(default="openai", env="DEFAULT_AI_PROVIDER")
    
    class Config:
        env_file = ".env"


class DatabaseConfig(BaseSettings):
    """Configuration for databases and storage."""
    
    # SQLite Configuration
    sqlite_path: str = Field(default="data/automation.db", env="SQLITE_PATH")
    
    # Vector Database Configuration
    vector_db_path: str = Field(default="data/vector_db", env="VECTOR_DB_PATH")
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")
    
    # Media Storage
    media_path: str = Field(default="data/media", env="MEDIA_PATH")
    
    class Config:
        env_file = ".env"


class SearchConfig(BaseSettings):
    """Configuration for search engines and data sources."""
    
    # Search API Keys
    google_search_api_key: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_API_KEY")
    google_search_cx: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_CX")
    bing_search_api_key: Optional[str] = Field(default=None, env="BING_SEARCH_API_KEY")
    
    # GitHub Configuration
    github_token: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    
    # Stack Overflow Configuration
    stack_overflow_key: Optional[str] = Field(default=None, env="STACK_OVERFLOW_KEY")
    
    # Search limits
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    search_timeout: int = Field(default=30, env="SEARCH_TIMEOUT")
    
    class Config:
        env_file = ".env"


class AutomationConfig(BaseSettings):
    """Configuration for automation engines."""
    
    # Browser Configuration
    browser_type: str = Field(default="chromium", env="BROWSER_TYPE")
    headless: bool = Field(default=True, env="HEADLESS")
    browser_timeout: int = Field(default=30000, env="BROWSER_TIMEOUT")
    
    # Parallel Execution
    max_parallel_agents: int = Field(default=5, env="MAX_PARALLEL_AGENTS")
    max_parallel_workflows: int = Field(default=3, env="MAX_PARALLEL_WORKFLOWS")
    
    # Screenshot and Video
    capture_screenshots: bool = Field(default=True, env="CAPTURE_SCREENSHOTS")
    capture_video: bool = Field(default=True, env="CAPTURE_VIDEO")
    video_quality: str = Field(default="medium", env="VIDEO_QUALITY")
    
    # Browser Configuration
    browser_args: List[str] = Field(default_factory=lambda: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-accelerated-2d-canvas",
        "--no-first-run",
        "--no-zygote",
        "--disable-gpu"
    ])
    viewport_width: int = Field(default=1920, env="VIEWPORT_WIDTH")
    viewport_height: int = Field(default=1080, env="VIEWPORT_HEIGHT")
    user_agent: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", env="USER_AGENT")
    locale: str = Field(default="en-US", env="LOCALE")
    timezone: str = Field(default="UTC", env="TIMEZONE")
    
    class Config:
        env_file = ".env"


class SecurityConfig(BaseSettings):
    """Configuration for security and compliance."""
    
    # Encryption
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    # Audit Logging
    audit_log_path: str = Field(default="data/audit.log", env="AUDIT_LOG_PATH")
    audit_retention_days: int = Field(default=365, env="AUDIT_RETENTION_DAYS")
    
    # PII Detection
    pii_detection_enabled: bool = Field(default=True, env="PII_DETECTION_ENABLED")
    pii_masking_enabled: bool = Field(default=True, env="PII_MASKING_ENABLED")
    
    # RBAC
    rbac_enabled: bool = Field(default=True, env="RBAC_ENABLED")
    
    class Config:
        env_file = ".env"


class APIConfig(BaseSettings):
    """Configuration for the API server."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    class Config:
        env_file = ".env"


class Config(BaseSettings):
    """Main configuration class that combines all sub-configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Paths
    data_path: str = Field(default="data", env="DATA_PATH")
    
    # Sub-configurations
    ai: AIConfig = AIConfig()
    database: DatabaseConfig = DatabaseConfig()
    search: SearchConfig = SearchConfig()
    automation: AutomationConfig = AutomationConfig()
    security: SecurityConfig = SecurityConfig()
    api: APIConfig = APIConfig()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            Path(self.database.sqlite_path).parent,
            Path(self.database.vector_db_path),
            Path(self.database.media_path),
            Path(self.security.audit_log_path).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"