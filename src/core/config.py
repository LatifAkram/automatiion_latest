"""
Configuration Management
=======================

Configuration management for the multi-agent automation platform.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    db_path: str = Field(default="data/automation.db", env="DB_PATH")
    vector_db_path: str = Field(default="data/vector_db", env="VECTOR_DB_PATH")
    media_path: str = Field(default="data/media", env="MEDIA_PATH")
    
    # SQLite settings
    timeout: int = Field(default=30, env="DB_TIMEOUT")
    check_same_thread: bool = Field(default=False, env="DB_CHECK_SAME_THREAD")
    
    class Config:
        env_file = ".env"


class AIConfig(BaseSettings):
    """AI provider configuration."""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=2000, env="ANTHROPIC_MAX_TOKENS")
    anthropic_temperature: float = Field(default=0.7, env="ANTHROPIC_TEMPERATURE")
    
    # Google
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-pro", env="GOOGLE_MODEL")
    google_max_tokens: int = Field(default=2000, env="GOOGLE_MAX_TOKENS")
    google_temperature: float = Field(default=0.7, env="GOOGLE_TEMPERATURE")
    
    # Local LLM
    local_llm_url: str = Field(default="http://127.0.0.1:1234", env="LOCAL_LLM_URL")
    local_llm_model: str = Field(default="deepseek-coder", env="LOCAL_LLM_MODEL")
    local_llm_max_tokens: int = Field(default=2000, env="LOCAL_LLM_MAX_TOKENS")
    local_llm_temperature: float = Field(default=0.7, env="LOCAL_LLM_TEMPERATURE")
    
    class Config:
        env_file = ".env"


class SearchConfig(BaseSettings):
    """Search configuration."""
    
    # Google Search
    google_search_api_key: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_API_KEY")
    google_search_cx: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_CX")
    
    # Bing Search
    bing_search_api_key: Optional[str] = Field(default=None, env="BING_SEARCH_API_KEY")
    bing_search_endpoint: str = Field(default="https://api.bing.microsoft.com/v7.0/search", env="BING_SEARCH_ENDPOINT")
    
    # GitHub
    github_token: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    
    # Reddit
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="MultiAgentAutomation/1.0", env="REDDIT_USER_AGENT")
    
    # YouTube
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    
    # Rate limiting
    search_rate_limit: int = Field(default=100, env="SEARCH_RATE_LIMIT")
    search_rate_limit_window: int = Field(default=3600, env="SEARCH_RATE_LIMIT_WINDOW")
    
    class Config:
        env_file = ".env"


class AutomationConfig(BaseSettings):
    """Automation configuration."""
    
    # Browser settings
    browser_type: str = Field(default="chromium", env="BROWSER_TYPE")
    headless: bool = Field(default=True, env="HEADLESS")
    browser_args: List[str] = Field(default=["--no-sandbox", "--disable-dev-shm-usage"], env="BROWSER_ARGS")
    viewport_width: int = Field(default=1920, env="VIEWPORT_WIDTH")
    viewport_height: int = Field(default=1080, env="VIEWPORT_HEIGHT")
    user_agent: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", env="USER_AGENT")
    locale: str = Field(default="en-US", env="LOCALE")
    timezone: str = Field(default="America/New_York", env="TIMEZONE")
    
    # Execution settings
    max_parallel_workflows: int = Field(default=5, env="MAX_PARALLEL_WORKFLOWS")
    max_parallel_agents: int = Field(default=3, env="MAX_PARALLEL_AGENTS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")
    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")
    retry_delay: int = Field(default=5, env="RETRY_DELAY")
    
    # Selector drift detection
    enable_selector_drift_detection: bool = Field(default=True, env="ENABLE_SELECTOR_DRIFT_DETECTION")
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")
    confidence_threshold: float = Field(default=0.7, env="CONFIDENCE_THRESHOLD")
    
    # Media capture
    enable_screenshots: bool = Field(default=True, env="ENABLE_SCREENSHOTS")
    enable_video_recording: bool = Field(default=False, env="ENABLE_VIDEO_RECORDING")
    screenshot_quality: str = Field(default="high", env="SCREENSHOT_QUALITY")
    
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


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Encryption
    encryption_key: str = Field(default="your-encryption-key-here", env="ENCRYPTION_KEY")
    
    # PII Detection
    enable_pii_detection: bool = Field(default=True, env="ENABLE_PII_DETECTION")
    pii_masking_enabled: bool = Field(default=True, env="PII_MASKING_ENABLED")
    
    class Config:
        env_file = ".env"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"


class Config(BaseSettings):
    """Main configuration class."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Data paths
    data_path: str = Field(default="data", env="DATA_PATH")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    ai: AIConfig = AIConfig()
    search: SearchConfig = SearchConfig()
    automation: AutomationConfig = AutomationConfig()
    api: APIConfig = APIConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        env_file = ".env"
        case_sensitive = False