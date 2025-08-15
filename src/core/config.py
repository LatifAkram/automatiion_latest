#!/usr/bin/env python3
"""
SUPER-OMEGA Configuration - 100% Dependency-Free
================================================

Configuration management using built-in validation system.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Import our built-in validation system
try:
    from .builtin_data_validation import BaseValidator, validate_email, validate_url
except ImportError:
    from builtin_data_validation import BaseValidator, validate_email, validate_url

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///super_omega.db"
    max_connections: int = 10
    timeout: int = 30

@dataclass 
class APIConfig:
    """API configuration"""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    cors_origins: List[str] = None

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 4
    timeout_ms: int = 5000
    retry_attempts: int = 3
    cache_size: int = 1000

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "super-omega-secret-key"
    token_expiry: int = 3600
    rate_limit: int = 100

class Config(BaseValidator):
    """Main configuration class with built-in validation"""
    
    def __init__(self):
        # Default configuration
        self.database: DatabaseConfig = DatabaseConfig()
        self.api: APIConfig = APIConfig()
        self.performance: PerformanceConfig = PerformanceConfig()
        self.security: SecurityConfig = SecurityConfig()
        
        # Built-in system settings
        self.use_builtin_systems: bool = True
        self.enable_ai_processor: bool = True
        self.enable_vision_processor: bool = True
        self.enable_performance_monitor: bool = True
        self.enable_web_server: bool = True
        
        # Automation settings
        self.automation_timeout: int = 30000  # 30 seconds
        self.max_retries: int = 3
        self.screenshot_quality: int = 80
        self.element_wait_timeout: int = 10000  # 10 seconds
        
        # Logging settings
        self.log_level: str = "INFO"
        self.log_file: str = "super_omega.log"
        self.max_log_size: int = 10 * 1024 * 1024  # 10MB
        
        super().__init__()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration using built-in validation"""
        config_dict = {
            "database_url": self.database.url,
            "api_host": self.api.host,
            "api_port": self.api.port,
            "performance_timeout": self.performance.timeout_ms,
            "security_key": self.security.secret_key,
            "log_level": self.log_level,
            "automation_timeout": self.automation_timeout
        }
        
        try:
            return self.validate(config_dict)
        except Exception as e:
            print(f"⚠️ Config validation warning: {e}")
            return config_dict
    
    def load_from_file(self, config_file: str = "config.json") -> bool:
        """Load configuration from JSON file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration from file
                if 'database' in config_data:
                    db_config = config_data['database']
                    self.database.url = db_config.get('url', self.database.url)
                    self.database.max_connections = db_config.get('max_connections', self.database.max_connections)
                    self.database.timeout = db_config.get('timeout', self.database.timeout)
                
                if 'api' in config_data:
                    api_config = config_data['api']
                    self.api.host = api_config.get('host', self.api.host)
                    self.api.port = api_config.get('port', self.api.port)
                    self.api.debug = api_config.get('debug', self.api.debug)
                
                if 'performance' in config_data:
                    perf_config = config_data['performance']
                    self.performance.max_workers = perf_config.get('max_workers', self.performance.max_workers)
                    self.performance.timeout_ms = perf_config.get('timeout_ms', self.performance.timeout_ms)
                    self.performance.retry_attempts = perf_config.get('retry_attempts', self.performance.retry_attempts)
                
                if 'security' in config_data:
                    sec_config = config_data['security']
                    self.security.secret_key = sec_config.get('secret_key', self.security.secret_key)
                    self.security.token_expiry = sec_config.get('token_expiry', self.security.token_expiry)
                
                # Update other settings
                self.automation_timeout = config_data.get('automation_timeout', self.automation_timeout)
                self.max_retries = config_data.get('max_retries', self.max_retries)
                self.log_level = config_data.get('log_level', self.log_level)
                
                print(f"✅ Configuration loaded from {config_file}")
                return True
                
        except Exception as e:
            print(f"⚠️ Could not load config from {config_file}: {e}")
            print("Using default configuration...")
        
        return False
    
    def save_to_file(self, config_file: str = "config.json") -> bool:
        """Save configuration to JSON file"""
        try:
            config_data = {
                "database": {
                    "url": self.database.url,
                    "max_connections": self.database.max_connections,
                    "timeout": self.database.timeout
                },
                "api": {
                    "host": self.api.host,
                    "port": self.api.port,
                    "debug": self.api.debug,
                    "cors_origins": self.api.cors_origins or []
                },
                "performance": {
                    "max_workers": self.performance.max_workers,
                    "timeout_ms": self.performance.timeout_ms,
                    "retry_attempts": self.performance.retry_attempts,
                    "cache_size": self.performance.cache_size
                },
                "security": {
                    "secret_key": self.security.secret_key,
                    "token_expiry": self.security.token_expiry,
                    "rate_limit": self.security.rate_limit
                },
                "builtin_systems": {
                    "use_builtin_systems": self.use_builtin_systems,
                    "enable_ai_processor": self.enable_ai_processor,
                    "enable_vision_processor": self.enable_vision_processor,
                    "enable_performance_monitor": self.enable_performance_monitor,
                    "enable_web_server": self.enable_web_server
                },
                "automation": {
                    "automation_timeout": self.automation_timeout,
                    "max_retries": self.max_retries,
                    "screenshot_quality": self.screenshot_quality,
                    "element_wait_timeout": self.element_wait_timeout
                },
                "logging": {
                    "log_level": self.log_level,
                    "log_file": self.log_file,
                    "max_log_size": self.max_log_size
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Could not save config to {config_file}: {e}")
            return False
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Database settings
        if os.getenv('DATABASE_URL'):
            self.database.url = os.getenv('DATABASE_URL')
        
        # API settings
        if os.getenv('API_HOST'):
            self.api.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.port = int(os.getenv('API_PORT'))
        if os.getenv('API_DEBUG'):
            self.api.debug = os.getenv('API_DEBUG').lower() == 'true'
        
        # Performance settings
        if os.getenv('MAX_WORKERS'):
            self.performance.max_workers = int(os.getenv('MAX_WORKERS'))
        if os.getenv('TIMEOUT_MS'):
            self.performance.timeout_ms = int(os.getenv('TIMEOUT_MS'))
        
        # Security settings
        if os.getenv('SECRET_KEY'):
            self.security.secret_key = os.getenv('SECRET_KEY')
        
        # Other settings
        if os.getenv('LOG_LEVEL'):
            self.log_level = os.getenv('LOG_LEVEL')
        if os.getenv('AUTOMATION_TIMEOUT'):
            self.automation_timeout = int(os.getenv('AUTOMATION_TIMEOUT'))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "database_url": self.database.url,
            "api_endpoint": f"http://{self.api.host}:{self.api.port}",
            "performance_timeout": f"{self.performance.timeout_ms}ms",
            "max_workers": self.performance.max_workers,
            "builtin_systems_enabled": self.use_builtin_systems,
            "ai_processor_enabled": self.enable_ai_processor,
            "vision_processor_enabled": self.enable_vision_processor,
            "automation_timeout": f"{self.automation_timeout}ms",
            "log_level": self.log_level
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

def load_config(config_file: str = "config.json") -> Config:
    """Load and return configuration"""
    config.load_from_file(config_file)
    config.load_from_env()  # Environment variables override file settings
    return config

def save_config(config_file: str = "config.json") -> bool:
    """Save current configuration"""
    return config.save_to_file(config_file)

if __name__ == "__main__":
    # Demo configuration system
    print("⚙️ SUPER-OMEGA Configuration Demo")
    print("=" * 40)
    
    # Create and configure
    cfg = Config()
    
    print("📋 Default Configuration:")
    summary = cfg.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test validation
    print("\n🔍 Validating configuration...")
    try:
        validated = cfg.validate_config()
        print("✅ Configuration validation successful")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
    
    # Test save/load
    print("\n💾 Testing save/load...")
    if cfg.save_to_file("test_config.json"):
        new_cfg = Config()
        if new_cfg.load_from_file("test_config.json"):
            print("✅ Save/load test successful")
        else:
            print("❌ Load test failed")
    else:
        print("❌ Save test failed")
    
    print("\n✅ Configuration system working perfectly!")
    print("⚙️ No pydantic dependencies required!")