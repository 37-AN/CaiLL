"""
Configuration settings for the AI Trading System.

This module handles all configuration management using environment variables
and provides type-safe configuration access throughout the application.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """
    Application settings using Pydantic for validation and type safety.
    """
    
    # Application Settings
    APP_NAME: str = "AI Trading System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = ""
    INFLUXDB_ORG: str = "trading_org"
    INFLUXDB_BUCKET: str = "trading_data"
    
    # Vector Database Settings
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    
    # Market Data API Keys
    ALPHA_VANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    YAHOO_FINANCE_API_KEY: str = ""
    
    # Trading Broker API Keys
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    INTERACTIVE_BROKERS_HOST: str = "localhost"
    INTERACTIVE_BROKERS_PORT: int = 7497
    INTERACTIVE_BROKERS_CLIENT_ID: int = 1
    
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET_KEY: str = ""
    BINANCE_TESTNET: bool = True
    
    # News & Sentiment API Keys
    NEWS_API_KEY: str = ""
    TWITTER_API_KEY: str = ""
    TWITTER_API_SECRET: str = ""
    TWITTER_ACCESS_TOKEN: str = ""
    TWITTER_ACCESS_TOKEN_SECRET: str = ""
    
    # Authentication Settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Trading Configuration
    PAPER_TRADING: bool = True
    LIVE_TRADING: bool = False
    ENABLE_TRADING: bool = False  # Master switch for trading
    TRADING_START_TIME: str = "09:30"
    TRADING_END_TIME: str = "16:00"
    TIMEZONE: str = "America/New_York"
    
    # Risk Management Settings
    MAX_PORTFOLIO_RISK: float = 0.02  # 2% max portfolio risk
    MAX_POSITION_SIZE: float = 0.10    # 10% max position size
    MAX_DRAWDOWN: float = 0.15         # 15% max drawdown
    MIN_WIN_RATE: float = 0.55         # 55% minimum win rate
    
    # Machine Learning Settings
    MODEL_RETRAIN_INTERVAL: int = 24   # hours
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPISODES: int = 1000
    
    # Monitoring Settings
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    SENTRY_DSN: Optional[str] = None
    
    # Notification Settings
    SLACK_WEBHOOK_URL: Optional[str] = None
    EMAIL_SMTP_HOST: str = "smtp.gmail.com"
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    # Development Settings
    DEVELOPMENT_MODE: bool = True
    TEST_DATA_PATH: str = "./data/test"
    PRODUCTION_DATA_PATH: str = "./data/production"
    
    # Security Settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    CORS_CREDENTIALS: bool = True
    
    # Performance Settings
    WORKER_PROCESSES: int = 4
    MAX_CONNECTIONS: int = 1000
    CONNECTION_TIMEOUT: int = 30
    
    # Feature Flags
    ENABLE_CRYPTO_TRADING: bool = True
    ENABLE_OPTIONS_TRADING: bool = True
    ENABLE_NEWS_SENTIMENT: bool = True
    ENABLE_EDUCATIONAL_CONTENT: bool = True
    ENABLE_BACKTESTING: bool = True
    
    # Cache Configuration
    CACHE_TTL: int = 300  # seconds
    CACHE_MAX_SIZE: int = 1000
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Backup Configuration
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL: int = 24  # hours
    BACKUP_RETENTION_DAYS: int = 30
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("MAX_PORTFOLIO_RISK", "MAX_POSITION_SIZE", "MAX_DRAWDOWN")
    def validate_risk_percentages(cls, v):
        """Validate risk management percentages."""
        if not 0 <= v <= 1:
            raise ValueError("Risk percentages must be between 0 and 1")
        return v
    
    @validator("MIN_WIN_RATE")
    def validate_win_rate(cls, v):
        """Validate minimum win rate."""
        if not 0 <= v <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Global settings instance
settings = get_settings()


def get_database_url() -> str:
    """Get database URL."""
    return settings.DATABASE_URL


def get_redis_url() -> str:
    """Get Redis URL."""
    return settings.REDIS_URL


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.DEVELOPMENT_MODE


def is_paper_trading() -> bool:
    """Check if paper trading mode is enabled."""
    return settings.PAPER_TRADING


def is_live_trading() -> bool:
    """Check if live trading is enabled."""
    return settings.LIVE_TRADING and not settings.PAPER_TRADING


def is_trading_enabled() -> bool:
    """Check if trading is enabled."""
    return settings.ENABLE_TRADING


def get_risk_limits() -> dict:
    """Get risk management limits."""
    return {
        "max_portfolio_risk": settings.MAX_PORTFOLIO_RISK,
        "max_position_size": settings.MAX_POSITION_SIZE,
        "max_drawdown": settings.MAX_DRAWDOWN,
        "min_win_rate": settings.MIN_WIN_RATE
    }


def get_trading_hours() -> dict:
    """Get trading hours configuration."""
    return {
        "start_time": settings.TRADING_START_TIME,
        "end_time": settings.TRADING_END_TIME,
        "timezone": settings.TIMEZONE
    }


def get_api_credentials() -> dict:
    """Get API credentials for external services."""
    return {
        "alpha_vantage": settings.ALPHA_VANTAGE_API_KEY,
        "polygon": settings.POLYGON_API_KEY,
        "alpaca": {
            "api_key": settings.ALPACA_API_KEY,
            "secret_key": settings.ALPACA_SECRET_KEY,
            "base_url": settings.ALPACA_BASE_URL
        },
        "binance": {
            "api_key": settings.BINANCE_API_KEY,
            "secret_key": settings.BINANCE_SECRET_KEY,
            "testnet": settings.BINANCE_TESTNET
        },
        "pinecone": {
            "api_key": settings.PINECONE_API_KEY,
            "environment": settings.PINECONE_ENVIRONMENT
        }
    }


def get_feature_flags() -> dict:
    """Get feature flag settings."""
    return {
        "crypto_trading": settings.ENABLE_CRYPTO_TRADING,
        "options_trading": settings.ENABLE_OPTIONS_TRADING,
        "news_sentiment": settings.ENABLE_NEWS_SENTIMENT,
        "educational_content": settings.ENABLE_EDUCATIONAL_CONTENT,
        "backtesting": settings.ENABLE_BACKTESTING
    }