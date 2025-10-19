"""
Base Data Collector Interface

This module defines the abstract base class for all data collectors.
It provides a consistent interface for collecting market data from different sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import asyncio
import logging
from pydantic import BaseModel

from backend.core.exceptions import DataCollectionError, DataValidationError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class DataType(Enum):
    """Types of market data that can be collected."""
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TICK = "tick"     # Tick-by-tick data
    QUOTE = "quote"   # Bid/Ask quotes
    NEWS = "news"     # News articles
    SENTIMENT = "sentiment"  # Sentiment data
    FUNDAMENTAL = "fundamental"  # Fundamental data
    OPTIONS = "options"  # Options data
    ORDER_BOOK = "order_book"  # Order book data


class DataFrequency(Enum):
    """Data frequency intervals."""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    frequency: DataFrequency
    data: Dict[str, Any]
    source: str
    quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.symbol:
            raise DataValidationError("Symbol cannot be empty", self.symbol)
        if not isinstance(self.timestamp, datetime):
            raise DataValidationError("Timestamp must be datetime", self.symbol)
        if not 0 <= self.quality_score <= 1:
            raise DataValidationError("Quality score must be between 0 and 1", self.symbol)


@dataclass
class OHLCVData:
    """OHLCV (Open, High, Low, Close, Volume) data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    frequency: DataFrequency
    source: str
    adjusted_close: Optional[float] = None
    
    def __post_init__(self):
        """Validate OHLCV data."""
        if self.high < max(self.open, self.close):
            raise DataValidationError("High cannot be lower than open or close", self.symbol)
        if self.low > min(self.open, self.close):
            raise DataValidationError("Low cannot be higher than open or close", self.symbol)
        if self.volume < 0:
            raise DataValidationError("Volume cannot be negative", self.symbol)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "frequency": self.frequency.value,
            "source": self.source,
            "adjusted_close": self.adjusted_close
        }


@dataclass
class QuoteData:
    """Quote (bid/ask) data structure."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    source: str
    
    def __post_init__(self):
        """Validate quote data."""
        if self.bid_price >= self.ask_price:
            raise DataValidationError("Bid price must be lower than ask price", self.symbol)
        if self.bid_size < 0 or self.ask_size < 0:
            raise DataValidationError("Bid/ask sizes cannot be negative", self.symbol)
    
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2


@dataclass
class NewsData:
    """News data structure."""
    symbol: Optional[str]  # None if general market news
    timestamp: datetime
    title: str
    content: str
    source: str
    url: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate news data."""
        if not self.title.strip():
            raise DataValidationError("Title cannot be empty", self.symbol or "general")
        if not self.content.strip():
            raise DataValidationError("Content cannot be empty", self.symbol or "general")


class BaseDataCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    This class defines the interface that all data collectors must implement,
    ensuring consistency across different data sources.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the data collector.
        
        Args:
            name: Name of the collector
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = get_logger(f"collector.{name}")
        self.is_running = False
        self.rate_limiter = RateLimiter(config.get("rate_limit", 100))
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the collector (connect to APIs, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1
    ) -> List[OHLCVData]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
        """
        pass
    
    @abstractmethod
    async def get_real_time_data(
        self,
        symbol: str
    ) -> AsyncGenerator[MarketData, None]:
        """
        Get real-time market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Yields:
            Real-time market data
        """
        pass
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by this collector.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is valid and supported
        """
        try:
            # Try to get a small amount of recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            data = await self.get_historical_data(
                symbol, start_date, end_date, DataFrequency.DAY_1
            )
            return len(data) > 0
            
        except Exception as e:
            self.logger.warning(f"Symbol validation failed for {symbol}: {e}")
            return False
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from this collector.
        
        Returns:
            List of available symbols
        """
        # Default implementation - should be overridden by subclasses
        return []
    
    def get_supported_frequencies(self) -> List[DataFrequency]:
        """
        Get list of supported data frequencies.
        
        Returns:
            List of supported frequencies
        """
        # Default implementation - should be overridden by subclasses
        return [
            DataFrequency.MINUTE_1,
            DataFrequency.MINUTE_5,
            DataFrequency.MINUTE_15,
            DataFrequency.MINUTE_30,
            DataFrequency.HOUR_1,
            DataFrequency.DAY_1
        ]
    
    async def start_collection(self, symbols: List[str]) -> None:
        """
        Start collecting data for the given symbols.
        
        Args:
            symbols: List of symbols to collect data for
        """
        if self.is_running:
            self.logger.warning("Collection is already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        try:
            await self._start_collection_loop(symbols)
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            raise DataCollectionError(f"Failed to start collection: {e}", self.name)
        finally:
            self.is_running = False
    
    async def stop_collection(self) -> None:
        """Stop data collection."""
        self.is_running = False
        self.logger.info("Stopping data collection")
    
    async def _start_collection_loop(self, symbols: List[str]) -> None:
        """
        Internal method for the collection loop.
        
        Args:
            symbols: List of symbols to collect data for
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _start_collection_loop")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to API calls."""
        await self.rate_limiter.acquire()


class RateLimiter:
    """
    Simple rate limiter for API calls.
    """
    
    def __init__(self, max_requests_per_second: int):
        self.max_requests = max_requests_per_second
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self.lock:
            now = datetime.now()
            # Remove old requests (older than 1 second)
            self.requests = [
                req_time for req_time in self.requests
                if (now - req_time).total_seconds() < 1
            ]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = 1.0 - (now - oldest_request).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Add current request
            self.requests.append(now)


class DataCollectorRegistry:
    """
    Registry for managing multiple data collectors.
    """
    
    def __init__(self):
        self.collectors: Dict[str, BaseDataCollector] = {}
        self.logger = get_logger("collector_registry")
    
    def register_collector(self, collector: BaseDataCollector) -> None:
        """
        Register a data collector.
        
        Args:
            collector: Data collector instance
        """
        self.collectors[collector.name] = collector
        self.logger.info(f"Registered data collector: {collector.name}")
    
    def get_collector(self, name: str) -> Optional[BaseDataCollector]:
        """
        Get a data collector by name.
        
        Args:
            name: Collector name
            
        Returns:
            Data collector instance or None
        """
        return self.collectors.get(name)
    
    def get_collectors_by_type(self, data_type: DataType) -> List[BaseDataCollector]:
        """
        Get collectors that support a specific data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            List of collectors that support the data type
        """
        # This is a simplified implementation
        # In practice, collectors would declare their supported data types
        return list(self.collectors.values())
    
    async def initialize_all(self) -> None:
        """Initialize all registered collectors."""
        for collector in self.collectors.values():
            try:
                await collector.initialize()
                self.logger.info(f"Initialized collector: {collector.name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize collector {collector.name}: {e}")
                raise
    
    async def cleanup_all(self) -> None:
        """Cleanup all registered collectors."""
        for collector in self.collectors.values():
            try:
                await collector.cleanup()
                self.logger.info(f"Cleaned up collector: {collector.name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup collector {collector.name}: {e}")


# Global registry instance
collector_registry = DataCollectorRegistry()