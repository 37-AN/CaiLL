"""
Market Data Service

This service orchestrates all data collectors and provides a unified interface
for accessing market data across different asset classes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from backend.data_collectors.stock_collector import StockDataCollector
from backend.data_collectors.crypto_collector import CryptoDataCollector
from backend.data_collectors.news_collector import NewsDataCollector
from backend.data_collectors.base_collector import (
    MarketData, OHLCVData, NewsData, DataType, DataFrequency
)
from backend.core.config import settings
from backend.core.exceptions import DataCollectionError
from backend.core.logging import get_logger, trading_logger

logger = get_logger(__name__)


class MarketDataService:
    """
    Main market data service that manages all data collectors.
    
    This service provides a unified interface for collecting and accessing
    market data across stocks, cryptocurrencies, and news.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("market_data_service")
        self.is_initialized = False
        self.is_running = False
        
        # Initialize collectors
        self.stock_collector = StockDataCollector(self.config.get("stocks", {}))
        self.crypto_collector = CryptoDataCollector(self.config.get("crypto", {}))
        self.news_collector = NewsDataCollector(self.config.get("news", {}))
        
        # Data storage (in production, this would be databases)
        self.data_cache = {}
        self.active_subscriptions = {}
        
    async def initialize(self) -> None:
        """Initialize all data collectors."""
        try:
            self.logger.info("Initializing Market Data Service...")
            
            # Initialize all collectors
            await self.stock_collector.initialize()
            await self.crypto_collector.initialize()
            await self.news_collector.initialize()
            
            self.is_initialized = True
            self.logger.info("Market Data Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Market Data Service: {e}")
            raise DataCollectionError(f"Service initialization failed: {e}", "market_data_service")
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        try:
            self.logger.info("Cleaning up Market Data Service...")
            
            # Stop all collections
            await self.stop_all_collections()
            
            # Cleanup collectors
            await self.stock_collector.cleanup()
            await self.crypto_collector.cleanup()
            await self.news_collector.cleanup()
            
            self.is_initialized = False
            self.logger.info("Market Data Service cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1,
        asset_type: str = "stock",
        preferred_source: Optional[str] = None
    ) -> List[OHLCVData]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            asset_type: Type of asset (stock, crypto)
            preferred_source: Preferred data source
            
        Returns:
            List of OHLCV data points
        """
        if not self.is_initialized:
            raise DataCollectionError("Service not initialized", "market_data_service")
        
        try:
            self.logger.info(f"Getting historical data for {symbol} ({asset_type})")
            
            # Route to appropriate collector
            if asset_type.lower() == "stock":
                data = await self.stock_collector.get_historical_data(
                    symbol, start_date, end_date, frequency, preferred_source
                )
            elif asset_type.lower() == "crypto":
                data = await self.crypto_collector.get_historical_data(
                    symbol, start_date, end_date, frequency, preferred_source
                )
            else:
                raise DataCollectionError(f"Unsupported asset type: {asset_type}", "market_data_service")
            
            # Cache the data
            cache_key = f"{symbol}_{asset_type}_{frequency.value}_{start_date.date()}_{end_date.date()}"
            self.data_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(),
                "ttl": 3600  # 1 hour
            }
            
            trading_logger.data_quality(
                data_source=preferred_source or "auto",
                quality_score=1.0,
                issues=[]
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            raise
    
    async def get_real_time_data(
        self,
        symbol: str,
        asset_type: str = "stock"
    ) -> Optional[MarketData]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Trading symbol
            asset_type: Type of asset (stock, crypto)
            
        Returns:
            Current market data or None
        """
        if not self.is_initialized:
            raise DataCollectionError("Service not initialized", "market_data_service")
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{asset_type}_current"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if (datetime.now() - cached_data["timestamp"]).seconds < 60:  # 1 minute cache
                    return cached_data["data"]
            
            # Get fresh data
            if asset_type.lower() == "stock":
                # For stocks, we'll get the most recent data point
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                
                data = await self.stock_collector.get_historical_data(
                    symbol, start_date, end_date, DataFrequency.MINUTE_1
                )
                
                if data:
                    latest_data = data[-1]
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=latest_data.timestamp,
                        data_type=DataType.OHLCV,
                        frequency=DataFrequency.MINUTE_1,
                        data={
                            "open": latest_data.open,
                            "high": latest_data.high,
                            "low": latest_data.low,
                            "close": latest_data.close,
                            "volume": latest_data.volume
                        },
                        source=latest_data.source
                    )
                    
                    # Cache the result
                    self.data_cache[cache_key] = {
                        "data": market_data,
                        "timestamp": datetime.now(),
                        "ttl": 60  # 1 minute
                    }
                    
                    return market_data
                    
            elif asset_type.lower() == "crypto":
                # Similar implementation for crypto
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=1)
                
                data = await self.crypto_collector.get_historical_data(
                    symbol, start_date, end_date, DataFrequency.MINUTE_1
                )
                
                if data:
                    latest_data = data[-1]
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=latest_data.timestamp,
                        data_type=DataType.OHLCV,
                        frequency=DataFrequency.MINUTE_1,
                        data={
                            "open": latest_data.open,
                            "high": latest_data.high,
                            "low": latest_data.low,
                            "close": latest_data.close,
                            "volume": latest_data.volume
                        },
                        source=latest_data.source
                    )
                    
                    # Cache the result
                    self.data_cache[cache_key] = {
                        "data": market_data,
                        "timestamp": datetime.now(),
                        "ttl": 60  # 1 minute
                    }
                    
                    return market_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {e}")
            return None
    
    async def get_news(
        self,
        query: str = "stocks OR trading OR finance OR market",
        symbol: Optional[str] = None,
        hours: int = 24,
        preferred_source: Optional[str] = None
    ) -> List[NewsData]:
        """
        Get news articles.
        
        Args:
            query: Search query
            symbol: Specific symbol to search for
            hours: Number of hours to look back
            preferred_source: Preferred news source
            
        Returns:
            List of news articles
        """
        if not self.is_initialized:
            raise DataCollectionError("Service not initialized", "market_data_service")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours)
            
            # If symbol is provided, search for company-specific news
            if symbol:
                news = await self.news_collector.get_company_news(
                    symbol, start_date, end_date
                )
            else:
                news = await self.news_collector.get_news(
                    query, start_date, end_date
                )
            
            # Filter by relevance and sentiment
            filtered_news = [
                article for article in news
                if (article.relevance_score or 0) > 0.3  # Minimum relevance threshold
            ]
            
            # Sort by relevance and timestamp
            filtered_news.sort(
                key=lambda x: (x.relevance_score or 0, x.timestamp),
                reverse=True
            )
            
            return filtered_news[:50]  # Return top 50 articles
            
        except Exception as e:
            self.logger.error(f"Error getting news: {e}")
            return []
    
    async def get_sentiment_analysis(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Number of hours to analyze
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Get news for the symbol
            news = await self.get_news(symbol=symbol, hours=hours)
            
            if not news:
                return {
                    "symbol": symbol,
                    "period_hours": hours,
                    "overall_sentiment": 0.0,
                    "sentiment_trend": "neutral",
                    "news_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "last_updated": datetime.now().isoformat()
                }
            
            # Calculate sentiment metrics
            sentiments = [article.sentiment_score or 0 for article in news]
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Determine trend
            if overall_sentiment > 0.2:
                trend = "positive"
            elif overall_sentiment < -0.2:
                trend = "negative"
            else:
                trend = "neutral"
            
            return {
                "symbol": symbol,
                "period_hours": hours,
                "overall_sentiment": overall_sentiment,
                "sentiment_trend": trend,
                "news_count": len(news),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "last_updated": datetime.now().isoformat(),
                "recent_headlines": [article.title for article in news[:5]]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis for {symbol}: {e}")
            return {}
    
    async def start_data_collection(self, symbols: Dict[str, List[str]]) -> None:
        """
        Start real-time data collection for multiple symbols.
        
        Args:
            symbols: Dictionary mapping asset types to symbol lists
        """
        if not self.is_initialized:
            raise DataCollectionError("Service not initialized", "market_data_service")
        
        if self.is_running:
            self.logger.warning("Data collection is already running")
            return
        
        try:
            self.is_running = True
            self.logger.info("Starting real-time data collection...")
            
            # Start stock data collection
            if "stocks" in symbols and symbols["stocks"]:
                await self.stock_collector.start_real_time_collection(symbols["stocks"])
                self.active_subscriptions["stocks"] = symbols["stocks"]
            
            # Start crypto data collection
            if "crypto" in symbols and symbols["crypto"]:
                await self.crypto_collector.start_real_time_collection(symbols["crypto"])
                self.active_subscriptions["crypto"] = symbols["crypto"]
            
            # Start news collection
            all_symbols = []
            if "stocks" in symbols:
                all_symbols.extend(symbols["stocks"])
            if "crypto" in symbols:
                all_symbols.extend(symbols["crypto"])
            
            if all_symbols:
                await self.news_collector.start_real_time_collection(all_symbols)
                self.active_subscriptions["news"] = all_symbols
            
            trading_logger.system_event(
                event_type="data_collection_started",
                message=f"Started collection for {len(all_symbols)} symbols",
                component="market_data_service"
            )
            
        except Exception as e:
            self.logger.error(f"Error starting data collection: {e}")
            self.is_running = False
            raise
    
    async def stop_data_collection(self) -> None:
        """Stop all real-time data collection."""
        if not self.is_running:
            self.logger.warning("Data collection is not running")
            return
        
        try:
            self.logger.info("Stopping real-time data collection...")
            
            # Stop all collectors
            await self.stock_collector.stop_real_time_collection()
            await self.crypto_collector.stop_real_time_collection()
            await self.news_collector.stop_real_time_collection()
            
            # Clear subscriptions
            self.active_subscriptions.clear()
            self.is_running = False
            
            trading_logger.system_event(
                event_type="data_collection_stopped",
                message="All data collection stopped",
                component="market_data_service"
            )
            
        except Exception as e:
            self.logger.error(f"Error stopping data collection: {e}")
    
    async def stop_all_collections(self) -> None:
        """Stop all collections (alias for stop_data_collection)."""
        await self.stop_data_collection()
    
    async def validate_symbols(self, symbols: Dict[str, List[str]]) -> Dict[str, Dict[str, bool]]:
        """
        Validate multiple symbols across different asset types.
        
        Args:
            symbols: Dictionary mapping asset types to symbol lists
            
        Returns:
            Dictionary mapping asset types to validation results
        """
        if not self.is_initialized:
            raise DataCollectionError("Service not initialized", "market_data_service")
        
        results = {}
        
        # Validate stock symbols
        if "stocks" in symbols:
            stock_results = await self.stock_collector.validate_symbols(symbols["stocks"])
            results["stocks"] = stock_results
        
        # Validate crypto symbols
        if "crypto" in symbols:
            crypto_results = await self.crypto_collector.validate_symbols(symbols["crypto"])
            results["crypto"] = crypto_results
        
        return results
    
    async def get_available_symbols(self, asset_type: str) -> List[str]:
        """
        Get list of available symbols for an asset type.
        
        Args:
            asset_type: Type of asset (stock, crypto)
            
        Returns:
            List of available symbols
        """
        try:
            if asset_type.lower() == "crypto":
                return await self.crypto_collector.get_available_symbols()
            elif asset_type.lower() == "stock":
                # For stocks, we'll return a list of popular symbols
                # In practice, this would come from a database or API
                return [
                    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM",
                    "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "PFE", "CSCO",
                    "ADBE", "CRM", "NFLX", "PYPL", "INTC", "CMCSA", "PEP", "COST",
                    "T", "DIS", "VZ", "KO", "NKE", "ABT", "MRK", "DHR", "ACN"
                ]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting available symbols for {asset_type}: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the market data service.
        
        Returns:
            Service status information
        """
        return {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "collectors": {
                "stocks": "initialized" if self.stock_collector else "not_initialized",
                "crypto": "initialized" if self.crypto_collector else "not_initialized",
                "news": "initialized" if self.news_collector else "not_initialized"
            },
            "active_subscriptions": self.active_subscriptions,
            "cache_size": len(self.data_cache),
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.data_cache),
            "cache_keys": list(self.data_cache.keys()),
            "memory_usage": "N/A"  # Could implement actual memory usage tracking
        }