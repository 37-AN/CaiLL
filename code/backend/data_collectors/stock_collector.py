"""
Stock Data Collector

This module implements data collection for stocks from multiple sources including
Yahoo Finance, Alpha Vantage, and Polygon.io. It provides real-time and historical
data with proper error handling and rate limiting.
"""

import asyncio
import aiohttp
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import asdict

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

from backend.data_collectors.base_collector import (
    BaseDataCollector, MarketData, OHLCVData, QuoteData, 
    DataType, DataFrequency, RateLimiter
)
from backend.core.config import settings
from backend.core.exceptions import DataCollectionError, APIError, RateLimitError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class YahooFinanceCollector(BaseDataCollector):
    """
    Yahoo Finance data collector using yfinance library.
    
    This collector provides free access to stock market data with good coverage
    of international markets and historical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("yahoo_finance", config)
        self.session = None
        
    async def initialize(self) -> None:
        """Initialize the Yahoo Finance collector."""
        self.session = aiohttp.ClientSession()
        self.logger.info("Yahoo Finance collector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.logger.info("Yahoo Finance collector cleaned up")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1
    ) -> List[OHLCVData]:
        """
        Get historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
        """
        await self._apply_rate_limit()
        
        try:
            # Map frequency to yfinance interval
            interval_map = {
                DataFrequency.MINUTE_1: "1m",
                DataFrequency.MINUTE_5: "5m",
                DataFrequency.MINUTE_15: "15m",
                DataFrequency.MINUTE_30: "30m",
                DataFrequency.HOUR_1: "1h",
                DataFrequency.DAY_1: "1d",
                DataFrequency.WEEK_1: "1wk",
                DataFrequency.MONTH_1: "1mo"
            }
            
            interval = interval_map.get(frequency, "1d")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if hist.empty:
                self.logger.warning(f"No data found for {symbol}")
                return []
            
            # Convert to OHLCVData objects
            ohlcv_data = []
            for timestamp, row in hist.iterrows():
                try:
                    data = OHLCVData(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']),
                        frequency=frequency,
                        source="yahoo_finance",
                        adjusted_close=float(row['Close'])  # Already adjusted
                    )
                    ohlcv_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Error processing data point for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(ohlcv_data)} data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to fetch historical data: {e}", "yahoo_finance")
    
    async def get_real_time_data(
        self,
        symbol: str
    ) -> AsyncGenerator[MarketData, None]:
        """
        Get real-time data for a symbol.
        
        Note: Yahoo Finance doesn't provide true real-time data, but we can
        get frequent updates that simulate real-time data.
        
        Args:
            symbol: Stock symbol
            
        Yields:
            Real-time market data
        """
        while self.is_running:
            try:
                await self._apply_rate_limit()
                
                # Get current data
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info and 'regularMarketPrice' in info:
                    current_price = info['regularMarketPrice']
                    prev_close = info.get('previousClose', current_price)
                    
                    # Create a simple OHLCV data point
                    now = datetime.now()
                    data = MarketData(
                        symbol=symbol,
                        timestamp=now,
                        data_type=DataType.QUOTE,
                        frequency=DataFrequency.TICK,
                        data={
                            "price": current_price,
                            "change": current_price - prev_close,
                            "change_percent": ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0,
                            "volume": info.get('regularMarketVolume', 0),
                            "bid": info.get('bid', 0),
                            "ask": info.get('ask', 0),
                            "bid_size": info.get('bidSize', 0),
                            "ask_size": info.get('askSize', 0)
                        },
                        source="yahoo_finance"
                    )
                    
                    yield data
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 60))
                
            except Exception as e:
                self.logger.error(f"Error in real-time data for {symbol}: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company information dictionary
        """
        try:
            await self._apply_rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Filter relevant information
            company_info = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "eps": info.get("trailingEps", 0),
                "revenue": info.get("totalRevenue", 0),
                "gross_profit": info.get("grossProfits", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "profit_margin": info.get("profitMargins", 0),
                "return_on_equity": info.get("returnOnEquity", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "quick_ratio": info.get("quickRatio", 0),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "country": info.get("country", ""),
                "currency": info.get("currency", ""),
                "exchange": info.get("exchange", ""),
                "market": info.get("market", ""),
                "quote_type": info.get("quoteType", ""),
                "timezone": info.get("timeZoneFullName", ""),
                "last_updated": datetime.now().isoformat()
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    async def _start_collection_loop(self, symbols: List[str]) -> None:
        """
        Start the data collection loop for multiple symbols.
        
        Args:
            symbols: List of symbols to collect data for
        """
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(
                self._collect_symbol_data(symbol)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete (they should run indefinitely)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_symbol_data(self, symbol: str) -> None:
        """
        Collect data for a single symbol.
        
        Args:
            symbol: Symbol to collect data for
        """
        try:
            async for data in self.get_real_time_data(symbol):
                if not self.is_running:
                    break
                
                # Here you would typically send the data to a message queue
                # or store it in a database
                self.logger.debug(f"Received data for {symbol}: {data.data}")
                
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")


class AlphaVantageCollector(BaseDataCollector):
    """
    Alpha Vantage data collector.
    
    This collector provides high-quality financial data with good API documentation
    and reasonable rate limits for the free tier.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("alpha_vantage", config)
        self.api_key = config.get("api_key", settings.ALPHA_VANTAGE_API_KEY)
        self.ts = None
        
    async def initialize(self) -> None:
        """Initialize the Alpha Vantage collector."""
        if not self.api_key:
            raise DataCollectionError("Alpha Vantage API key is required", "alpha_vantage")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.logger.info("Alpha Vantage collector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Alpha Vantage collector cleaned up")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1
    ) -> List[OHLCVData]:
        """
        Get historical stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
        """
        await self._apply_rate_limit()
        
        try:
            # Alpha Vantage has different functions for different timeframes
            if frequency == DataFrequency.DAY_1:
                # Use daily adjusted data
                data, meta_data = self.ts.get_daily_adjusted(
                    symbol=symbol,
                    outputsize='full'
                )
            elif frequency in [DataFrequency.MINUTE_1, DataFrequency.MINUTE_5, DataFrequency.MINUTE_15, DataFrequency.MINUTE_30]:
                # Use intraday data
                interval_map = {
                    DataFrequency.MINUTE_1: "1min",
                    DataFrequency.MINUTE_5: "5min",
                    DataFrequency.MINUTE_15: "15min",
                    DataFrequency.MINUTE_30: "30min"
                }
                interval = interval_map[frequency]
                
                data, meta_data = self.ts.get_intraday(
                    symbol=symbol,
                    interval=interval,
                    outputsize='full'
                )
            else:
                raise DataCollectionError(f"Unsupported frequency: {frequency}", "alpha_vantage")
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return []
            
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Convert to OHLCVData objects
            ohlcv_data = []
            for timestamp, row in data.iterrows():
                try:
                    if frequency == DataFrequency.DAY_1:
                        # Daily data has different column names
                        data = OHLCVData(
                            symbol=symbol,
                            timestamp=timestamp.to_pydatetime(),
                            open=float(row['1. open']),
                            high=float(row['2. high']),
                            low=float(row['3. low']),
                            close=float(row['4. close']),
                            volume=int(row['6. volume']),
                            frequency=frequency,
                            source="alpha_vantage",
                            adjusted_close=float(row['5. adjusted close'])
                        )
                    else:
                        # Intraday data
                        data = OHLCVData(
                            symbol=symbol,
                            timestamp=timestamp.to_pydatetime(),
                            open=float(row['1. open']),
                            high=float(row['2. high']),
                            low=float(row['3. low']),
                            close=float(row['4. close']),
                            volume=int(row['5. volume']),
                            frequency=frequency,
                            source="alpha_vantage"
                        )
                    ohlcv_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Error processing data point for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(ohlcv_data)} data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to fetch historical data: {e}", "alpha_vantage")
    
    async def get_real_time_data(
        self,
        symbol: str
    ) -> AsyncGenerator[MarketData, None]:
        """
        Get real-time data for a symbol.
        
        Alpha Vantage provides quote endpoints for real-time data.
        
        Args:
            symbol: Stock symbol
            
        Yields:
            Real-time market data
        """
        # Alpha Vantage doesn't have streaming real-time data
        # We'll use the quote endpoint periodically
        while self.is_running:
            try:
                await self._apply_rate_limit()
                
                # Get quote data
                from alpha_vantage.foreignexchange import ForeignExchange
                cc = ForeignExchange(key=self.api_key)
                
                # This is a simplified implementation
                # In practice, you'd use the appropriate Alpha Vantage endpoint
                data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data_type=DataType.QUOTE,
                    frequency=DataFrequency.TICK,
                    data={"message": "Real-time data not available in free tier"},
                    source="alpha_vantage"
                )
                
                yield data
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 300))  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in real-time data for {symbol}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _start_collection_loop(self, symbols: List[str]) -> None:
        """
        Start the data collection loop for multiple symbols.
        
        Args:
            symbols: List of symbols to collect data for
        """
        # Alpha Vantage has strict rate limits, so we'll collect sequentially
        for symbol in symbols:
            if not self.is_running:
                break
            
            try:
                async for data in self.get_real_time_data(symbol):
                    if not self.is_running:
                        break
                    
                    self.logger.debug(f"Received data for {symbol}: {data.data}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")


class StockDataCollector:
    """
    Main stock data collector that manages multiple data sources.
    
    This class provides a unified interface for collecting stock data from
    multiple sources with automatic failover and load balancing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("stock_data_collector")
        self.collectors = {}
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """Initialize all available stock data collectors."""
        # Yahoo Finance collector (always available)
        self.collectors["yahoo_finance"] = YahooFinanceCollector(
            self.config.get("yahoo_finance", {})
        )
        
        # Alpha Vantage collector (if API key is available)
        if settings.ALPHA_VANTAGE_API_KEY:
            self.collectors["alpha_vantage"] = AlphaVantageCollector(
                self.config.get("alpha_vantage", {"api_key": settings.ALPHA_VANTAGE_API_KEY})
            )
        
        self.logger.info(f"Initialized {len(self.collectors)} stock data collectors")
    
    async def initialize(self) -> None:
        """Initialize all collectors."""
        for name, collector in self.collectors.items():
            try:
                await collector.initialize()
                self.logger.info(f"Initialized {name} collector")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name} collector: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all collectors."""
        for name, collector in self.collectors.items():
            try:
                await collector.cleanup()
                self.logger.info(f"Cleaned up {name} collector")
            except Exception as e:
                self.logger.error(f"Failed to cleanup {name} collector: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1,
        preferred_source: Optional[str] = None
    ) -> List[OHLCVData]:
        """
        Get historical data from the best available source.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            preferred_source: Preferred data source
            
        Returns:
            List of OHLCV data points
        """
        # Try preferred source first
        if preferred_source and preferred_source in self.collectors:
            try:
                data = await self.collectors[preferred_source].get_historical_data(
                    symbol, start_date, end_date, frequency
                )
                if data:
                    return data
            except Exception as e:
                self.logger.warning(f"Preferred source {preferred_source} failed: {e}")
        
        # Try other sources
        for name, collector in self.collectors.items():
            if name == preferred_source:
                continue  # Already tried
            
            try:
                data = await collector.get_historical_data(
                    symbol, start_date, end_date, frequency
                )
                if data:
                    self.logger.info(f"Got data from {name} for {symbol}")
                    return data
            except Exception as e:
                self.logger.warning(f"Source {name} failed: {e}")
                continue
        
        raise DataCollectionError(f"No data available for {symbol}", "stock_data_collector")
    
    async def start_real_time_collection(self, symbols: List[str]) -> None:
        """
        Start real-time data collection for multiple symbols.
        
        Args:
            symbols: List of symbols to collect data for
        """
        self.logger.info(f"Starting real-time collection for {len(symbols)} symbols")
        
        # Start collection with Yahoo Finance (most reliable for free data)
        if "yahoo_finance" in self.collectors:
            await self.collectors["yahoo_finance"].start_collection(symbols)
    
    async def stop_real_time_collection(self) -> None:
        """Stop real-time data collection."""
        self.logger.info("Stopping real-time collection")
        
        for collector in self.collectors.values():
            await collector.stop_collection()
    
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate multiple symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbols to validation results
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Try with Yahoo Finance first
                is_valid = await self.collectors["yahoo_finance"].validate_symbol(symbol)
                results[symbol] = is_valid
                
                if is_valid:
                    self.logger.info(f"Symbol {symbol} is valid")
                else:
                    self.logger.warning(f"Symbol {symbol} is invalid")
                    
            except Exception as e:
                self.logger.error(f"Error validating symbol {symbol}: {e}")
                results[symbol] = False
        
        return results