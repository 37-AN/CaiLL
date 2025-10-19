"""
Cryptocurrency Data Collector

This module implements data collection for cryptocurrencies from multiple exchanges
including Binance, Coinbase, and CoinGecko. It provides real-time and historical
data with proper error handling and rate limiting.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import asdict

import ccxt
import pandas as pd

from backend.data_collectors.base_collector import (
    BaseDataCollector, MarketData, OHLCVData, QuoteData, 
    DataType, DataFrequency, RateLimiter
)
from backend.core.config import settings
from backend.core.exceptions import DataCollectionError, APIError, RateLimitError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class BinanceCollector(BaseDataCollector):
    """
    Binance cryptocurrency data collector.
    
    This collector provides access to Binance exchange data with high quality
    real-time data and comprehensive historical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("binance", config)
        self.exchange = None
        self.ws_session = None
        
    async def initialize(self) -> None:
        """Initialize the Binance collector."""
        try:
            # Initialize CCXT exchange
            self.exchange = ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': settings.BINANCE_TESTNET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            self.logger.info("Binance collector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance collector: {e}")
            raise DataCollectionError(f"Binance initialization failed: {e}", "binance")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.exchange:
            await self.exchange.close()
        if self.ws_session:
            await self.ws_session.close()
        self.logger.info("Binance collector cleaned up")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1
    ) -> List[OHLCVData]:
        """
        Get historical cryptocurrency data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
        """
        await self._apply_rate_limit()
        
        try:
            # Map frequency to Binance timeframe
            timeframe_map = {
                DataFrequency.MINUTE_1: '1m',
                DataFrequency.MINUTE_5: '5m',
                DataFrequency.MINUTE_15: '15m',
                DataFrequency.MINUTE_30: '30m',
                DataFrequency.HOUR_1: '1h',
                DataFrequency.HOUR_4: '4h',
                DataFrequency.DAY_1: '1d',
                DataFrequency.WEEK_1: '1w'
            }
            
            timeframe = timeframe_map.get(frequency, '1d')
            
            # Convert dates to timestamps
            since = self.exchange.parse8601(start_date.isoformat())
            limit = 1000  # Binance limit per request
            
            all_ohlcv = []
            current_since = since
            
            while current_since < self.exchange.parse8601(end_date.isoformat()):
                # Fetch OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # Update since for next request
                current_since = ohlcv[-1][0] + 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Convert to OHLCVData objects
            ohlcv_data = []
            for candle in all_ohlcv:
                try:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000)
                    
                    if timestamp > end_date:
                        break
                    
                    data = OHLCVData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=int(candle[5]),
                        frequency=frequency,
                        source="binance"
                    )
                    ohlcv_data.append(data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing candle data: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(ohlcv_data)} data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to fetch historical data: {e}", "binance")
    
    async def get_real_time_data(
        self,
        symbol: str
    ) -> AsyncGenerator[MarketData, None]:
        """
        Get real-time data for a cryptocurrency symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Yields:
            Real-time market data
        """
        while self.is_running:
            try:
                await self._apply_rate_limit()
                
                # Get ticker data
                ticker = await self.exchange.fetch_ticker(symbol)
                
                # Create market data
                data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data_type=DataType.QUOTE,
                    frequency=DataFrequency.TICK,
                    data={
                        "price": ticker['last'],
                        "bid": ticker['bid'],
                        "ask": ticker['ask'],
                        "bid_size": ticker['bidVolume'],
                        "ask_size": ticker['askVolume'],
                        "volume": ticker['baseVolume'],
                        "change": ticker['change'],
                        "change_percent": ticker['percentage'],
                        "high": ticker['high'],
                        "low": ticker['low'],
                        "open": ticker['open']
                    },
                    source="binance"
                )
                
                yield data
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 5))
                
            except Exception as e:
                self.logger.error(f"Error in real-time data for {symbol}: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to fetch
            
        Returns:
            Order book data
        """
        await self._apply_rate_limit()
        
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "bids": order_book['bids'][:limit],
                "asks": order_book['asks'][:limit],
                "source": "binance"
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """
        Get trading fees for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Trading fee information
        """
        try:
            fees = await self.exchange.fetch_trading_fees()
            
            return {
                "symbol": symbol,
                "maker_fee": fees['trading']['maker'],
                "taker_fee": fees['trading']['taker'],
                "source": "binance"
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching trading fees for {symbol}: {e}")
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
        
        # Wait for all tasks to complete
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
                
                self.logger.debug(f"Received data for {symbol}: {data.data}")
                
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")


class CoinGeckoCollector(BaseDataCollector):
    """
    CoinGecko cryptocurrency data collector.
    
    This collector provides free access to cryptocurrency data with good coverage
    of various coins and historical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("coingecko", config)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        
    async def initialize(self) -> None:
        """Initialize the CoinGecko collector."""
        self.session = aiohttp.ClientSession()
        self.logger.info("CoinGecko collector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.logger.info("CoinGecko collector cleaned up")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY_1
    ) -> List[OHLCVData]:
        """
        Get historical cryptocurrency data from CoinGecko.
        
        Args:
            symbol: CoinGecko coin ID (e.g., 'bitcoin')
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            List of OHLCV data points
        """
        await self._apply_rate_limit()
        
        try:
            # CoinGecko uses coin IDs, not symbols
            coin_id = symbol.lower()
            
            # Convert dates to timestamps
            from_timestamp = int(start_date.timestamp())
            to_timestamp = int(end_date.timestamp())
            
            # Map frequency to CoinGecko days
            days_map = {
                DataFrequency.HOUR_1: max(1, (to_timestamp - from_timestamp) // 3600),
                DataFrequency.DAY_1: max(1, (to_timestamp - from_timestamp) // 86400),
                DataFrequency.WEEK_1: max(1, (to_timestamp - from_timestamp) // 604800),
                DataFrequency.MONTH_1: max(1, (to_timestamp - from_timestamp) // 2592000)
            }
            
            days = days_map.get(frequency, max(1, (to_timestamp - from_timestamp) // 86400))
            
            # Make API request
            url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': from_timestamp,
                'to': to_timestamp,
                'days': min(days, 365)  # CoinGecko limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise APIError(f"CoinGecko API error: {response.status}", "coingecko", response.status)
                
                data = await response.json()
            
            # Convert to OHLCVData objects
            ohlcv_data = []
            for price_point in data.get('prices', []):
                try:
                    timestamp = datetime.fromtimestamp(price_point[0] / 1000)
                    
                    # CoinGecko only provides price data, not OHLCV
                    # We'll create a simple OHLCV with same values
                    price = float(price_point[1])
                    
                    data = OHLCVData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0,  # Not available in this endpoint
                        frequency=frequency,
                        source="coingecko"
                    )
                    ohlcv_data.append(data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing price point: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(ohlcv_data)} data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to fetch historical data: {e}", "coingecko")
    
    async def get_real_time_data(
        self,
        symbol: str
    ) -> AsyncGenerator[MarketData, None]:
        """
        Get real-time data for a cryptocurrency.
        
        CoinGecko doesn't provide true real-time data, but we can get
        frequent updates that simulate real-time data.
        
        Args:
            symbol: CoinGecko coin ID
            
        Yields:
            Real-time market data
        """
        while self.is_running:
            try:
                await self._apply_rate_limit()
                
                # Get current price
                url = f"{self.base_url}/simple/price"
                params = {
                    'ids': symbol.lower(),
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        raise APIError(f"CoinGecko API error: {response.status}", "coingecko", response.status)
                    
                    data = await response.json()
                
                if symbol.lower() in data:
                    coin_data = data[symbol.lower()]
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        data_type=DataType.QUOTE,
                        frequency=DataFrequency.TICK,
                        data={
                            "price": coin_data.get('usd', 0),
                            "change_24h": coin_data.get('usd_24h_change', 0),
                            "volume_24h": coin_data.get('usd_24h_vol', 0)
                        },
                        source="coingecko"
                    )
                    
                    yield market_data
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 60))
                
            except Exception as e:
                self.logger.error(f"Error in real-time data for {symbol}: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def get_coin_list(self) -> List[Dict[str, str]]:
        """
        Get list of available coins from CoinGecko.
        
        Returns:
            List of coins with ID and symbol
        """
        await self._apply_rate_limit()
        
        try:
            url = f"{self.base_url}/coins/list"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise APIError(f"CoinGecko API error: {response.status}", "coingecko", response.status)
                
                data = await response.json()
            
            return [
                {
                    "id": coin['id'],
                    "symbol": coin['symbol'].upper(),
                    "name": coin['name']
                }
                for coin in data
            ]
            
        except Exception as e:
            self.logger.error(f"Error fetching coin list: {e}")
            return []
    
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
        
        # Wait for all tasks to complete
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
                
                self.logger.debug(f"Received data for {symbol}: {data.data}")
                
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")


class CryptoDataCollector:
    """
    Main cryptocurrency data collector that manages multiple exchanges.
    
    This class provides a unified interface for collecting cryptocurrency data
    from multiple exchanges with automatic failover and load balancing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("crypto_data_collector")
        self.collectors = {}
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """Initialize all available crypto data collectors."""
        # Binance collector (if API keys are available)
        if settings.BINANCE_API_KEY and settings.BINANCE_SECRET_KEY:
            self.collectors["binance"] = BinanceCollector(
                self.config.get("binance", {})
            )
        
        # CoinGecko collector (always available)
        self.collectors["coingecko"] = CoinGeckoCollector(
            self.config.get("coingecko", {})
        )
        
        self.logger.info(f"Initialized {len(self.collectors)} crypto data collectors")
    
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
            symbol: Cryptocurrency symbol
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
        
        raise DataCollectionError(f"No data available for {symbol}", "crypto_data_collector")
    
    async def start_real_time_collection(self, symbols: List[str]) -> None:
        """
        Start real-time data collection for multiple symbols.
        
        Args:
            symbols: List of symbols to collect data for
        """
        self.logger.info(f"Starting real-time collection for {len(symbols)} symbols")
        
        # Start collection with available collectors
        for name, collector in self.collectors.items():
            try:
                await collector.start_collection(symbols)
                self.logger.info(f"Started collection with {name}")
            except Exception as e:
                self.logger.error(f"Failed to start collection with {name}: {e}")
    
    async def stop_real_time_collection(self) -> None:
        """Stop real-time data collection."""
        self.logger.info("Stopping real-time collection")
        
        for collector in self.collectors.values():
            await collector.stop_collection()
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from all collectors.
        
        Returns:
            List of available symbols
        """
        all_symbols = set()
        
        # Get symbols from CoinGecko (most comprehensive)
        if "coingecko" in self.collectors:
            try:
                coins = await self.collectors["coingecko"].get_coin_list()
                all_symbols.update([coin["symbol"] for coin in coins])
            except Exception as e:
                self.logger.error(f"Error getting symbols from CoinGecko: {e}")
        
        return list(all_symbols)
    
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
                # Try with CoinGecko first (most comprehensive)
                if "coingecko" in self.collectors:
                    is_valid = await self.collectors["coingecko"].validate_symbol(symbol)
                    results[symbol] = is_valid
                else:
                    results[symbol] = False
                
                if results[symbol]:
                    self.logger.info(f"Symbol {symbol} is valid")
                else:
                    self.logger.warning(f"Symbol {symbol} is invalid")
                    
            except Exception as e:
                self.logger.error(f"Error validating symbol {symbol}: {e}")
                results[symbol] = False
        
        return results