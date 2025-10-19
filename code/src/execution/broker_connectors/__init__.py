"""
Broker Connectors - AI Trading System

This module provides standardized connectors for different brokers and exchanges,
enabling the trading system to work with multiple financial institutions.

Educational Note:
Broker connectors abstract the differences between various APIs, providing
a unified interface for trading operations. This allows the system to
switch between brokers or use multiple brokers simultaneously.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import hashlib
import hmac
import base64
from urllib.parse import urlencode

# Import our trading components
from ..paper_trader import Order, OrderType, OrderSide, OrderStatus, Position, MarketData
from ..live_trader import BrokerConfig, BrokerConnector, RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset classes supported by brokers"""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"
    ETF = "etf"


class OrderDuration(Enum):
    """Order duration types"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    OPG = "opg"  # Market On Open
    CLS = "cls"  # Market On Close


@dataclass
class BrokerCapabilities:
    """Broker capabilities and features"""
    supported_asset_classes: List[AssetClass]
    supported_order_types: List[OrderType]
    supported_order_durations: List[OrderDuration]
    supports_short_selling: bool
    supports_margin_trading: bool
    supports_options_trading: bool
    supports_crypto_trading: bool
    supports_forex_trading: bool
    has_real_time_data: bool
    has_historical_data: bool
    has_streaming_data: bool
    commission_schedule: Dict[str, float]
    margin_requirements: Dict[str, float]
    rate_limits: Dict[str, int]


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    account_type: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    day_trading_buying_power: float
    maintenance_margin: float
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class BrokerPosition:
    """Broker position information"""
    symbol: str
    asset_class: AssetClass
    quantity: int
    side: str  # long or short
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    average_entry_price: float
    current_price: float
    last_day_price: float
    change_today: float


class AlpacaConnector(BrokerConnector):
    """
    Alpaca Markets Connector
    
    Educational Note:
    Alpaca is a commission-free API-first broker that's popular
    with algorithmic traders. This connector handles Alpaca's
    specific API requirements and data formats.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.base_url = config.base_url or (
            "https://paper-api.alpaca.markets" if config.paper_trading 
            else "https://api.alpaca.markets"
        )
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.api_secret
        }
        self.capabilities = self._get_capabilities()
    
    def _get_capabilities(self) -> BrokerCapabilities:
        """Get Alpaca capabilities"""
        return BrokerCapabilities(
            supported_asset_classes=[AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO],
            supported_order_types=[OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.STOP_LIMIT],
            supported_order_durations=[OrderDuration.DAY, OrderDuration.GTC, OrderDuration.IOC, OrderDuration.FOK],
            supports_short_selling=True,
            supports_margin_trading=True,
            supports_options_trading=False,
            supports_crypto_trading=True,
            supports_forex_trading=False,
            has_real_time_data=True,
            has_historical_data=True,
            has_streaming_data=True,
            commission_schedule={'equity': 0.0, 'crypto': 0.0},
            margin_requirements={'equity': 0.25, 'crypto': 0.3},
            rate_limits={'orders': 200, 'data': 200}
        )
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            
            self.session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection
            await self.rate_limiter.acquire()
            async with self.session.get("/v2/account") as response:
                if response.status == 200:
                    self.connection_status = ConnectionStatus.CONNECTED
                    logger.info("Connected to Alpaca API")
                    return True
                else:
                    error_data = await response.json()
                    logger.error(f"Alpaca connection failed: {error_data}")
                    self.connection_status = ConnectionStatus.ERROR
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            self.connection_status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Alpaca API")
    
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/account") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return AccountInfo(
                        account_id=data.get("id", ""),
                        account_type=data.get("account_type", ""),
                        buying_power=float(data.get("buying_power", 0)),
                        cash=float(data.get("cash", 0)),
                        portfolio_value=float(data.get("portfolio_value", 0)),
                        equity=float(data.get("equity", 0)),
                        day_trading_buying_power=float(data.get("daytrading_buying_power", 0)),
                        maintenance_margin=float(data.get("maintenance_margin", 0)),
                        day_trade_count=int(data.get("daytrade_count", 0)),
                        pattern_day_trader=data.get("pattern_day_trader", False),
                        trading_blocked=data.get("trading_blocked", False),
                        transfers_blocked=data.get("transfers_blocked", False),
                        account_blocked=data.get("account_blocked", False),
                        created_at=datetime.fromisoformat(data.get("created_at", "").replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(data.get("updated_at", "").replace("Z", "+00:00"))
                    )
                else:
                    raise Exception(f"Failed to get account: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca account: {e}")
            raise
    
    async def get_positions(self) -> List[BrokerPosition]:
        """Get current positions"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/positions") as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for pos_data in data:
                        position = BrokerPosition(
                            symbol=pos_data.get("symbol", ""),
                            asset_class=AssetClass.EQUITY,  # Alpaca doesn't specify this
                            quantity=int(pos_data.get("qty", 0)),
                            side="long" if int(pos_data.get("qty", 0)) > 0 else "short",
                            market_value=float(pos_data.get("market_value", 0)),
                            cost_basis=float(pos_data.get("cost_basis", 0)),
                            unrealized_pnl=float(pos_data.get("unrealized_pl", 0)),
                            realized_pnl=float(pos_data.get("unrealized_pl", 0)),  # Alpaca combines these
                            average_entry_price=float(pos_data.get("avg_entry_price", 0)),
                            current_price=float(pos_data.get("current_price", 0)),
                            last_day_price=float(pos_data.get("lastday_price", 0)),
                            change_today=float(pos_data.get("change_today", 0))
                        )
                        positions.append(position)
                    
                    return positions
                else:
                    raise Exception(f"Failed to get positions: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            raise
    
    async def place_order(self, order: Order) -> Any:
        """Place order with Alpaca"""
        await self.rate_limiter.acquire()
        
        # Convert order to Alpaca format
        alpaca_order = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force
        }
        
        if order.order_type == OrderType.LIMIT and order.price:
            alpaca_order["limit_price"] = str(order.price)
        elif order.order_type == OrderType.STOP_LOSS and order.stop_price:
            alpaca_order["stop_price"] = str(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            alpaca_order["limit_price"] = str(order.price)
            alpaca_order["stop_price"] = str(order.stop_price)
        
        try:
            async with self.session.post("/v2/orders", json=alpaca_order) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    logger.error(f"Alpaca order error: {error_data}")
                    raise Exception(f"Order failed: {error_data}")
                    
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.delete(f"/v2/orders/{order_id}") as response:
                return response.status in [200, 204]
                
        except Exception as e:
            logger.error(f"Error cancelling Alpaca order: {e}")
            return False
    
    async def get_orders(self) -> List[Dict]:
        """Get orders"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/orders") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get orders: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca orders: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data"""
        await self.rate_limiter.acquire()
        
        try:
            # Get latest quote
            async with self.session.get(f"/v2/stocks/{symbol}/quotes/latest") as response:
                if response.status == 200:
                    quote_data = await response.json()
                    
                    # Get latest trade
                    async with self.session.get(f"/v2/stocks/{symbol}/trades/latest") as trade_response:
                        if trade_response.status == 200:
                            trade_data = await trade_response.json()
                            
                            return MarketData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                bid=float(quote_data.get("bp", 0)),
                                ask=float(quote_data.get("ap", 0)),
                                bid_size=int(quote_data.get("bs", 0)),
                                ask_size=int(quote_data.get("as", 0)),
                                last=float(trade_data.get("p", 0)),
                                volume=int(trade_data.get("s", 0)),
                                open=0.0,  # Not available in latest data
                                high=0.0,
                                low=0.0,
                                close=float(trade_data.get("p", 0))
                            )
                        else:
                            raise Exception(f"Failed to get trade data for {symbol}")
                else:
                    raise Exception(f"Failed to get quote data for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca market data: {e}")
            raise


class BinanceConnector(BrokerConnector):
    """
    Binance Connector
    
    Educational Note:
    Binance is a leading cryptocurrency exchange. This connector handles
    Binance's API authentication and specific requirements for crypto trading.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.base_url = config.base_url or (
            "https://testnet.binance.vision" if config.paper_trading 
            else "https://api.binance.com"
        )
        self.headers = {"X-MBX-APIKEY": config.api_key}
        self.capabilities = self._get_capabilities()
    
    def _get_capabilities(self) -> BrokerCapabilities:
        """Get Binance capabilities"""
        return BrokerCapabilities(
            supported_asset_classes=[AssetClass.CRYPTO],
            supported_order_types=[OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS],
            supported_order_durations=[OrderDuration.GTC, OrderDuration.IOC, OrderDuration.FOK],
            supports_short_selling=True,
            supports_margin_trading=True,
            supports_options_trading=False,
            supports_crypto_trading=True,
            supports_forex_trading=False,
            has_real_time_data=True,
            has_historical_data=True,
            has_streaming_data=True,
            commission_schedule={'crypto': 0.001},  # 0.1% trading fee
            margin_requirements={'crypto': 0.1},
            rate_limits={'orders': 1200, 'data': 1200}
        )
    
    def _create_signature(self, params: Dict[str, Any]) -> str:
        """Create Binance API signature"""
        query_string = urlencode(params)
        return hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection
            params = {"timestamp": int(datetime.now().timestamp() * 1000)}
            params["signature"] = self._create_signature(params)
            
            async with self.session.get("/api/v3/account", params=params) as response:
                if response.status == 200:
                    self.connection_status = ConnectionStatus.CONNECTED
                    logger.info("Connected to Binance API")
                    return True
                else:
                    error_data = await response.json()
                    logger.error(f"Binance connection failed: {error_data}")
                    self.connection_status = ConnectionStatus.ERROR
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Binance: {e}")
            self.connection_status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from Binance API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Binance API")
    
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        await self.rate_limiter.acquire()
        
        try:
            params = {"timestamp": int(datetime.now().timestamp() * 1000)}
            params["signature"] = self._create_signature(params)
            
            async with self.session.get("/api/v3/account", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Calculate balances
                    total_balance = 0.0
                    for balance in data.get("balances", []):
                        if float(balance.get("free", 0)) > 0:
                            # Convert to USD (simplified)
                            total_balance += float(balance.get("free", 0))
                    
                    return AccountInfo(
                        account_id=data.get("accountType", ""),
                        account_type=data.get("accountType", ""),
                        buying_power=total_balance,
                        cash=total_balance,
                        portfolio_value=total_balance,
                        equity=total_balance,
                        day_trading_buying_power=total_balance,
                        maintenance_margin=0.0,  # Not provided by Binance
                        day_trade_count=0,  # Not tracked by Binance
                        pattern_day_trader=False,
                        trading_blocked=data.get("canTrade", True) == False,
                        transfers_blocked=data.get("canWithdraw", True) == False,
                        account_blocked=data.get("canTrade", True) == False,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                else:
                    raise Exception(f"Failed to get account: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Binance account: {e}")
            raise
    
    async def get_positions(self) -> List[BrokerPosition]:
        """Get current positions"""
        await self.rate_limiter.acquire()
        
        try:
            params = {"timestamp": int(datetime.now().timestamp() * 1000)}
            params["signature"] = self._create_signature(params)
            
            async with self.session.get("/api/v3/account", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for balance in data.get("balances", []):
                        free_balance = float(balance.get("free", 0))
                        if free_balance > 0:
                            # Get current price (simplified)
                            symbol = f"{balance['asset']}USDT"
                            try:
                                price_data = await self._get_symbol_price(symbol)
                                current_price = float(price_data.get("price", 0))
                                
                                position = BrokerPosition(
                                    symbol=symbol,
                                    asset_class=AssetClass.CRYPTO,
                                    quantity=int(free_balance * 100000000),  # Convert to satoshis/wei equivalent
                                    side="long",
                                    market_value=free_balance * current_price,
                                    cost_basis=free_balance * current_price,  # Simplified
                                    unrealized_pnl=0.0,  # Simplified
                                    realized_pnl=0.0,
                                    average_entry_price=current_price,
                                    current_price=current_price,
                                    last_day_price=current_price,
                                    change_today=0.0
                                )
                                positions.append(position)
                            except:
                                continue
                    
                    return positions
                else:
                    raise Exception(f"Failed to get positions: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Binance positions: {e}")
            raise
    
    async def _get_symbol_price(self, symbol: str) -> Dict:
        """Get symbol price"""
        try:
            async with self.session.get("/api/v3/ticker/price", params={"symbol": symbol}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"price": "0"}
        except:
            return {"price": "0"}
    
    async def place_order(self, order: Order) -> Any:
        """Place order with Binance"""
        await self.rate_limiter.acquire()
        
        # Convert order to Binance format
        params = {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": order.order_type.value.upper(),
            "quantity": order.quantity,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        if order.order_type == OrderType.LIMIT and order.price:
            params["price"] = order.price
            params["timeInForce"] = "GTC"
        
        params["signature"] = self._create_signature(params)
        
        try:
            async with self.session.post("/api/v3/order", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    logger.error(f"Binance order error: {error_data}")
                    raise Exception(f"Order failed: {error_data}")
                    
        except Exception as e:
            logger.error(f"Error placing Binance order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        await self.rate_limiter.acquire()
        
        try:
            params = {
                "symbol": "",  # Would need symbol from order
                "orderId": order_id,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            params["signature"] = self._create_signature(params)
            
            async with self.session.delete("/api/v3/order", params=params) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error cancelling Binance order: {e}")
            return False
    
    async def get_orders(self) -> List[Dict]:
        """Get orders"""
        await self.rate_limiter.acquire()
        
        try:
            params = {
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            params["signature"] = self._create_signature(params)
            
            async with self.session.get("/api/v3/openOrders", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get orders: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting Binance orders: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data"""
        await self.rate_limiter.acquire()
        
        try:
            # Get 24hr ticker data
            async with self.session.get("/api/v3/ticker/24hr", params={"symbol": symbol}) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get order book
                    async with self.session.get("/api/v3/depth", params={"symbol": symbol, "limit": 5}) as depth_response:
                        if depth_response.status == 200:
                            depth_data = await depth_response.json()
                            
                            bids = depth_data.get("bids", [])
                            asks = depth_data.get("asks", [])
                            
                            return MarketData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                bid=float(bids[0][0]) if bids else 0.0,
                                ask=float(asks[0][0]) if asks else 0.0,
                                bid_size=float(bids[0][1]) if bids else 0.0,
                                ask_size=float(asks[0][1]) if asks else 0.0,
                                last=float(data.get("lastPrice", 0)),
                                volume=float(data.get("volume", 0)),
                                open=float(data.get("openPrice", 0)),
                                high=float(data.get("highPrice", 0)),
                                low=float(data.get("lowPrice", 0)),
                                close=float(data.get("prevClosePrice", 0))
                            )
                        else:
                            raise Exception(f"Failed to get depth data for {symbol}")
                else:
                    raise Exception(f"Failed to get ticker data for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error getting Binance market data: {e}")
            raise


class BrokerManager:
    """
    Broker Manager
    
    Educational Note:
    The Broker Manager coordinates multiple broker connections,
    providing a unified interface for trading across different venues.
    This enables diversification of execution venues and access to
    different asset classes.
    """
    
    def __init__(self):
        self.connectors: Dict[str, BrokerConnector] = {}
        self.primary_broker: Optional[str] = None
        self.fallback_brokers: List[str] = []
        
    def add_broker(self, name: str, connector: BrokerConnector, is_primary: bool = False):
        """Add broker connector"""
        self.connectors[name] = connector
        
        if is_primary:
            self.primary_broker = name
        else:
            self.fallback_brokers.append(name)
        
        logger.info(f"Added broker: {name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all brokers"""
        results = {}
        
        for name, connector in self.connectors.items():
            try:
                success = await connector.connect()
                results[name] = success
                
                if success:
                    logger.info(f"Connected to {name}")
                else:
                    logger.error(f"Failed to connect to {name}")
                    
            except Exception as e:
                logger.error(f"Error connecting to {name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all brokers"""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    def get_primary_connector(self) -> Optional[BrokerConnector]:
        """Get primary broker connector"""
        if self.primary_broker and self.primary_broker in self.connectors:
            return self.connectors[self.primary_broker]
        return None
    
    def get_connector(self, name: str) -> Optional[BrokerConnector]:
        """Get specific broker connector"""
        return self.connectors.get(name)
    
    def get_available_connectors(self) -> List[BrokerConnector]:
        """Get all connected brokers"""
        return [
            connector for connector in self.connectors.values()
            if connector.connection_status == ConnectionStatus.CONNECTED
        ]
    
    async def get_best_connector_for_symbol(self, symbol: str, asset_class: AssetClass) -> Optional[BrokerConnector]:
        """Get best connector for trading a symbol"""
        
        available = self.get_available_connectors()
        
        # Filter by asset class support
        suitable = [
            connector for connector in available
            if asset_class in connector.capabilities.supported_asset_classes
        ]
        
        if not suitable:
            return None
        
        # Select based on priority (primary first, then by reliability)
        if self.get_primary_connector() in suitable:
            return self.get_primary_connector()
        
        # Sort by reliability score
        suitable.sort(key=lambda c: c.capabilities.rate_limits.get('orders', 0), reverse=True)
        
        return suitable[0]
    
    def get_broker_capabilities(self, name: str) -> Optional[BrokerCapabilities]:
        """Get broker capabilities"""
        connector = self.connectors.get(name)
        return connector.capabilities if connector else None
    
    def get_all_capabilities(self) -> Dict[str, BrokerCapabilities]:
        """Get all broker capabilities"""
        return {
            name: connector.capabilities
            for name, connector in self.connectors.items()
        }


def explain_broker_connectors():
    """
    Educational explanation of broker connectors
    """
    
    print("=== Broker Connectors Educational Guide ===\n")
    
    concepts = {
        'Broker Connector': "Software layer that standardizes communication with different broker APIs",
        
        'API Standardization': "Creating a unified interface despite different broker APIs",
        
        'Asset Classes': "Different types of financial instruments (stocks, crypto, options, etc.)",
        
        'Order Types': "Different ways to execute trades (market, limit, stop, etc.)",
        
        'Rate Limiting': "Controlling API request frequency to avoid being blocked",
        
        'Authentication': "Secure methods to verify identity with brokers",
        
        'Error Handling': "Managing API failures and network issues",
        
        'Fallback Brokers': "Alternative brokers when primary is unavailable",
        
        'Multi-Broker Trading': "Using multiple brokers simultaneously for better execution"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Popular Brokers for Algorithmic Trading ===")
    brokers = {
        'Alpaca': "API-first broker, commission-free stocks and ETFs, great for beginners",
        'Interactive Brokers': "Full-service broker, global markets, professional tools",
        'Binance': "Leading crypto exchange, low fees, wide selection of coins",
        'Coinbase': "User-friendly crypto exchange, good for retail traders",
        'TD Ameritrade': "Traditional broker with good API access",
        'Kraken': "Established crypto exchange with good security"
    }
    
    for broker, description in brokers.items():
        print(f"{broker}:")
        print(f"  {description}\n")
    
    print("=== Best Practices for Broker Connectors ===")
    practices = [
        "1. Always handle rate limits properly",
        "2. Implement robust error handling and retries",
        "3. Use secure authentication methods",
        "4. Monitor connection status continuously",
        "5. Have fallback brokers for redundancy",
        "6. Log all API calls and responses",
        "7. Test thoroughly with paper trading",
        "8. Handle different data formats from brokers",
        "9. Implement proper timeout handling",
        "10. Keep API credentials secure and rotated"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Common Integration Challenges ===")
    challenges = [
        "• Different API formats and authentication methods",
        "• Rate limits and server-side throttling",
        "• Network latency and connection issues",
        "• Data format inconsistencies",
        "• Market hours and timezone differences",
        "• Symbol naming conventions",
        "• Order type support variations",
        "• Error code standardization",
        "• Real-time data streaming complexities",
        "• Regulatory and compliance differences"
    ]
    
    for challenge in challenges:
        print(challenge)


if __name__ == "__main__":
    # Example usage
    explain_broker_connectors()
    
    print("\n=== Broker Connector Example ===")
    
    # Create broker manager
    manager = BrokerManager()
    
    # Example configurations (would use real API keys in production)
    alpaca_config = BrokerConfig(
        broker_type=BrokerType.ALPACA,
        api_key="your_alpaca_key",
        api_secret="your_alpaca_secret",
        paper_trading=True
    )
    
    binance_config = BrokerConfig(
        broker_type=BrokerType.BINANCE,
        api_key="your_binance_key",
        api_secret="your_binance_secret",
        paper_trading=True
    )
    
    print("To use broker connectors:")
    print("1. Create broker configurations with API keys")
    print("2. Initialize appropriate connector for each broker")
    print("3. Add connectors to the broker manager")
    print("4. Connect to all brokers")
    print("5. Use the manager for unified trading operations")
    
    print(f"\nExample setup:")
    print(f"- Alpaca for stock and ETF trading")
    print(f"- Binance for cryptocurrency trading")
    print(f"- Automatic fallback between brokers")
    print(f"- Unified order management across venues")
    
    # Note: We don't actually connect in this example
    # as it requires valid API credentials
    print("\n⚠️  Remember to:")
    print("• Keep API keys secure")
    print("• Use paper trading for testing")
    print("• Monitor rate limits")
    print("• Handle connection failures gracefully")
    print("• Implement proper error handling")