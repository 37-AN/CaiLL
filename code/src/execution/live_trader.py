"""
Live Trading Engine - AI Trading System

This module implements the live trading execution system that connects
to real broker APIs for actual trading with real money.

⚠️ CRITICAL WARNING ⚠️
This module handles REAL MONEY trading. Always:
1. Test thoroughly with paper trading first
2. Start with small amounts of capital
3. Use proper risk management
4. Monitor positions actively
5. Have emergency stop mechanisms

Educational Note:
Live trading introduces real-world complexities including:
- API rate limits and downtime
- Real slippage and market impact
- Emotional decision making
- Regulatory requirements
- Tax implications
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import logging
from abc import ABC, abstractmethod
import json

# Import our paper trading components for reuse
from .paper_trader import (
    Order, OrderType, OrderSide, OrderStatus, Position, PositionType,
    Trade, Portfolio, MarketData, SlippageModel, CommissionModel
)
from ..risk_management.position_sizer import PositionSizingManager
from ..risk_management.risk_calculator import RiskCalculator
from ..risk_management.circuit_breakers import CircuitBreakerManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes"""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    BINANCE = "binance"
    COINBASE = "coinbase"
    CUSTOM = "custom"


class ConnectionStatus(Enum):
    """Connection status to broker"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_type: BrokerType
    api_key: str
    api_secret: str
    base_url: Optional[str] = None
    paper_trading: bool = True
    rate_limit: int = 200  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Trade execution report"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    avg_price: float
    commission: float
    status: OrderStatus
    timestamp: datetime
    execution_venue: str
    liquidity: str
    notes: str = ""


@dataclass
class RiskLimits:
    """Trading risk limits"""
    max_position_size: float = 0.10  # 10% of portfolio per position
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per day
    max_leverage: float = 2.0
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_concentration: float = 0.25  # 25% max in single position
    min_account_balance: float = 1000  # Minimum account balance


class BrokerConnector(ABC):
    """Abstract base class for broker connectors"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.session = None
        self.rate_limiter = RateLimiter(config.rate_limit)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionReport:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get open orders"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data for symbol"""
        pass


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit"""
        async with self.lock:
            now = datetime.now()
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < timedelta(minutes=1)]
            
            # Check if we can make a call
            if len(self.calls) >= self.calls_per_minute:
                # Wait until we can make a call
                oldest_call = min(self.calls)
                wait_time = timedelta(minutes=1) - (now - oldest_call)
                await asyncio.sleep(wait_time.total_seconds())
            
            self.calls.append(now)


class AlpacaConnector(BrokerConnector):
    """
    Alpaca broker connector
    
    Educational Note:
    Alpaca is popular for algorithmic trading due to its API-first approach
    and commission-free trading. This connector handles the specifics of
    Alpaca's API structure and requirements.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.base_url = config.base_url or ("https://paper-api.alpaca.markets" if config.paper_trading else "https://api.alpaca.markets")
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.api_secret
        }
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            
            # Create session
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
                    self.connection_status = ConnectionStatus.ERROR
                    logger.error(f"Failed to connect to Alpaca: {response.status}")
                    return False
                    
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Alpaca API")
    
    async def place_order(self, order: Order) -> ExecutionReport:
        """Place order with Alpaca"""
        
        await self.rate_limiter.acquire()
        
        # Convert order to Alpaca format
        alpaca_order = {
            "symbol": order.symbol,
            "qty": order.quantity,
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force
        }
        
        if order.order_type == OrderType.LIMIT and order.price:
            alpaca_order["limit_price"] = order.price
        elif order.order_type == OrderType.STOP_LOSS and order.stop_price:
            alpaca_order["stop_price"] = order.stop_price
        
        try:
            async with self.session.post("/v2/orders", json=alpaca_order) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert back to our format
                    return ExecutionReport(
                        order_id=data.get("id", order.id),
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        filled_quantity=int(data.get("filled_qty", 0)),
                        avg_price=float(data.get("filled_avg_price", 0)),
                        commission=0.0,  # Alpaca has no commission
                        status=self._convert_alpaca_status(data.get("status", "new")),
                        timestamp=datetime.now(),
                        execution_venue="Alpaca",
                        liquidity="Unknown"
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Alpaca order error: {error_text}")
                    raise Exception(f"Order failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error placing order with Alpaca: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca"""
        
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.delete(f"/v2/orders/{order_id}") as response:
                return response.status == 200 or response.status == 204
                
        except Exception as e:
            logger.error(f"Error cancelling order with Alpaca: {e}")
            return False
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information from Alpaca"""
        
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/account") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get account: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting account from Alpaca: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Alpaca"""
        
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/positions") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get positions: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting positions from Alpaca: {e}")
            raise
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get orders from Alpaca"""
        
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get("/v2/orders") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get orders: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting orders from Alpaca: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data from Alpaca"""
        
        await self.rate_limiter.acquire()
        
        try:
            # Get latest trade
            async with self.session.get(f"/v2/stocks/{symbol}/trades/latest") as response:
                if response.status == 200:
                    trade_data = await response.json()
                    
                    # Get latest quote
                    async with self.session.get(f"/v2/stocks/{symbol}/quotes/latest") as quote_response:
                        if quote_response.status == 200:
                            quote_data = await quote_response.json()
                            
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
                            raise Exception(f"Failed to get quote for {symbol}")
                else:
                    raise Exception(f"Failed to get trade for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error getting market data from Alpaca: {e}")
            raise
    
    def _convert_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca status to our OrderStatus"""
        
        status_mapping = {
            "new": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIAL_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }
        
        return status_mapping.get(alpaca_status, OrderStatus.PENDING)


class LiveTradingEngine:
    """
    Live Trading Engine
    
    Educational Note:
    This is the core component that handles real money trading.
    It includes comprehensive safety checks, risk management,
    and monitoring to protect your capital.
    
    ⚠️ ALWAYS TEST WITH PAPER TRADING FIRST ⚠️
    """
    
    def __init__(
        self,
        broker_config: BrokerConfig,
        risk_limits: RiskLimits,
        trading_mode: TradingMode = TradingMode.PAPER
    ):
        self.broker_config = broker_config
        self.risk_limits = risk_limits
        self.trading_mode = trading_mode
        
        # Initialize broker connector
        self.broker_connector = self._create_broker_connector(broker_config)
        
        # Risk management
        self.position_sizer = PositionSizingManager()
        self.risk_calculator = RiskCalculator()
        self.circuit_breaker_manager = CircuitBreakerManager()
        
        # State tracking
        self.is_running = False
        self.emergency_stop = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Monitoring
        self.last_heartbeat = datetime.now()
        self.connection_check_interval = 60  # seconds
        
        # Statistics
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
    
    def _create_broker_connector(self, config: BrokerConfig) -> BrokerConnector:
        """Create appropriate broker connector"""
        
        if config.broker_type == BrokerType.ALPACA:
            return AlpacaConnector(config)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker_type}")
    
    async def start(self) -> bool:
        """Start the live trading engine"""
        
        logger.warning("🚨 STARTING LIVE TRADING ENGINE 🚨")
        logger.warning("This will trade with REAL MONEY if not in paper mode")
        
        if self.trading_mode == TradingMode.LIVE:
            logger.error("⚠️ LIVE TRADING MODE ENABLED ⚠️")
            logger.error("You are about to trade with REAL MONEY")
            logger.error("Make sure you have:")
            logger.error("1. Tested thoroughly with paper trading")
            logger.error("2. Set appropriate risk limits")
            logger.error("3. Monitored the system carefully")
            
            # Add confirmation delay for safety
            await asyncio.sleep(5)
        
        try:
            # Connect to broker
            connected = await self.broker_connector.connect()
            if not connected:
                logger.error("Failed to connect to broker")
                return False
            
            # Initialize positions and orders
            await self._sync_with_broker()
            
            # Start monitoring
            self.is_running = True
            asyncio.create_task(self._monitor_connection())
            asyncio.create_task(self._monitor_positions())
            asyncio.create_task(self._reset_daily_stats())
            
            logger.info(f"Live trading engine started in {self.trading_mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Error starting live trading engine: {e}")
            return False
    
    async def stop(self):
        """Stop the live trading engine"""
        
        logger.info("Stopping live trading engine")
        self.is_running = False
        
        # Cancel all open orders
        await self._cancel_all_orders()
        
        # Disconnect from broker
        await self.broker_connector.disconnect()
        
        logger.info("Live trading engine stopped")
    
    async def place_order(self, order: Order) -> Optional[ExecutionReport]:
        """Place an order with comprehensive safety checks"""
        
        if not self.is_running:
            logger.error("Trading engine is not running")
            return None
        
        if self.emergency_stop:
            logger.error("Emergency stop is active - no new orders allowed")
            return None
        
        # Pre-trade risk checks
        if not await self._pre_trade_risk_check(order):
            logger.error(f"Order {order.id} failed risk check")
            return None
        
        try:
            # Place order with broker
            if self.trading_mode == TradingMode.PAPER:
                # Use paper trading engine
                report = await self._place_paper_order(order)
            else:
                # Live order
                report = await self.broker_connector.place_order(order)
            
            # Store order
            self.orders[order.id] = order
            
            # Update statistics
            self.daily_trades += 1
            
            logger.info(f"Order placed: {order.symbol} {order.side.value} {order.quantity} @ {order.price or 'MARKET'}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error placing order {order.id}: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        
        try:
            if self.trading_mode == TradingMode.PAPER:
                # Paper trading cancellation
                if order_id in self.orders:
                    self.orders[order_id].status = OrderStatus.CANCELLED
                    return True
                return False
            else:
                # Live cancellation
                success = await self.broker_connector.cancel_order(order_id)
                if success and order_id in self.orders:
                    self.orders[order_id].status = OrderStatus.CANCELLED
                
                return success
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def emergency_stop_all(self):
        """Emergency stop - cancel all orders and prevent new trades"""
        
        logger.error("🆘 EMERGENCY STOP ACTIVATED 🆘")
        self.emergency_stop = True
        
        # Cancel all orders
        await self._cancel_all_orders()
        
        # Send alert
        logger.error("All trading halted due to emergency stop")
    
    async def _pre_trade_risk_check(self, order: Order) -> bool:
        """Comprehensive pre-trade risk checks"""
        
        # Check account balance
        account = await self.broker_connector.get_account()
        portfolio_value = float(account.get("portfolio_value", 0))
        
        if portfolio_value < self.risk_limits.min_account_balance:
            logger.error(f"Account balance ${portfolio_value:,.2f} below minimum ${self.risk_limits.min_account_balance:,.2f}")
            return False
        
        # Check position size
        position_value = order.quantity * (order.price or await self._get_market_price(order.symbol))
        position_ratio = position_value / portfolio_value
        
        if position_ratio > self.risk_limits.max_position_size:
            logger.error(f"Position size {position_ratio:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.risk_limits.max_daily_loss * portfolio_value:
            logger.error(f"Daily loss ${abs(self.daily_pnl):,.2f} exceeds limit")
            return False
        
        # Check concentration risk
        current_position = self.positions.get(order.symbol, Position(
            symbol=order.symbol,
            quantity=0,
            avg_price=0,
            market_value=0,
            unrealized_pnl=0,
            realized_pnl=0,
            position_type=PositionType.FLAT,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ))
        
        new_quantity = current_position.quantity
        if order.side == OrderSide.BUY:
            new_quantity += order.quantity
        else:
            new_quantity -= order.quantity
        
        new_position_value = abs(new_quantity) * (order.price or await self._get_market_price(order.symbol))
        new_concentration = new_position_value / portfolio_value
        
        if new_concentration > self.risk_limits.max_concentration:
            logger.error(f"Concentration {new_concentration:.2%} exceeds limit {self.risk_limits.max_concentration:.2%}")
            return False
        
        # Check circuit breakers
        portfolio_data = await self._get_portfolio_data()
        triggered_breakers = self.circuit_breaker_manager.check_circuit_breakers(portfolio_data)
        
        if triggered_breakers:
            logger.error(f"Circuit breakers triggered: {triggered_breakers}")
            return False
        
        return True
    
    async def _sync_with_broker(self):
        """Sync local state with broker"""
        
        try:
            # Get positions
            broker_positions = await self.broker_connector.get_positions()
            for pos_data in broker_positions:
                position = self._convert_broker_position(pos_data)
                self.positions[position.symbol] = position
            
            # Get orders
            broker_orders = await self.broker_connector.get_orders()
            for order_data in broker_orders:
                order = self._convert_broker_order(order_data)
                self.orders[order.id] = order
                
        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")
    
    async def _monitor_connection(self):
        """Monitor connection to broker"""
        
        while self.is_running:
            try:
                # Check connection
                account = await self.broker_connector.get_account()
                self.last_heartbeat = datetime.now()
                
                # Update daily P&L
                await self._update_daily_pnl()
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                self.broker_connector.connection_status = ConnectionStatus.ERROR
            
            await asyncio.sleep(self.connection_check_interval)
    
    async def _monitor_positions(self):
        """Monitor positions and update values"""
        
        while self.is_running:
            try:
                # Update position values
                for position in self.positions.values():
                    current_price = await self._get_market_price(position.symbol)
                    position.market_value = current_price * abs(position.quantity)
                    
                    # Update unrealized P&L
                    if position.position_type == PositionType.LONG:
                        position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
            
            await asyncio.sleep(30)  # Update every 30 seconds
    
    async def _reset_daily_stats(self):
        """Reset daily statistics at midnight"""
        
        while self.is_running:
            now = datetime.now()
            
            if now.date() > self.last_reset_date:
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.last_reset_date = now.date()
                logger.info("Daily statistics reset")
            
            await asyncio.sleep(3600)  # Check every hour
    
    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        
        for order_id, order in list(self.orders.items()):
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
                await self.cancel_order(order_id)
    
    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        
        try:
            market_data = await self.broker_connector.get_market_data(symbol)
            return market_data.last
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return 0.0
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data for risk checks"""
        
        account = await self.broker_connector.get_account()
        
        return {
            'portfolio_value': float(account.get('portfolio_value', 0)),
            'cash': float(account.get('cash', 0)),
            'positions': {
                symbol: {
                    'current_value': pos.market_value,
                    'entry_value': pos.avg_price * abs(pos.quantity)
                }
                for symbol, pos in self.positions.items()
            },
            'total_position_value': sum(pos.market_value for pos in self.positions.values())
        }
    
    async def _update_daily_pnl(self):
        """Update daily P&L"""
        
        # This is a simplified calculation
        # In reality, you'd track this more carefully
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        self.daily_pnl = total_unrealized + total_realized
    
    async def _place_paper_order(self, order: Order) -> ExecutionReport:
        """Place paper order (simplified)"""
        
        # This would integrate with the paper trading engine
        # For now, return a simulated execution
        price = await self._get_market_price(order.symbol)
        
        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_price=price,
            commission=0.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            execution_venue="Paper",
            liquidity="Simulated"
        )
    
    def _convert_broker_position(self, pos_data: Dict) -> Position:
        """Convert broker position data to our format"""
        
        return Position(
            symbol=pos_data.get("symbol", ""),
            quantity=int(pos_data.get("qty", 0)),
            avg_price=float(pos_data.get("avg_entry_price", 0)),
            market_value=float(pos_data.get("market_value", 0)),
            unrealized_pnl=float(pos_data.get("unrealized_pl", 0)),
            realized_pnl=float(pos_data.get("unrealized_pl", 0)),  # Simplified
            position_type=PositionType.LONG if int(pos_data.get("qty", 0)) > 0 else PositionType.SHORT,
            created_at=datetime.now(),  # Not available from broker
            updated_at=datetime.now()
        )
    
    def _convert_broker_order(self, order_data: Dict) -> Order:
        """Convert broker order data to our format"""
        
        return Order(
            id=order_data.get("id", ""),
            symbol=order_data.get("symbol", ""),
            side=OrderSide(order_data.get("side", "buy")),
            order_type=OrderType(order_data.get("type", "market")),
            quantity=int(order_data.get("qty", 0)),
            price=float(order_data.get("limit_price", 0)) or None,
            stop_price=float(order_data.get("stop_price", 0)) or None,
            time_in_force=order_data.get("time_in_force", "GTC"),
            created_at=datetime.fromisoformat(order_data.get("created_at", "").replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(order_data.get("updated_at", "").replace("Z", "+00:00")),
            status=self._convert_order_status(order_data.get("status", "new")),
            filled_quantity=int(order_data.get("filled_qty", 0)),
            avg_fill_price=float(order_data.get("filled_avg_price", 0))
        )
    
    def _convert_order_status(self, broker_status: str) -> OrderStatus:
        """Convert broker order status to our format"""
        
        status_mapping = {
            "new": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIAL_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }
        
        return status_mapping.get(broker_status, OrderStatus.PENDING)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        
        return {
            'is_running': self.is_running,
            'trading_mode': self.trading_mode.value,
            'emergency_stop': self.emergency_stop,
            'connection_status': self.broker_connector.connection_status.value,
            'last_heartbeat': self.last_heartbeat,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'open_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]]),
            'positions': len(self.positions),
            'total_trades': len(self.trades)
        }


def explain_live_trading():
    """
    Educational explanation of live trading
    """
    
    print("=== Live Trading Educational Guide ===\n")
    
    warnings = [
        "Live trading involves REAL MONEY and REAL RISK",
        "Always test thoroughly with paper trading first",
        "Start with small amounts of capital",
        "Monitor positions actively",
        "Have emergency stop mechanisms",
        "Understand tax implications",
        "Keep detailed records",
        "Never risk more than you can afford to lose"
    ]
    
    print("🚨 CRITICAL WARNINGS 🚨")
    for i, warning in enumerate(warnings, 1):
        print(f"{i}. {warning}")
    
    print("\n=== Live Trading vs Paper Trading ===")
    differences = {
        "Execution Speed": "Live: Real market latency | Paper: Simulated",
        "Slippage": "Live: Real market impact | Paper: Model-based",
        "Psychology": "Live: Real emotions | Paper: No pressure",
        "Costs": "Live: Real commissions/fees | Paper: Simulated",
        "Risk": "Live: Real money loss | Paper: No financial risk",
        "Regulation": "Live: Regulatory requirements | Paper: None"
    }
    
    for aspect, comparison in differences.items():
        print(f"{aspect}:")
        print(f"  {comparison}\n")
    
    print("=== Best Practices for Live Trading ===")
    practices = [
        "1. Start with 1-5% of total capital",
        "2. Use conservative position sizing",
        "3. Set strict stop-losses",
        "4. Monitor system constantly",
        "5. Have backup internet connection",
        "6. Test broker API thoroughly",
        "7. Implement circuit breakers",
        "8. Keep trading journal",
        "9. Review performance regularly",
        "10. Have exit strategy"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Common Live Trading Mistakes ===")
    mistakes = [
        "• Going live too early",
        "• Using too much leverage",
        "• Ignoring risk management",
        "• Emotional decision making",
        "• Not monitoring positions",
        "• Overtrading",
        "• Not having emergency procedures",
        "• Poor record keeping",
        "• Ignoring tax implications",
        "• Not reviewing performance"
    ]
    
    for mistake in mistakes:
        print(mistake)


if __name__ == "__main__":
    # Example usage
    explain_live_trading()
    
    print("\n=== Live Trading Engine Example ===")
    print("Note: This example uses PAPER trading mode for safety")
    
    # Create broker configuration (paper trading)
    broker_config = BrokerConfig(
        broker_type=BrokerType.ALPACA,
        api_key="your_api_key_here",
        api_secret="your_api_secret_here",
        paper_trading=True
    )
    
    # Create risk limits
    risk_limits = RiskLimits(
        max_position_size=0.05,  # 5% per position
        max_daily_loss=0.02,     # 2% daily loss
        max_leverage=1.5         # 1.5x leverage
    )
    
    # Create live trading engine (in paper mode)
    engine = LiveTradingEngine(
        broker_config=broker_config,
        risk_limits=risk_limits,
        trading_mode=TradingMode.PAPER
    )
    
    async def run_example():
        print("This would start the live trading engine")
        print("In a real implementation, you would:")
        print("1. Set up proper API keys")
        print("2. Configure risk limits")
        print("3. Test thoroughly")
        print("4. Start with small amounts")
        print("5. Monitor carefully")
        
        # Note: We don't actually start the engine in this example
        # as it requires valid API credentials
    
    # Run the example
    asyncio.run(run_example())