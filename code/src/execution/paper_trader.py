"""
Paper Trading Engine - AI Trading System

This module implements a comprehensive paper trading engine for testing
strategies without risking real money. It simulates real market conditions
including slippage, latency, and market impact.

Educational Note:
Paper trading is essential for strategy development. It allows you to:
1. Test strategies without financial risk
2. Understand strategy behavior in different market conditions
3. Identify bugs and edge cases
4. Build confidence before trading with real money
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import uuid
from abc import ABC, abstractmethod
import random

# Import our risk management components
from ..risk_management.position_sizer import PositionSizingManager, TradeParameters
from ..risk_management.risk_calculator import RiskCalculator
from ..risk_management.circuit_breakers import CircuitBreakerManager


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    volume: int
    open: float
    high: float
    low: float
    close: float
    vwap: Optional[float] = None


@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    position_type: PositionType
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Executed trade"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    total_value: float
    positions: Dict[str, Position]
    orders: Dict[str, Order]
    trades: List[Trade]
    margin_used: float
    margin_available: float
    leverage: float
    updated_at: datetime = field(default_factory=datetime.now)


class SlippageModel(ABC):
    """Abstract base class for slippage models"""
    
    @abstractmethod
    def calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate slippage for an order"""
        pass


class LinearSlippageModel(SlippageModel):
    """
    Linear slippage model
    
    Educational Note:
    Slippage increases with order size. Larger orders move the market
    more, resulting in worse execution prices.
    """
    
    def __init__(self, base_slippage: float = 0.0001, volume_impact: float = 0.000001):
        self.base_slippage = base_slippage  # 0.01% base slippage
        self.volume_impact = volume_impact  # Impact per share
    
    def calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate slippage based on order size and market conditions"""
        
        # Base slippage
        slippage = self.base_slippage
        
        # Volume impact
        volume_impact = self.volume_impact * order.quantity
        
        # Market conditions impact
        spread = (market_data.ask - market_data.bid) / market_data.last
        spread_impact = spread * 0.5  # Half the spread as slippage
        
        total_slippage = slippage + volume_impact + spread_impact
        
        # Directional slippage
        if order.side == OrderSide.BUY:
            return total_slippage  # Buy at higher price
        else:
            return -total_slippage  # Sell at lower price


class VolatilitySlippageModel(SlippageModel):
    """
    Volatility-based slippage model
    
    Educational Note:
    Higher volatility leads to more slippage as prices move
    more rapidly and liquidity becomes less predictable.
    """
    
    def __init__(self, base_slippage: float = 0.0001, volatility_multiplier: float = 0.5):
        self.base_slippage = base_slippage
        self.volatility_multiplier = volatility_multiplier
    
    def calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate slippage based on volatility"""
        
        # Calculate intraday volatility
        high_low_range = (market_data.high - market_data.low) / market_data.last
        
        # Adjust slippage based on volatility
        volatility_adjustment = high_low_range * self.volatility_multiplier
        
        total_slippage = self.base_slippage + volatility_adjustment
        
        # Add size impact
        size_impact = order.quantity * 0.000001
        total_slippage += size_impact
        
        if order.side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage


class CommissionModel(ABC):
    """Abstract base class for commission models"""
    
    @abstractmethod
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: int) -> float:
        """Calculate commission for a fill"""
        pass


class PerShareCommission(CommissionModel):
    """Per-share commission model"""
    
    def __init__(self, commission_per_share: float = 0.005, minimum_commission: float = 1.0):
        self.commission_per_share = commission_per_share
        self.minimum_commission = minimum_commission
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: int) -> float:
        """Calculate per-share commission"""
        commission = fill_quantity * self.commission_per_share
        return max(commission, self.minimum_commission)


class PercentCommission(CommissionModel):
    """Percentage-based commission model"""
    
    def __init__(self, commission_rate: float = 0.001, minimum_commission: float = 1.0):
        self.commission_rate = commission_rate
        self.minimum_commission = minimum_commission
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: int) -> float:
        """Calculate percentage commission"""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_rate
        return max(commission, self.minimum_commission)


class PaperTradingEngine:
    """
    Paper Trading Engine
    
    Educational Note:
    This engine simulates real trading conditions including:
    - Realistic order execution
    - Slippage and commission
    - Portfolio management
    - Risk management integration
    - Performance tracking
    """
    
    def __init__(
        self,
        initial_cash: float = 100000,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        latency_ms: int = 10
    ):
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.slippage_model = slippage_model or LinearSlippageModel()
        self.commission_model = commission_model or PerShareCommission()
        self.latency_ms = latency_ms
        
        # Portfolio and orders
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.market_data: Dict[str, MarketData] = {}
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
        # Risk management
        self.position_sizer = PositionSizingManager()
        self.risk_calculator = RiskCalculator()
        self.circuit_breaker_manager = CircuitBreakerManager()
        
        # Simulation state
        self.is_running = False
        self.current_time = datetime.now()
        
        # Statistics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.order_count = 0
        self.trade_count = 0
    
    async def place_order(self, order: Order) -> str:
        """
        Place an order
        
        Educational Note:
        Orders are not executed immediately. They go through a realistic
        execution process with latency and market conditions.
        """
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order.id
        
        # Add to orders
        self.orders[order.id] = order
        order.status = OrderStatus.SUBMITTED
        order.updated_at = self.current_time
        
        self.order_count += 1
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Process order execution
        asyncio.create_task(self._process_order(order))
        
        return order.id
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * (order.price or self._get_market_price(order.symbol))
            commission = self.commission_model.calculate_commission(order, order.price or self._get_market_price(order.symbol), order.quantity)
            total_required = required_cash + commission
            
            if self.current_cash < total_required:
                return False
        
        # Check if we have position for sell orders
        if order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                return False
        
        return True
    
    async def _process_order(self, order: Order):
        """Process order execution"""
        
        while order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
            # Get current market data
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                await asyncio.sleep(0.1)
                continue
            
            # Check if order should execute
            should_execute, execution_price = self._should_execute_order(order, market_data)
            
            if should_execute:
                # Calculate slippage
                slippage = self.slippage_model.calculate_slippage(order, market_data)
                fill_price = execution_price * (1 + slippage)
                
                # Calculate fill quantity
                fill_quantity = min(order.quantity - order.filled_quantity, 
                                   self._get_available_liquidity(order, market_data))
                
                if fill_quantity > 0:
                    # Execute fill
                    await self._execute_fill(order, fill_price, fill_quantity)
                
                # Check if order is complete
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIAL_FILLED
            
            # Check order expiry
            if self._is_order_expired(order):
                order.status = OrderStatus.CANCELLED
                break
            
            await asyncio.sleep(0.1)  # Check every 100ms
    
    def _should_execute_order(self, order: Order, market_data: MarketData) -> Tuple[bool, float]:
        """Determine if order should execute and at what price"""
        
        if order.order_type == OrderType.MARKET:
            # Market orders execute immediately
            if order.side == OrderSide.BUY:
                return True, market_data.ask
            else:
                return True, market_data.bid
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute only if price is favorable
            if order.side == OrderSide.BUY and market_data.ask <= order.price:
                return True, min(order.price, market_data.ask)
            elif order.side == OrderSide.SELL and market_data.bid >= order.price:
                return True, max(order.price, market_data.bid)
        
        elif order.order_type == OrderType.STOP_LOSS:
            # Stop orders execute when price crosses stop price
            if order.side == OrderSide.BUY and market_data.last >= order.stop_price:
                return True, market_data.last
            elif order.side == OrderSide.SELL and market_data.last <= order.stop_price:
                return True, market_data.last
        
        return False, 0.0
    
    def _get_available_liquidity(self, order: Order, market_data: MarketData) -> int:
        """Get available liquidity for order"""
        
        # Simplified liquidity model
        if order.side == OrderSide.BUY:
            return market_data.ask_size
        else:
            return market_data.bid_size
    
    async def _execute_fill(self, order: Order, fill_price: float, fill_quantity: int):
        """Execute order fill"""
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price, fill_quantity)
        
        # Update order
        order.filled_quantity += fill_quantity
        order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - fill_quantity)) + 
                               (fill_price * fill_quantity)) / order.filled_quantity
        order.commission += commission
        order.updated_at = self.current_time
        
        # Record fill
        fill = {
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': self.current_time
        }
        order.fills.append(fill)
        
        # Create trade record
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            timestamp=self.current_time
        )
        self.trades.append(trade)
        self.trade_count += 1
        
        # Update position
        self._update_position(order, fill_price, fill_quantity)
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.current_cash -= (fill_price * fill_quantity + commission)
        else:
            self.current_cash += (fill_price * fill_quantity - commission)
        
        # Update statistics
        self.total_commission += commission
        self.total_slippage += abs(fill_price - self._get_market_price(order.symbol)) * fill_quantity
    
    def _update_position(self, order: Order, fill_price: float, fill_quantity: int):
        """Update position after fill"""
        
        symbol = order.symbol
        current_position = self.positions.get(symbol)
        
        if not current_position:
            # New position
            if order.side == OrderSide.BUY:
                position_type = PositionType.LONG
                quantity = fill_quantity
            else:
                position_type = PositionType.SHORT
                quantity = -fill_quantity
            
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill_price,
                market_value=fill_price * abs(quantity),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                position_type=position_type,
                created_at=self.current_time,
                updated_at=self.current_time
            )
            self.positions[symbol] = position
        
        else:
            # Update existing position
            if order.side == OrderSide.BUY:
                # Adding to long position or covering short
                if current_position.quantity >= 0:
                    # Adding to long
                    total_cost = current_position.avg_price * current_position.quantity + fill_price * fill_quantity
                    total_quantity = current_position.quantity + fill_quantity
                    current_position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    current_position.quantity = total_quantity
                else:
                    # Covering short
                    cover_quantity = min(fill_quantity, abs(current_position.quantity))
                    realized_pnl = (current_position.avg_price - fill_price) * cover_quantity
                    current_position.realized_pnl += realized_pnl
                    current_position.quantity += fill_quantity
                    
                    if current_position.quantity > 0:
                        current_position.position_type = PositionType.LONG
                        current_position.avg_price = fill_price
            
            else:  # SELL
                # Selling from long position or adding to short
                if current_position.quantity > 0:
                    # Selling from long
                    sell_quantity = min(fill_quantity, current_position.quantity)
                    realized_pnl = (fill_price - current_position.avg_price) * sell_quantity
                    current_position.realized_pnl += realized_pnl
                    current_position.quantity -= sell_quantity
                    
                    if current_position.quantity <= 0:
                        current_position.position_type = PositionType.FLAT
                        current_position.quantity = 0
                else:
                    # Adding to short
                    total_cost = current_position.avg_price * abs(current_position.quantity) + fill_price * fill_quantity
                    total_quantity = abs(current_position.quantity) + fill_quantity
                    current_position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    current_position.quantity = -total_quantity
                    current_position.position_type = PositionType.SHORT
            
            current_position.updated_at = self.current_time
            
            # Remove position if flat
            if current_position.quantity == 0:
                del self.positions[symbol]
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        market_data = self.market_data.get(symbol)
        if market_data:
            return market_data.last
        return 0.0
    
    def _is_order_expired(self, order: Order) -> bool:
        """Check if order has expired"""
        
        if order.time_in_force == "DAY":
            # Order expires at end of day
            return self.current_time.date() > order.created_at.date()
        
        return False
    
    def update_market_data(self, market_data: MarketData):
        """Update market data and recalculate portfolio values"""
        
        self.market_data[market_data.symbol] = market_data
        self.current_time = market_data.timestamp
        
        # Update position values
        total_position_value = 0.0
        for position in self.positions.values():
            current_price = self._get_market_price(position.symbol)
            position.market_value = current_price * abs(position.quantity)
            
            # Calculate unrealized P&L
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
            else:  # SHORT
                position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
            
            total_position_value += position.market_value
        
        # Calculate total portfolio value
        total_value = self.current_cash + total_position_value
        
        # Update equity curve
        self.equity_curve.append((self.current_time, total_value))
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state"""
        
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self.current_cash + total_position_value
        
        return Portfolio(
            cash=self.current_cash,
            total_value=total_value,
            positions=self.positions.copy(),
            orders=self.orders.copy(),
            trades=self.trades.copy(),
            margin_used=0.0,  # Simplified for paper trading
            margin_available=total_value,
            leverage=total_position_value / total_value if total_value > 0 else 0,
            updated_at=self.current_time
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        if not self.equity_curve:
            return {}
        
        # Basic metrics
        current_value = self.equity_curve[-1][1]
        initial_value = self.initial_cash
        total_return = (current_value - initial_value) / initial_value
        
        # Calculate daily returns if not available
        if not self.daily_returns and len(self.equity_curve) > 1:
            values = [point[1] for point in self.equity_curve]
            self.daily_returns = [(values[i] - values[i-1]) / values[i-1] 
                                for i in range(1, len(values))]
        
        # Risk metrics
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            volatility = returns_series.std() * np.sqrt(252)
            sharpe_ratio = (returns_series.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            equity_values = [point[1] for point in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0.0
            
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # Trading statistics
        win_trades = sum(1 for trade in self.trades if self._get_trade_pnl(trade) > 0)
        win_rate = win_trades / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'current_value': current_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'order_count': self.order_count,
            'trade_count': self.trade_count,
            'positions_count': len(self.positions),
            'cash': self.current_cash,
            'leverage': sum(pos.market_value for pos in self.positions.values()) / current_value if current_value > 0 else 0
        }
    
    def _get_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade (simplified)"""
        # This is a simplified calculation
        # In reality, you'd need to match opening and closing trades
        return 0.0
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        
        order = self.orders.get(order_id)
        if order and order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
            order.status = OrderStatus.CANCELLED
            order.updated_at = self.current_time
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get status of an order"""
        order = self.orders.get(order_id)
        return order.status if order else None
    
    def reset(self):
        """Reset the paper trading engine"""
        
        self.current_cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.order_count = 0
        self.trade_count = 0


def explain_paper_trading():
    """
    Educational explanation of paper trading
    """
    
    print("=== Paper Trading Educational Guide ===\n")
    
    concepts = {
        'Paper Trading': "Simulated trading without real money, essential for strategy testing",
        
        'Slippage': "Difference between expected and actual execution price",
        
        'Commission': "Transaction costs charged by brokers",
        
        'Latency': "Delay between order submission and execution",
        
        'Market Impact': "Price movement caused by large orders",
        
        'Liquidity': "Ability to execute orders without affecting price",
        
        'Order Types': "Different ways to execute trades (market, limit, stop, etc.)",
        
        'Position Management': "Tracking and managing open positions",
        
        'Risk Management': "Controlling risk through position sizing and stops"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Paper Trading Best Practices ===")
    print("1. Trade with realistic amounts - don't use $1M if you only have $10K")
    print("2. Include realistic slippage and commission")
    print("3. Test over different market conditions")
    print("4. Keep detailed records of all trades")
    print("5. Review performance regularly")
    print("6. Don't cherry-pick results - be honest about performance")
    print("7. Paper trade for at least 3-6 months before going live")
    
    print("\n=== Common Paper Trading Mistakes ===")
    print("• Ignoring slippage and commission")
    print("• Using unrealistic position sizes")
    print("• Not testing during market volatility")
    print("• Overfitting to historical data")
    print("• Ignoring psychological factors")
    print("• Not having a clear exit strategy")
    print("• Switching to live trading too early")


if __name__ == "__main__":
    # Example usage
    explain_paper_trading()
    
    # Create paper trading engine
    engine = PaperTradingEngine(
        initial_cash=100000,
        slippage_model=LinearSlippageModel(),
        commission_model=PerShareCommission()
    )
    
    # Create sample market data
    market_data = MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        bid=149.50,
        ask=150.50,
        bid_size=1000,
        ask_size=1000,
        last=150.00,
        volume=1000000,
        open=148.00,
        high=151.00,
        low=147.50,
        close=150.00
    )
    
    print("\n=== Paper Trading Example ===")
    print(f"Initial cash: ${engine.initial_cash:,.2f}")
    
    # Update market data
    engine.update_market_data(market_data)
    
    # Create and place a buy order
    buy_order = Order(
        id=str(uuid.uuid4()),
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    async def run_example():
        # Place order
        order_id = await engine.place_order(buy_order)
        print(f"Placed buy order: {order_id}")
        
        # Wait for execution
        await asyncio.sleep(0.5)
        
        # Check portfolio
        portfolio = engine.get_portfolio()
        print(f"Portfolio value: ${portfolio.total_value:,.2f}")
        print(f"Cash: ${portfolio.cash:,.2f}")
        print(f"Positions: {list(portfolio.positions.keys())}")
        
        # Get performance metrics
        metrics = engine.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Run the example
    asyncio.run(run_example())