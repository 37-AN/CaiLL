"""
Backtesting Engine - AI Trading System

This module implements a comprehensive backtesting engine for validating
trading strategies on historical data with realistic market conditions.

Educational Note:
Backtesting is essential for strategy development. It allows you to:
1. Test strategies on historical data without risking real money
2. Understand strategy behavior in different market conditions
3. Identify potential issues before live trading
4. Optimize strategy parameters
5. Build confidence in strategy performance

Remember: Past performance does not guarantee future results!
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import uuid
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import our trading components
from ..execution.paper_trader import (
    Order, OrderType, OrderSide, OrderStatus, Position, PositionType,
    Trade, Portfolio, MarketData, SlippageModel, CommissionModel
)
from ..risk_management.position_sizer import PositionSizingManager, TradeParameters
from ..risk_management.risk_calculator import RiskCalculator
from ..risk_management.circuit_breakers import CircuitBreakerManager


class BacktestMode(Enum):
    """Backtesting modes"""
    VECTORIZED = "vectorized"  # Fast, vectorized backtesting
    EVENT_DRIVEN = "event_driven"  # Realistic event-by-event simulation
    HYBRID = "hybrid"  # Combination of both


class DataFrequency(Enum):
    """Data frequencies"""
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MarketRegime(Enum):
    """Market regimes for analysis"""
    BULL = "bull"      # Rising market
    BEAR = "bear"      # Falling market
    SIDEWAYS = "sideways"  # Range-bound market
    VOLATILE = "volatile"   # High volatility
    LOW_VOL = "low_vol"     # Low volatility


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000
    data_frequency: DataFrequency = DataFrequency.DAILY
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    benchmark: Optional[str] = None
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    allow_short_selling: bool = True
    allow_margin: bool = True
    position_limit: float = 0.2  # 20% max per position
    rebalance_frequency: str = "daily"
    lookback_window: int = 252  # 1 year lookback for indicators
    min_trade_size: float = 100  # Minimum trade size
    max_positions: int = 20  # Maximum number of positions
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results of a backtest"""
    strategy_name: str
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float
    
    # Portfolio metrics
    final_portfolio_value: float
    equity_curve: pd.Series
    returns: pd.Series
    positions_history: List[Dict]
    trades_history: List[Trade]
    orders_history: List[Order]
    
    # Risk metrics
    var_95: float
    cvar_95: float
    beta: Optional[float]
    alpha: Optional[float]
    information_ratio: Optional[float]
    
    # Regime analysis
    regime_performance: Dict[MarketRegime, Dict[str, float]]
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.position_sizer = PositionSizingManager()
        self.risk_calculator = RiskCalculator()
        self.circuit_breakers = CircuitBreakerManager()
    
    @abstractmethod
    async def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """Generate trading signals based on data"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass
    
    async def on_data(self, data: pd.DataFrame, current_time: datetime, portfolio: Portfolio) -> List[Order]:
        """Process new data and generate orders"""
        
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = await self.generate_signals(data_with_indicators, current_time)
        
        # Convert signals to orders
        orders = []
        for signal in signals:
            order = await self._signal_to_order(signal, portfolio, current_time)
            if order:
                orders.append(order)
        
        return orders
    
    async def _signal_to_order(self, signal: Dict, portfolio: Portfolio, current_time: datetime) -> Optional[Order]:
        """Convert trading signal to order"""
        
        symbol = signal.get('symbol')
        action = signal.get('action')  # 'buy', 'sell', 'hold'
        quantity = signal.get('quantity', 0)
        price = signal.get('price')
        order_type = OrderType(signal.get('order_type', 'market'))
        
        if action == 'hold' or quantity == 0:
            return None
        
        # Create order
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if action == 'buy' else OrderSide.SELL,
            order_type=order_type,
            quantity=abs(quantity),
            price=price,
            created_at=current_time
        )
        
        return order


class BacktestEngine:
    """
    Comprehensive Backtesting Engine
    
    Educational Note:
    This engine provides realistic backtesting with proper handling of:
    - Transaction costs (commission and slippage)
    - Market impact
    - Portfolio constraints
    - Risk management
    - Realistic order execution
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.current_time: Optional[datetime] = None
        
        # Trading components
        self.position_sizer = PositionSizingManager()
        self.risk_calculator = RiskCalculator()
        self.circuit_breakers = CircuitBreakerManager()
        
        # State tracking
        self.portfolio: Optional[Portfolio] = None
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.positions_history: List[Dict] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.regime_tracker: Dict[datetime, MarketRegime] = {}
        
        # Data handlers
        self.market_data_simulator = MarketDataSimulator(config)
        self.execution_simulator = ExecutionSimulator(config)
    
    async def run_backtest(self, strategy: TradingStrategy, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run backtest with given strategy and data"""
        
        print(f"Starting backtest for strategy: {strategy.name}")
        print(f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        
        # Initialize
        await self._initialize_backtest(data)
        
        # Main backtesting loop
        if self.config.mode == BacktestMode.EVENT_DRIVEN:
            await self._run_event_driven(strategy, data)
        elif self.config.mode == BacktestMode.VECTORIZED:
            await self._run_vectorized(strategy, data)
        else:
            await self._run_hybrid(strategy, data)
        
        # Calculate results
        result = await self._calculate_results(strategy)
        
        print(f"Backtest completed. Total return: {result.total_return:.2%}")
        print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"Max drawdown: {result.max_drawdown:.2%}")
        print(f"Total trades: {result.total_trades}")
        
        return result
    
    async def _initialize_backtest(self, data: Dict[str, pd.DataFrame]):
        """Initialize backtest state"""
        
        # Cache data
        self.data_cache = data
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            cash=self.config.initial_cash,
            total_value=self.config.initial_cash,
            positions={},
            orders={},
            trades=[],
            margin_used=0.0,
            margin_available=self.config.initial_cash,
            leverage=0.0,
            updated_at=self.config.start_date
        )
        
        # Initialize equity curve
        self.equity_curve = [(self.config.start_date, self.config.initial_cash)]
        
        # Initialize market data simulator
        await self.market_data_simulator.initialize(data)
        
        # Initialize execution simulator
        await self.execution_simulator.initialize()
    
    async def _run_event_driven(self, strategy: TradingStrategy, data: Dict[str, pd.DataFrame]):
        """Run event-driven backtesting"""
        
        # Get all unique timestamps
        all_timestamps = set()
        for symbol, df in data.items():
            all_timestamps.update(df.index)
        
        all_timestamps = sorted(all_timestamps)
        
        # Process each timestamp
        for i, timestamp in enumerate(all_timestamps):
            if timestamp < self.config.start_date or timestamp > self.config.end_date:
                continue
            
            self.current_time = timestamp
            
            # Get current data for all symbols
            current_data = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    # Get lookback window
                    end_idx = df.index.get_loc(timestamp)
                    start_idx = max(0, end_idx - self.config.lookback_window + 1)
                    current_data[symbol] = df.iloc[start_idx:end_idx + 1]
            
            if not current_data:
                continue
            
            # Update portfolio with current prices
            await self._update_portfolio_values(current_data)
            
            # Generate and execute orders
            orders = await strategy.on_data(
                pd.concat(current_data.values(), keys=current_data.keys()),
                timestamp,
                self.portfolio
            )
            
            # Execute orders
            await self._execute_orders(orders, current_data)
            
            # Risk management checks
            await self._risk_management_check(current_data)
            
            # Record state
            await self._record_state(timestamp)
            
            # Progress indicator
            if i % 100 == 0:
                progress = (i / len(all_timestamps)) * 100
                print(f"Progress: {progress:.1f}% - Portfolio value: ${self.portfolio.total_value:,.2f}")
    
    async def _run_vectorized(self, strategy: TradingStrategy, data: Dict[str, pd.DataFrame]):
        """Run vectorized backtesting (faster but less realistic)"""
        
        # This is a simplified vectorized implementation
        # In practice, you'd want more sophisticated vectorization
        
        combined_data = pd.concat(data.values(), keys=data.keys())
        combined_data = combined_data.sort_index()
        
        # Filter by date range
        mask = (combined_data.index >= self.config.start_date) & (combined_data.index <= self.config.end_date)
        filtered_data = combined_data[mask]
        
        # Calculate indicators for all data
        indicators = strategy.calculate_indicators(filtered_data)
        
        # Generate signals (simplified)
        signals = await strategy.generate_signals(indicators, self.config.end_date)
        
        # Simulate execution (simplified)
        for signal in signals:
            await self._simulate_vectorized_trade(signal, filtered_data)
    
    async def _run_hybrid(self, strategy: TradingStrategy, data: Dict[str, pd.DataFrame]):
        """Run hybrid backtesting (combination of event-driven and vectorized)"""
        
        # Use vectorized for signal generation
        # Use event-driven for execution
        
        # This is a placeholder for hybrid implementation
        await self._run_event_driven(strategy, data)
    
    async def _update_portfolio_values(self, current_data: Dict[str, pd.DataFrame]):
        """Update portfolio values with current market data"""
        
        total_position_value = 0.0
        
        for symbol, position in self.portfolio.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol].iloc[-1]['close']
                position.market_value = abs(position.quantity) * current_price
                
                # Update unrealized P&L
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
                
                total_position_value += position.market_value
        
        # Update portfolio totals
        self.portfolio.total_value = self.portfolio.cash + total_position_value
        self.portfolio.leverage = total_position_value / self.portfolio.total_value if self.portfolio.total_value > 0 else 0
        self.portfolio.updated_at = self.current_time
    
    async def _execute_orders(self, orders: List[Order], current_data: Dict[str, pd.DataFrame]):
        """Execute orders with realistic simulation"""
        
        for order in orders:
            if order.symbol not in current_data:
                continue
            
            # Simulate order execution
            execution_result = await self.execution_simulator.execute_order(
                order, current_data[order.symbol].iloc[-1]
            )
            
            if execution_result:
                # Update portfolio
                await self._process_trade(execution_result, order)
                
                # Record order
                self.orders.append(order)
    
    async def _process_trade(self, execution_result: Dict, order: Order):
        """Process executed trade"""
        
        # Create trade record
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=execution_result['quantity'],
            price=execution_result['price'],
            commission=execution_result['commission'],
            timestamp=self.current_time
        )
        
        self.trades.append(trade)
        
        # Update portfolio cash
        if order.side == OrderSide.BUY:
            self.portfolio.cash -= (execution_result['price'] * execution_result['quantity'] + execution_result['commission'])
        else:
            self.portfolio.cash += (execution_result['price'] * execution_result['quantity'] - execution_result['commission'])
        
        # Update or create position
        await self._update_position(order, execution_result)
    
    async def _update_position(self, order: Order, execution_result: Dict):
        """Update position after trade"""
        
        symbol = order.symbol
        quantity = execution_result['quantity']
        price = execution_result['price']
        
        if symbol not in self.portfolio.positions:
            # New position
            position_type = PositionType.LONG if order.side == OrderSide.BUY else PositionType.SHORT
            actual_quantity = quantity if order.side == OrderSide.BUY else -quantity
            
            position = Position(
                symbol=symbol,
                quantity=actual_quantity,
                avg_price=price,
                market_value=price * abs(actual_quantity),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                position_type=position_type,
                created_at=self.current_time,
                updated_at=self.current_time
            )
            
            self.portfolio.positions[symbol] = position
        
        else:
            # Update existing position
            position = self.portfolio.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # Adding to position
                if position.quantity >= 0:
                    # Adding to long
                    total_cost = position.avg_price * position.quantity + price * quantity
                    total_quantity = position.quantity + quantity
                    position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    position.quantity = total_quantity
                else:
                    # Covering short
                    cover_quantity = min(quantity, abs(position.quantity))
                    realized_pnl = (position.avg_price - price) * cover_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity += quantity
                    
                    if position.quantity > 0:
                        position.position_type = PositionType.LONG
                        position.avg_price = price
            
            else:  # SELL
                # Selling from position
                if position.quantity > 0:
                    # Selling from long
                    sell_quantity = min(quantity, position.quantity)
                    realized_pnl = (price - position.avg_price) * sell_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= sell_quantity
                    
                    if position.quantity <= 0:
                        position.position_type = PositionType.FLAT
                        position.quantity = 0
                else:
                    # Adding to short
                    total_cost = position.avg_price * abs(position.quantity) + price * quantity
                    total_quantity = abs(position.quantity) + quantity
                    position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    position.quantity = -total_quantity
                    position.position_type = PositionType.SHORT
            
            position.updated_at = self.current_time
            
            # Remove position if flat
            if position.quantity == 0:
                del self.portfolio.positions[symbol]
    
    async def _risk_management_check(self, current_data: Dict[str, pd.DataFrame]):
        """Perform risk management checks"""
        
        # Prepare data for circuit breakers
        portfolio_data = {
            'portfolio_value': self.portfolio.total_value,
            'initial_portfolio_value': self.config.initial_cash,
            'positions': {
                symbol: {
                    'current_value': pos.market_value,
                    'entry_value': pos.avg_price * abs(pos.quantity)
                }
                for symbol, pos in self.portfolio.positions.items()
            },
            'total_position_value': sum(pos.market_value for pos in self.portfolio.positions.values()),
            'portfolio_history': [value for _, value in self.equity_curve[-100:]],  # Last 100 points
            'portfolio_returns': self.daily_returns[-252:]  # Last year of returns
        }
        
        # Check circuit breakers
        triggered_breakers = self.circuit_breaker_manager.check_circuit_breakers(portfolio_data)
        
        if triggered_breakers:
            print(f"Risk warning: Circuit breakers triggered: {triggered_breakers}")
            
            # Could implement automatic position reduction here
            for breaker_name in triggered_breakers:
                if "loss" in breaker_name.lower() or "drawdown" in breaker_name.lower():
                    # Reduce positions on major loss events
                    await self._emergency_position_reduction()
    
    async def _emergency_position_reduction(self):
        """Emergency position reduction"""
        
        print("Emergency: Reducing positions due to risk limits")
        
        # Liquidate all positions
        for symbol in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[symbol]
            
            # Create liquidation order
            liquidation_order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=abs(position.quantity),
                created_at=self.current_time
            )
            
            # Simulate immediate execution
            current_data = self.data_cache.get(symbol, pd.DataFrame())
            if not current_data.empty and self.current_time in current_data.index:
                current_price = current_data.loc[self.current_time, 'close']
                
                execution_result = {
                    'quantity': abs(position.quantity),
                    'price': current_price,
                    'commission': abs(position.quantity) * current_price * self.config.commission_rate
                }
                
                await self._process_trade(liquidation_order, execution_result)
    
    async def _record_state(self, timestamp: datetime):
        """Record current state for analysis"""
        
        # Record equity curve
        self.equity_curve.append((timestamp, self.portfolio.total_value))
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            current_value = self.equity_curve[-1][1]
            daily_return = (current_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        # Record positions
        positions_snapshot = {
            symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'position_type': pos.position_type.value
            }
            for symbol, pos in self.portfolio.positions.items()
        }
        
        self.positions_history.append({
            'timestamp': timestamp,
            'positions': positions_snapshot,
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'leverage': self.portfolio.leverage
        })
        
        # Detect market regime
        regime = self._detect_market_regime()
        self.regime_tracker[timestamp] = regime
    
    def _detect_market_regime(self) -> MarketRegime:
        """Detect current market regime"""
        
        if len(self.daily_returns) < 20:
            return MarketRegime.SIDEWAYS
        
        recent_returns = self.daily_returns[-20:]
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Classify regime
        if avg_return > 0.01:  # > 1% average return
            return MarketRegime.BULL
        elif avg_return < -0.01:  # < -1% average return
            return MarketRegime.BEAR
        elif volatility > 0.02:  # > 2% daily volatility
            return MarketRegime.VOLATILE
        elif volatility < 0.005:  # < 0.5% daily volatility
            return MarketRegime.LOW_VOL
        else:
            return MarketRegime.SIDEWAYS
    
    async def _simulate_vectorized_trade(self, signal: Dict, data: pd.DataFrame):
        """Simulate trade in vectorized manner"""
        
        # This is a simplified vectorized implementation
        # In practice, you'd want more sophisticated handling
        
        symbol = signal.get('symbol')
        action = signal.get('action')
        
        if symbol not in data.columns:
            return
        
        prices = data[symbol]['close']
        
        # Simple buy/hold strategy simulation
        if action == 'buy':
            # Calculate returns
            returns = prices.pct_change().fillna(0)
            
            # Apply to equity curve (simplified)
            for i, (timestamp, price) in enumerate(prices.items()):
                if i >= len(self.equity_curve):
                    # Update portfolio value based on price changes
                    if len(self.equity_curve) > 0:
                        last_value = self.equity_curve[-1][1]
                        new_value = last_value * (1 + returns.iloc[i])
                        self.equity_curve.append((timestamp, new_value))
    
    async def _calculate_results(self, strategy: TradingStrategy) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        if not self.equity_curve:
            raise ValueError("No equity curve data available")
        
        # Convert to pandas for easier analysis
        equity_series = pd.Series(
            [value for _, value in self.equity_curve],
            index=[timestamp for timestamp, _ in self.equity_curve]
        )
        
        returns_series = pd.Series(self.daily_returns)
        
        # Basic performance metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_cash) / self.config.initial_cash
        days = (equity_series.index[-1] - equity_series.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
        
        # Risk-adjusted ratios
        risk_free_rate = 0.02  # 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if self._calculate_trade_pnl(t) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        winning_pnls = [self._calculate_trade_pnl(t) for t in self.trades if self._calculate_trade_pnl(t) > 0]
        losing_pnls = [abs(self._calculate_trade_pnl(t)) for t in self.trades if self._calculate_trade_pnl(t) < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        total_wins = sum(winning_pnls)
        total_losses = sum(losing_pnls)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average trade duration (simplified)
        avg_trade_duration = 1.0  # Placeholder
        
        # Risk metrics
        var_95 = returns_series.quantile(0.05) if len(returns_series) > 0 else 0
        cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series) > 0 else 0
        
        # Beta and Alpha (if benchmark provided)
        beta = None
        alpha = None
        information_ratio = None
        
        if self.config.benchmark and self.config.benchmark in self.data_cache:
            benchmark_data = self.data_cache[self.config.benchmark]
            benchmark_returns = benchmark_data['close'].pct_change().reindex(returns_series.index).fillna(0)
            
            if len(benchmark_returns) == len(returns_series):
                # Calculate beta
                covariance = np.cov(returns_series, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
                
                # Information ratio
                active_returns = returns_series - benchmark_returns
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        
        # Regime analysis
        regime_performance = self._analyze_regime_performance()
        
        return BacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            final_portfolio_value=self.portfolio.total_value,
            equity_curve=equity_series,
            returns=returns_series,
            positions_history=self.positions_history,
            trades_history=self.trades,
            orders_history=self.orders,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            regime_performance=regime_performance,
            metadata={
                'strategy_config': strategy.config,
                'final_positions': len(self.portfolio.positions),
                'final_cash': self.portfolio.cash,
                'max_leverage': max([p['leverage'] for p in self.positions_history]) if self.positions_history else 0
            }
        )
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade (simplified)"""
        # This is a simplified calculation
        # In reality, you'd need to match opening and closing trades
        return 0.0
    
    def _analyze_regime_performance(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Analyze performance across different market regimes"""
        
        regime_performance = {}
        
        for regime in MarketRegime:
            regime_periods = [
                timestamp for timestamp, r in self.regime_tracker.items() 
                if r == regime
            ]
            
            if not regime_periods:
                continue
            
            # Calculate returns for this regime
            regime_returns = []
            for i, timestamp in enumerate(self.equity_curve):
                if timestamp in regime_periods and i > 0:
                    prev_value = self.equity_curve[i-1][1]
                    current_value = self.equity_curve[i][1]
                    regime_returns.append((current_value - prev_value) / prev_value)
            
            if regime_returns:
                regime_performance[regime] = {
                    'total_return': np.prod([1 + r for r in regime_returns]) - 1,
                    'volatility': np.std(regime_returns) * np.sqrt(252),
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252) if np.std(regime_returns) > 0 else 0,
                    'max_drawdown': min([cumulative_drawdown(regime_returns[:i+1]) for i in range(len(regime_returns))]),
                    'periods': len(regime_returns)
                }
        
        return regime_performance


class MarketDataSimulator:
    """Simulates realistic market data for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    async def initialize(self, data: Dict[str, pd.DataFrame]):
        """Initialize with historical data"""
        self.data_cache = data
    
    async def get_market_data(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get market data for symbol at timestamp"""
        
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        
        if timestamp not in df.index:
            return None
        
        row = df.loc[timestamp]
        
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'bid': row['close'] * 0.999,  # Simulated bid
            'ask': row['close'] * 1.001,  # Simulated ask
            'bid_size': row['volume'] // 4,
            'ask_size': row['volume'] // 4
        }


class ExecutionSimulator:
    """Simulates realistic order execution"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    async def initialize(self):
        """Initialize execution simulator"""
        pass
    
    async def execute_order(self, order: Order, market_data: Dict) -> Optional[Dict]:
        """Execute order with realistic simulation"""
        
        # Simulate execution based on order type
        if order.order_type == OrderType.MARKET:
            return await self._execute_market_order(order, market_data)
        elif order.order_type == OrderType.LIMIT:
            return await self._execute_limit_order(order, market_data)
        else:
            return await self._execute_simple_order(order, market_data)
    
    async def _execute_market_order(self, order: Order, market_data: Dict) -> Dict:
        """Execute market order"""
        
        # Get execution price
        if order.side == OrderSide.BUY:
            base_price = market_data['ask']
        else:
            base_price = market_data['bid']
        
        # Apply slippage
        slippage = np.random.normal(0, self.config.slippage_rate)
        execution_price = base_price * (1 + slippage)
        
        # Calculate commission
        commission = order.quantity * execution_price * self.config.commission_rate
        
        return {
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'timestamp': order.created_at
        }
    
    async def _execute_limit_order(self, order: Order, market_data: Dict) -> Optional[Dict]:
        """Execute limit order"""
        
        # Check if limit price is met
        if order.side == OrderSide.BUY and market_data['ask'] <= order.price:
            execution_price = min(order.price, market_data['ask'])
        elif order.side == OrderSide.SELL and market_data['bid'] >= order.price:
            execution_price = max(order.price, market_data['bid'])
        else:
            return None  # Limit order not executed
        
        # Calculate commission
        commission = order.quantity * execution_price * self.config.commission_rate
        
        return {
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'timestamp': order.created_at
        }
    
    async def _execute_simple_order(self, order: Order, market_data: Dict) -> Dict:
        """Simple order execution"""
        
        execution_price = market_data['close']
        commission = order.quantity * execution_price * self.config.commission_rate
        
        return {
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'timestamp': order.created_at
        }


def cumulative_drawdown(returns: List[float]) -> float:
    """Calculate cumulative drawdown"""
    cumulative = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def explain_backtesting():
    """
    Educational explanation of backtesting
    """
    
    print("=== Backtesting Educational Guide ===\n")
    
    concepts = {
        'Backtesting': "Testing trading strategies on historical data to evaluate performance",
        
        'Event-Driven Backtesting': "Simulates trades tick-by-tick for maximum realism",
        
        'Vectorized Backtesting': "Fast calculation using vector operations, less realistic",
        
        'Market Regime': "Different market conditions (bull, bear, volatile, sideways)",
        
        'Slippage': "Difference between expected and actual execution price",
        
        'Commission': "Transaction costs that reduce strategy returns",
        
        'Maximum Drawdown': "Largest peak-to-trough decline in portfolio value",
        
        'Sharpe Ratio': "Risk-adjusted return measure (higher is better)",
        
        'Walk-Forward Analysis': "Testing strategy on rolling time windows",
        
        'Monte Carlo Simulation': "Testing strategy performance under uncertainty"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Backtesting Best Practices ===")
    practices = [
        "1. Use out-of-sample data for testing",
        "2. Include realistic transaction costs",
        "3. Test across different market regimes",
        "4. Avoid overfitting to historical data",
        "5. Use proper statistical validation",
        "6. Consider market impact and liquidity",
        "7. Test with multiple parameter sets",
        "8. Validate with walk-forward analysis",
        "9. Perform stress testing",
        "10. Understand limitations of historical testing"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Common Backtesting Pitfalls ===")
    pitfalls = [
        "• Ignoring transaction costs and slippage",
        "• Using future data (lookahead bias)",
        "• Overfitting to historical data",
        "• Not testing across market regimes",
        "• Ignoring survivorship bias",
        "• Using insufficient data",
        "• Not accounting for liquidity constraints",
        "• Ignoring market impact",
        "• Not validating statistical significance",
        "• Assuming past performance predicts future results"
    ]
    
    for pitfall in pitfalls:
        print(pitfall)
    
    print("\n=== Backtesting vs Live Trading ===")
    differences = {
        "Data Quality": "Backtest: Clean historical data | Live: Real-time noisy data",
        "Execution": "Backtest: Simulated execution | Live: Real market execution",
        "Costs": "Backtest: Estimated costs | Live: Actual costs",
        "Liquidity": "Backtest: Assumed infinite | Live: Limited liquidity",
        "Market Impact": "Backtest: Often ignored | Live: Real impact",
        "Psychology": "Backtest: No emotion | Live: Real psychological pressure",
        "Latency": "Backtest: Instant execution | Live: Network delays",
        "Errors": "Backtest: Controlled environment | Live: Unexpected issues"
    }
    
    for aspect, comparison in differences.items():
        print(f"{aspect}:")
        print(f"  {comparison}\n")


if __name__ == "__main__":
    # Example usage
    explain_backtesting()
    
    print("\n=== Backtesting Engine Example ===")
    print("To use the backtesting engine:")
    print("1. Create a BacktestConfig with your parameters")
    print("2. Implement a TradingStrategy class")
    print("3. Prepare historical market data")
    print("4. Run the backtest")
    print("5. Analyze the results")
    
    # Example configuration
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_cash=100000,
        data_frequency=DataFrequency.DAILY,
        mode=BacktestMode.EVENT_DRIVEN
    )
    
    print(f"\nSample configuration:")
    print(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Initial cash: ${config.initial_cash:,}")
    print(f"  Mode: {config.mode.value}")
    print(f"  Frequency: {config.data_frequency.value}")
    
    print("\nThe engine provides:")
    print("• Realistic transaction cost modeling")
    print("• Multiple backtesting modes")
    print("• Comprehensive performance metrics")
    print("• Market regime analysis")
    print("• Risk management integration")
    print("• Detailed trade-by-trade analysis")