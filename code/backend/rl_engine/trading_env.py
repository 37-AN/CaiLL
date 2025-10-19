"""
Trading Environment - Phase 3.1

This module implements a custom OpenAI Gym-compatible trading environment
that simulates real market conditions for reinforcement learning agents.

Educational Note:
The trading environment is the foundation of any RL trading system.
It defines how agents interact with markets, receive rewards, and learn
optimal trading strategies through trial and error.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings

# Import our feature pipeline
from ..features.feature_pipeline import FeaturePipeline, FeatureSet
from ..features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class AssetType(Enum):
    """Types of tradable assets"""
    STOCK = "stock"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"

class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class MarketState:
    """Current market state"""
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    features: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class Position:
    """Current position"""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    total_pnl: float = 0.0
    
    def update_total_value(self):
        """Update total portfolio value"""
        position_value = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.total_value = self.cash + position_value
        self.total_pnl = position_value

@dataclass
class Transaction:
    """Transaction record"""
    timestamp: datetime
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float
    pnl: float = 0.0

class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    
    Educational Notes:
    - This environment follows the OpenAI Gym interface for compatibility
    - State space includes market features and portfolio information
    - Action space includes position sizing and order types
    - Reward function balances profit, risk, and transaction costs
    """
    
    def __init__(self, 
                 symbol: str,
                 initial_cash: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 max_position_size: int = 1000,
                 lookback_window: int = 100,
                 episode_length: int = 252,
                 asset_type: AssetType = AssetType.STOCK):
        
        super().__init__()
        
        # Environment parameters
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.asset_type = asset_type
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline()
        
        # State tracking
        self.current_step = 0
        self.max_steps = 0
        self.market_data = None
        self.current_state = None
        self.portfolio = None
        self.transaction_history = []
        
        # Performance tracking
        self.episode_returns = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Define action and observation spaces
        self._setup_spaces()
        
        logger.info(f"TradingEnvironment initialized for {symbol}")
    
    def _setup_spaces(self):
        """
        Setup action and observation spaces
        
        Educational: Action space defines what agents can do.
        Observation space defines what agents can see.
        """
        
        # Action space: [position_size, order_type, price_offset]
        # position_size: -1 to 1 (normalized, -1 = max short, 1 = max long)
        # order_type: 0=market, 1=limit, 2=stop
        # price_offset: -0.01 to 0.01 (for limit/stop orders)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -0.01]),
            high=np.array([1.0, 2.0, 0.01]),
            dtype=np.float32
        )
        
        # Observation space will be set dynamically based on features
        # We'll use a Dict space for better organization
        self.observation_space = spaces.Dict({
            'prices': spaces.Box(low=0, high=np.inf, shape=(self.lookback_window, 5), dtype=np.float32),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32),  # Will be adjusted
            'portfolio': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state
        
        Educational: Each episode starts with a clean portfolio
        and a random starting point in the historical data.
        """
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio = Portfolio(cash=self.initial_cash)
        
        # Reset tracking
        self.current_step = 0
        self.transaction_history = []
        self.episode_returns = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Reset market data to random starting point
        if self.market_data is not None:
            max_start = len(self.market_data) - self.episode_length - self.lookback_window
            if max_start > 0:
                start_idx = np.random.randint(0, max_start)
                self.current_step = start_idx + self.lookback_window
                self.max_steps = start_idx + self.episode_length
        
        # Get initial observation
        observation = self._get_observation()
        
        logger.info(f"Environment reset. Starting step: {self.current_step}")
        
        return observation, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Educational: This is where the agent interacts with the market.
        The action is executed, market moves, and reward is calculated.
        """
        if self.market_data is None or self.current_step >= len(self.market_data) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Parse action
        position_size, order_type, price_offset = action
        position_size = np.clip(position_size, -1.0, 1.0)
        
        # Execute action
        transaction_cost = self._execute_action(position_size, order_type, price_offset)
        
        # Move to next time step
        self.current_step += 1
        
        # Update portfolio value
        self._update_portfolio()
        
        # Calculate reward
        reward = self._calculate_reward(transaction_cost)
        
        # Check if episode is done
        done = self._is_done()
        
        # Get observation
        observation = self._get_observation()
        
        # Track episode data
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        if len(self.episode_returns) > 0:
            self.episode_returns.append(self.portfolio.total_pnl)
        else:
            self.episode_returns.append(0)
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio.total_value,
            'total_pnl': self.portfolio.total_pnl,
            'transaction_cost': transaction_cost,
            'position_size': self._get_current_position_size(),
            'step': self.current_step
        }
        
        return observation, reward, done, False, info
    
    def _execute_action(self, position_size: float, order_type: float, price_offset: float) -> float:
        """
        Execute trading action
        
        Educational: This simulates real order execution with
        slippage, commission, and market impact.
        """
        if self.current_step >= len(self.market_data):
            return 0.0
        
        current_price = self.market_data.iloc[self.current_step]['close']
        current_volume = self.market_data.iloc[self.current_step]['volume']
        
        # Calculate desired position
        desired_position = int(position_size * self.max_position_size)
        current_position = self._get_current_position_size()
        position_change = desired_position - current_position
        
        if abs(position_change) < 1:  # No trade needed
            return 0.0
        
        # Calculate execution price with slippage
        if position_change > 0:  # Buying
            execution_price = current_price * (1 + self.slippage_rate)
        else:  # Selling
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Apply limit/stop order logic
        order_type_int = int(np.round(order_type))
        if order_type_int == 1:  # Limit order
            limit_price = current_price * (1 + price_offset)
            if position_change > 0 and execution_price > limit_price:
                return 0.0  # Order not filled
            elif position_change < 0 and execution_price < limit_price:
                return 0.0  # Order not filled
        elif order_type_int == 2:  # Stop order
            stop_price = current_price * (1 + price_offset)
            if position_change > 0 and execution_price < stop_price:
                return 0.0  # Stop not triggered
            elif position_change < 0 and execution_price > stop_price:
                return 0.0  # Stop not triggered
        
        # Calculate transaction cost
        transaction_cost = abs(position_change * execution_price * self.commission_rate)
        
        # Check if we have enough cash
        if position_change > 0:  # Buying
            required_cash = position_change * execution_price + transaction_cost
            if self.portfolio.cash < required_cash:
                return 0.0  # Insufficient funds
        
        # Execute trade
        self._execute_trade(position_change, execution_price, transaction_cost)
        
        return transaction_cost
    
    def _execute_trade(self, quantity: int, price: float, commission: float):
        """
        Execute a trade and update portfolio
        """
        if self.symbol not in self.portfolio.positions:
            # New position
            side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
            position = Position(
                symbol=self.symbol,
                side=side,
                quantity=abs(quantity),
                entry_price=price
            )
            self.portfolio.positions[self.symbol] = position
        else:
            # Existing position
            position = self.portfolio.positions[self.symbol]
            
            if (position.side == PositionSide.LONG and quantity > 0) or \
               (position.side == PositionSide.SHORT and quantity < 0):
                # Adding to position
                total_cost = position.quantity * position.entry_price + abs(quantity) * price
                position.quantity += abs(quantity)
                position.entry_price = total_cost / position.quantity
            else:
                # Reducing or closing position
                closing_quantity = min(abs(quantity), position.quantity)
                realized_pnl = self._calculate_realized_pnl(position, closing_quantity, price)
                position.realized_pnl += realized_pnl
                position.quantity -= closing_quantity
                
                if position.quantity == 0:
                    # Position closed
                    self.portfolio.cash += position.quantity * position.entry_price + realized_pnl
                    del self.portfolio.positions[self.symbol]
                else:
                    # Position reduced
                    self.portfolio.cash += closing_quantity * price
        
        # Update cash
        if quantity > 0:  # Buying
            self.portfolio.cash -= abs(quantity) * price + commission
        else:  # Selling
            self.portfolio.cash += abs(quantity) * price - commission
        
        # Record transaction
        transaction = Transaction(
            timestamp=datetime.now(),
            symbol=self.symbol,
            action="BUY" if quantity > 0 else "SELL",
            quantity=abs(quantity),
            price=price,
            commission=commission
        )
        self.transaction_history.append(transaction)
    
    def _calculate_realized_pnl(self, position: Position, quantity: int, exit_price: float) -> float:
        """Calculate realized P&L for a portion of position"""
        if position.side == PositionSide.LONG:
            return (exit_price - position.entry_price) * quantity
        else:
            return (position.entry_price - exit_price) * quantity
    
    def _update_portfolio(self):
        """Update portfolio values based on current prices"""
        if self.current_step >= len(self.market_data):
            return
        
        current_price = self.market_data.iloc[self.current_step]['close']
        
        for position in self.portfolio.positions.values():
            position.update_pnl(current_price)
        
        self.portfolio.update_total_value()
    
    def _calculate_reward(self, transaction_cost: float) -> float:
        """
        Calculate reward for the current step
        
        Educational: Reward design is crucial in RL.
        This reward function balances profit, risk, and costs.
        """
        if len(self.episode_returns) < 2:
            return 0.0
        
        # Portfolio return
        current_return = self.portfolio.total_pnl
        previous_return = self.episode_returns[-1]
        step_return = current_return - previous_return
        
        # Risk-adjusted return (Sharpe-like)
        if len(self.episode_returns) > 10:
            returns = np.array(self.episode_returns[-10:])
            if np.std(returns) > 0:
                risk_adjusted_return = step_return / np.std(returns)
            else:
                risk_adjusted_return = step_return
        else:
            risk_adjusted_return = step_return
        
        # Transaction cost penalty
        cost_penalty = -transaction_cost / self.initial_cash
        
        # Position size penalty (discourage over-leveraging)
        position_size = abs(self._get_current_position_size())
        position_penalty = -0.0001 * (position_size / self.max_position_size) ** 2
        
        # Drawdown penalty
        if len(self.episode_returns) > 20:
            peak = np.max(self.episode_returns[-20:])
            drawdown = (peak - current_return) / self.initial_cash
            drawdown_penalty = -0.1 * max(0, drawdown)
        else:
            drawdown_penalty = 0
        
        # Total reward
        total_reward = (
            0.4 * risk_adjusted_return +
            0.2 * cost_penalty +
            0.2 * position_penalty +
            0.2 * drawdown_penalty
        )
        
        return total_reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation for the agent
        
        Educational: The observation contains all information
        the agent needs to make decisions.
        """
        if self.market_data is None or self.current_step >= len(self.market_data):
            return self._get_empty_observation()
        
        # Get price history
        start_idx = max(0, self.current_step - self.lookback_window)
        price_data = self.market_data.iloc[start_idx:self.current_step + 1]
        
        # Price features (OHLCV)
        price_features = np.zeros((self.lookback_window, 5))
        if len(price_data) > 0:
            price_array = price_data[['open', 'high', 'low', 'close', 'volume']].values
            price_features[-len(price_array):] = price_array
        
        # Generate technical features
        if len(price_data) >= 20:
            try:
                feature_set = self.feature_pipeline.create_feature_set(
                    symbol=self.symbol,
                    market_data=price_data
                )
                
                # Extract feature values
                feature_values = []
                for feature_name, feature in feature_set.features.items():
                    if isinstance(feature.value, (int, float)):
                        feature_values.append(feature.value)
                
                # Pad or truncate to fixed size
                while len(feature_values) < 100:
                    feature_values.append(0.0)
                feature_values = feature_values[:100]
                technical_features = np.array(feature_values, dtype=np.float32)
                
            except Exception as e:
                logger.warning(f"Error generating features: {e}")
                technical_features = np.zeros(100, dtype=np.float32)
        else:
            technical_features = np.zeros(100, dtype=np.float32)
        
        # Portfolio features
        portfolio_features = np.array([
            self.portfolio.cash / self.initial_cash,  # Normalized cash
            self.portfolio.total_value / self.initial_cash,  # Normalized total value
            self.portfolio.total_pnl / self.initial_cash,  # Normalized P&L
            len(self.portfolio.positions),  # Number of positions
            sum(pos.quantity for pos in self.portfolio.positions.values()),  # Total position size
            self._get_current_position_size() / self.max_position_size,  # Normalized position
            len(self.transaction_history),  # Number of transactions
            sum(t.commission for t in self.transaction_history[-10:]) / self.initial_cash,  # Recent commissions
            0.0,  # Placeholder for Sharpe ratio
            0.0   # Placeholder for max drawdown
        ], dtype=np.float32)
        
        # Position features
        if self.symbol in self.portfolio.positions:
            position = self.portfolio.positions[self.symbol]
            position_features = np.array([
                1.0 if position.side == PositionSide.LONG else -1.0 if position.side == PositionSide.SHORT else 0.0,
                position.quantity / self.max_position_size,
                position.unrealized_pnl / self.initial_cash,
                position.realized_pnl / self.initial_cash,
                (datetime.now() - position.entry_time).total_seconds() / 86400  # Days held
            ], dtype=np.float32)
        else:
            position_features = np.zeros(5, dtype=np.float32)
        
        return {
            'prices': price_features,
            'features': technical_features,
            'portfolio': portfolio_features,
            'position': position_features
        }
    
    def _get_empty_observation(self) -> Dict[str, np.ndarray]:
        """Return empty observation for edge cases"""
        return {
            'prices': np.zeros((self.lookback_window, 5), dtype=np.float32),
            'features': np.zeros(100, dtype=np.float32),
            'portfolio': np.zeros(10, dtype=np.float32),
            'position': np.zeros(5, dtype=np.float32)
        }
    
    def _get_current_position_size(self) -> int:
        """Get current position size"""
        if self.symbol in self.portfolio.positions:
            position = self.portfolio.positions[self.symbol]
            return position.quantity if position.side == PositionSide.LONG else -position.quantity
        return 0
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Episode length limit
        if self.current_step >= self.max_steps:
            return True
        
        # Portfolio ruin
        if self.portfolio.total_value < self.initial_cash * 0.1:  # Lost 90%
            return True
        
        # Maximum profit target (optional)
        if self.portfolio.total_value > self.initial_cash * 2.0:  # Doubled
            return True
        
        return False
    
    def set_market_data(self, market_data: pd.DataFrame):
        """
        Set market data for the environment
        
        Educational: This allows the environment to use
        historical data for training and testing.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in market_data.columns for col in required_columns):
            raise ValueError(f"Market data must contain columns: {required_columns}")
        
        self.market_data = market_data.copy()
        self.max_steps = len(self.market_data) - 1
        
        logger.info(f"Market data set: {len(self.market_data)} periods")
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio performance statistics"""
        if not self.episode_returns:
            return {}
        
        returns = np.array(self.episode_returns)
        
        stats = {
            'total_return': self.portfolio.total_pnl / self.initial_cash,
            'total_return_pct': (self.portfolio.total_value / self.initial_cash - 1) * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'num_trades': len(self.transaction_history),
            'win_rate': self._calculate_win_rate(),
            'avg_trade': np.mean([t.pnl for t in self.transaction_history]) if self.transaction_history else 0,
            'total_commission': sum(t.commission for t in self.transaction_history)
        }
        
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.episode_returns) < 2:
            return 0.0
        
        peak = self.episode_returns[0]
        max_drawdown = 0.0
        
        for ret in self.episode_returns[1:]:
            if ret > peak:
                peak = ret
            else:
                drawdown = (peak - ret) / self.initial_cash
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.transaction_history:
            return 0.0
        
        winning_trades = sum(1 for t in self.transaction_history if t.pnl > 0)
        return winning_trades / len(self.transaction_history)
    
    def render(self, mode='human'):
        """
        Render the environment state
        
        Educational: Visualization helps understand
        what the agent is learning.
        """
        if mode == 'human':
            print(f"\n{'='*50}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio.total_value:,.2f}")
            print(f"Cash: ${self.portfolio.cash:,.2f}")
            print(f"Total P&L: ${self.portfolio.total_pnl:,.2f}")
            print(f"Position: {self._get_current_position_size()} shares")
            
            if self.symbol in self.portfolio.positions:
                pos = self.portfolio.positions[self.symbol]
                print(f"Position P&L: ${pos.unrealized_pnl:,.2f}")
            
            print(f"Number of Trades: {len(self.transaction_history)}")
            print(f"{'='*50}")
    
    def close(self):
        """Clean up environment resources"""
        logger.info("TradingEnvironment closed")

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Basic Environment Setup:
   env = TradingEnvironment(
       symbol="AAPL",
       initial_cash=100000,
       commission_rate=0.001
   )
   
   # Load historical data
   market_data = pd.read_csv('AAPL_data.csv')
   env.set_market_data(market_data)

2. Training Loop:
   obs, info = env.reset()
   done = False
   
   while not done:
       action = agent.predict(obs)  # Your RL agent
       obs, reward, done, truncated, info = env.step(action)
       
       if done:
           stats = env.get_portfolio_stats()
           print(f"Episode return: {stats['total_return_pct']:.2f}%")

3. Custom Reward Function:
   class CustomTradingEnv(TradingEnvironment):
       def _calculate_reward(self, transaction_cost):
           # Implement your custom reward logic
           return super()._calculate_reward(transaction_cost)

4. Multi-Asset Environment:
   env = TradingEnvironment(
       symbol="SPY",
       asset_type=AssetType.ETF,
       max_position_size=500
   )

Key Concepts:
- State Space: What the agent observes (prices, features, portfolio)
- Action Space: What the agent can do (buy, sell, hold)
- Reward Function: How the agent learns (profit, risk, costs)
- Episode: One complete trading period
- Transition: State, Action, Reward, Next State

Educational Notes:
- The environment simulates real market conditions
- Agents learn through trial and error
- Reward design is crucial for learning desired behaviors
- Feature engineering provides meaningful observations
- Risk management is built into the reward function
"""