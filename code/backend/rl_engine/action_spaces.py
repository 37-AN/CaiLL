"""
Action Spaces - Phase 3.1

This module implements different action spaces for various asset types
and trading strategies. Each action space defines how agents can interact
with the market.

Educational Note:
Action space design is crucial in reinforcement learning.
It defines what actions an agent can take and how those actions
are interpreted by the trading environment.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    BUY_LIMIT = 3
    SELL_LIMIT = 4
    BUY_STOP = 5
    SELL_STOP = 6
    CLOSE = 7

class AssetClass(Enum):
    """Asset classes with different trading characteristics"""
    STOCK = "stock"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"

@dataclass
class ActionConstraints:
    """Constraints on trading actions"""
    max_position_size: float = 1.0
    min_position_size: float = 0.01
    max_leverage: float = 1.0
    max_daily_trades: int = 100
    max_order_size: float = 0.1
    min_order_size: float = 0.001
    position_holding_period_min: int = 0
    position_holding_period_max: int = 1000

@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    action_type: ActionType
    quantity: int
    price: float
    cost: float
    error_message: Optional[str] = None
    execution_time: datetime = field(default_factory=datetime.now)

class BaseActionSpace(ABC):
    """
    Base class for action spaces
    
    Educational: All action spaces should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, 
                 asset_class: AssetClass,
                 constraints: ActionConstraints,
                 name: str = "base_action_space"):
        
        self.asset_class = asset_class
        self.constraints = constraints
        self.name = name
        self.action_history = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Return the gym action space"""
        pass
    
    @abstractmethod
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode raw action into trading parameters"""
        pass
    
    @abstractmethod
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate if action is allowed given current state"""
        pass
    
    @abstractmethod
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        """Execute the trading action"""
        pass
    
    def reset_daily_limits(self):
        """Reset daily trading limits"""
        self.daily_trade_count = 0
        self.last_trade_date = None
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action execution statistics"""
        if not self.action_history:
            return {}
        
        action_types = [a.action_type for a in self.action_history]
        type_counts = {action_type.value: action_types.count(action_type) for action_type in ActionType}
        
        return {
            'total_actions': len(self.action_history),
            'action_type_distribution': type_counts,
            'success_rate': sum(1 for a in self.action_history if a.success) / len(self.action_history),
            'average_cost': np.mean([a.cost for a in self.action_history]),
            'daily_trade_count': self.daily_trade_count
        }

class DiscreteActionSpace(BaseActionSpace):
    """
    Discrete action space for simple trading strategies
    
    Educational: Simple action space with discrete choices.
    Good for basic strategies and educational purposes.
    """
    
    def __init__(self, 
                 asset_class: AssetClass,
                 constraints: ActionConstraints,
                 position_sizes: List[float] = None):
        
        super().__init__(asset_class, constraints, "discrete_action_space")
        
        # Define position sizes
        self.position_sizes = position_sizes or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.num_actions = len(self.position_sizes)
        
        # Action mapping: 0=hold, 1=buy_25%, 2=buy_50%, etc.
        self.action_mapping = {
            i: size for i, size in enumerate(self.position_sizes)
        }
    
    def get_action_space(self) -> spaces.Space:
        """Return discrete action space"""
        return spaces.Discrete(self.num_actions)
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode discrete action to trading parameters"""
        action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        
        if action_int not in self.action_mapping:
            return {
                'action_type': ActionType.HOLD,
                'position_size': 0.0,
                'order_type': 'market',
                'price': None
            }
        
        position_size = self.action_mapping[action_int]
        
        if position_size == 0.0:
            action_type = ActionType.HOLD
        elif position_size > 0.0:
            action_type = ActionType.BUY
        else:
            action_type = ActionType.SELL
        
        return {
            'action_type': action_type,
            'position_size': abs(position_size),
            'order_type': 'market',
            'price': None
        }
    
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate discrete action"""
        # Check position size constraints
        position_size = action.get('position_size', 0)
        if position_size > self.constraints.max_position_size:
            return False
        
        # Check daily trade limits
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trade_count >= self.constraints.max_daily_trades:
            return False
        
        return True
    
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        
        if not self.validate_action(action, market_state):
            return ActionResult(
                success=False,
                action_type=action['action_type'],
                quantity=0,
                price=0,
                cost=0,
                error_message="Action validation failed"
            )
        
        # Simulate execution
        current_price = market_state.get('price', 100.0)
        position_size = action['position_size']
        
        # Calculate quantity
        portfolio_value = market_state.get('portfolio_value', 100000)
        order_value = portfolio_value * position_size
        quantity = int(order_value / current_price)
        
        # Calculate cost
        commission = order_value * 0.001  # 0.1% commission
        slippage = order_value * 0.0005  # 0.05% slippage
        total_cost = commission + slippage
        
        # Update daily trade count
        self.daily_trade_count += 1
        
        result = ActionResult(
            success=True,
            action_type=action['action_type'],
            quantity=quantity,
            price=current_price,
            cost=total_cost
        )
        
        self.action_history.append(result)
        return result

class MultiDiscreteActionSpace(BaseActionSpace):
    """
    Multi-discrete action space for more complex strategies
    
    Educational: Allows multiple choices in a single action.
    Action = [action_type, position_size, duration]
    """
    
    def __init__(self, 
                 asset_class: AssetClass,
                 constraints: ActionConstraints):
        
        super().__init__(asset_class, constraints, "multi_discrete_action_space")
        
        # Define action dimensions
        self.action_types = [ActionType.HOLD, ActionType.BUY, ActionType.SELL, 
                           ActionType.CLOSE, ActionType.BUY_LIMIT, ActionType.SELL_LIMIT]
        self.position_sizes = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.durations = [1, 5, 10, 20, 50]  # Holding periods
        
        self.action_space = spaces.MultiDiscrete([
            len(self.action_types),      # Action type
            len(self.position_sizes),    # Position size
            len(self.durations)          # Duration
        ])
    
    def get_action_space(self) -> spaces.Space:
        """Return multi-discrete action space"""
        return self.action_space
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode multi-discrete action"""
        action_type_idx, position_size_idx, duration_idx = action
        
        action_type = self.action_types[action_type_idx]
        position_size = self.position_sizes[position_size_idx]
        duration = self.durations[duration_idx]
        
        return {
            'action_type': action_type,
            'position_size': position_size,
            'duration': duration,
            'order_type': 'market' if action_type in [ActionType.BUY, ActionType.SELL] else 'limit',
            'price': None
        }
    
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate multi-discrete action"""
        # Check position size
        position_size = action.get('position_size', 0)
        if position_size > self.constraints.max_position_size:
            return False
        
        # Check daily trade limits
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trade_count >= self.constraints.max_daily_trades:
            return False
        
        # Check holding period constraints
        duration = action.get('duration', 1)
        if duration < self.constraints.position_holding_period_min:
            return False
        if duration > self.constraints.position_holding_period_max:
            return False
        
        return True
    
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        
        if not self.validate_action(action, market_state):
            return ActionResult(
                success=False,
                action_type=action['action_type'],
                quantity=0,
                price=0,
                cost=0,
                error_message="Action validation failed"
            )
        
        # Simulate execution
        current_price = market_state.get('price', 100.0)
        position_size = action['position_size']
        
        # Calculate quantity
        portfolio_value = market_state.get('portfolio_value', 100000)
        order_value = portfolio_value * position_size
        quantity = int(order_value / current_price)
        
        # Calculate cost (higher for limit orders)
        commission = order_value * 0.001
        if action['order_type'] == 'limit':
            slippage = order_value * 0.0001  # Lower slippage for limit orders
        else:
            slippage = order_value * 0.0005
        
        total_cost = commission + slippage
        
        # Update daily trade count
        self.daily_trade_count += 1
        
        result = ActionResult(
            success=True,
            action_type=action['action_type'],
            quantity=quantity,
            price=current_price,
            cost=total_cost
        )
        
        self.action_history.append(result)
        return result

class BoxActionSpace(BaseActionSpace):
    """
    Continuous box action space for advanced strategies
    
    Educational: Allows precise control over position sizing and timing.
    Action = [position_size, order_type, price_offset, stop_loss, take_profit]
    """
    
    def __init__(self, 
                 asset_class: AssetClass,
                 constraints: ActionConstraints):
        
        super().__init__(asset_class, constraints, "box_action_space")
        
        # Define continuous action ranges
        # position_size: -1 to 1 (negative = short, positive = long)
        # order_type: 0 to 2 (0=market, 1=limit, 2=stop)
        # price_offset: -0.02 to 0.02 (2% offset for limit/stop orders)
        # stop_loss: 0 to 0.1 (10% stop loss)
        # take_profit: 0 to 0.2 (20% take profit)
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -0.02, 0.0, 0.0]),
            high=np.array([1.0, 2.0, 0.02, 0.1, 0.2]),
            dtype=np.float32
        )
    
    def get_action_space(self) -> spaces.Space:
        """Return box action space"""
        return self.action_space
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode continuous action"""
        position_size, order_type, price_offset, stop_loss, take_profit = action
        
        # Determine action type
        if abs(position_size) < 0.01:  # Very small position = hold
            action_type = ActionType.HOLD
        elif position_size > 0:
            action_type = ActionType.BUY
        else:
            action_type = ActionType.SELL
        
        # Determine order type
        order_type_int = int(np.round(order_type))
        if order_type_int == 0:
            order_type_str = 'market'
        elif order_type_int == 1:
            order_type_str = 'limit'
        else:
            order_type_str = 'stop'
        
        return {
            'action_type': action_type,
            'position_size': abs(position_size),
            'order_type': order_type_str,
            'price_offset': price_offset,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate continuous action"""
        # Check position size
        position_size = action.get('position_size', 0)
        if position_size > self.constraints.max_position_size:
            return False
        if position_size < self.constraints.min_position_size and position_size > 0:
            return False
        
        # Check leverage
        leverage = market_state.get('leverage', 1.0)
        if leverage > self.constraints.max_leverage:
            return False
        
        # Check daily trade limits
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trade_count >= self.constraints.max_daily_trades:
            return False
        
        return True
    
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        
        if not self.validate_action(action, market_state):
            return ActionResult(
                success=False,
                action_type=action['action_type'],
                quantity=0,
                price=0,
                cost=0,
                error_message="Action validation failed"
            )
        
        # Simulate execution
        current_price = market_state.get('price', 100.0)
        position_size = action['position_size']
        price_offset = action.get('price_offset', 0.0)
        
        # Calculate execution price
        if action['order_type'] == 'market':
            execution_price = current_price * (1 + np.random.normal(0, 0.0005))  # Market impact
        elif action['order_type'] == 'limit':
            execution_price = current_price * (1 + price_offset)
        else:  # stop order
            execution_price = current_price * (1 + price_offset)
        
        # Calculate quantity
        portfolio_value = market_state.get('portfolio_value', 100000)
        order_value = portfolio_value * position_size
        quantity = int(order_value / execution_price)
        
        # Calculate cost
        commission = order_value * 0.001
        slippage = abs(execution_price - current_price) * quantity
        total_cost = commission + slippage
        
        # Update daily trade count
        self.daily_trade_count += 1
        
        result = ActionResult(
            success=True,
            action_type=action['action_type'],
            quantity=quantity,
            price=execution_price,
            cost=total_cost
        )
        
        self.action_history.append(result)
        return result

class OptionActionSpace(BaseActionSpace):
    """
    Specialized action space for options trading
    
    Educational: Options require different action spaces due to
    Greeks, expiration, and multiple strike prices.
    """
    
    def __init__(self, 
                 constraints: ActionConstraints,
                 strike_prices: List[float] = None,
                 expirations: List[timedelta] = None):
        
        super().__init__(AssetClass.OPTION, constraints, "option_action_space")
        
        # Option-specific parameters
        self.strike_prices = strike_prices or [90, 95, 100, 105, 110]
        self.expirations = expirations or [
            timedelta(days=7),   # Weekly
            timedelta(days=30),  # Monthly
            timedelta(days=90)   # Quarterly
        ]
        
        # Option strategies
        self.strategies = ['call', 'put', 'covered_call', 'protective_put', 
                          'straddle', 'strangle', 'spread']
        
        # Action space: [strategy, strike, expiration, quantity]
        self.action_space = spaces.MultiDiscrete([
            len(self.strategies),
            len(self.strike_prices),
            len(self.expirations),
            10  # Quantity (1-10 contracts)
        ])
    
    def get_action_space(self) -> spaces.Space:
        """Return options action space"""
        return self.action_space
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode options action"""
        strategy_idx, strike_idx, expiration_idx, quantity = action
        
        return {
            'strategy': self.strategies[strategy_idx],
            'strike_price': self.strike_prices[strike_idx],
            'expiration': self.expirations[expiration_idx],
            'quantity': quantity + 1,  # Convert from 0-based to 1-based
            'action_type': ActionType.BUY  # Simplified for options
        }
    
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate options action"""
        # Check position size
        quantity = action.get('quantity', 0)
        if quantity * 100 > self.constraints.max_position_size * market_state.get('portfolio_value', 100000):
            return False
        
        # Check if option is available
        current_price = market_state.get('price', 100)
        strike_price = action.get('strike_price', 100)
        
        # Strike should be reasonably close to current price
        if abs(strike_price - current_price) / current_price > 0.2:
            return False
        
        return True
    
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        
        if not self.validate_action(action, market_state):
            return ActionResult(
                success=False,
                action_type=action['action_type'],
                quantity=0,
                price=0,
                cost=0,
                error_message="Options action validation failed"
            )
        
        # Simulate options execution
        current_price = market_state.get('price', 100.0)
        strike_price = action['strike_price']
        quantity = action['quantity'] * 100  # Options contracts
        
        # Simplified option pricing (would use Black-Scholes in reality)
        intrinsic_value = max(0, current_price - strike_price)
        time_value = 2.0  # Simplified time value
        option_price = intrinsic_value + time_value
        
        # Calculate cost
        total_cost = quantity * option_price * 1.5  # Higher commission for options
        
        result = ActionResult(
            success=True,
            action_type=action['action_type'],
            quantity=quantity,
            price=option_price,
            cost=total_cost
        )
        
        self.action_history.append(result)
        return result

class CryptoActionSpace(BaseActionSpace):
    """
    Specialized action space for cryptocurrency trading
    
    Educational: Crypto markets have different characteristics
    like 24/7 trading, higher volatility, and different fee structures.
    """
    
    def __init__(self, 
                 constraints: ActionConstraints,
                 exchanges: List[str] = None):
        
        super().__init__(AssetClass.CRYPTO, constraints, "crypto_action_space")
        
        # Crypto-specific parameters
        self.exchanges = exchanges or ['binance', 'coinbase', 'kraken']
        self.order_types = ['market', 'limit', 'stop_limit', 'oco']  # One-cancels-other
        
        # Higher leverage for crypto
        self.constraints.max_leverage = 3.0
        
        # Action space: [position_size, order_type, exchange, leverage]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 1.0]),
            high=np.array([1.0, 3.0, 2.0, 3.0]),
            dtype=np.float32
        )
    
    def get_action_space(self) -> spaces.Space:
        """Return crypto action space"""
        return self.action_space
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode crypto action"""
        position_size, order_type, exchange, leverage = action
        
        order_type_int = int(np.round(order_type))
        exchange_int = int(np.round(exchange))
        
        return {
            'action_type': ActionType.BUY if position_size > 0 else ActionType.SELL,
            'position_size': abs(position_size),
            'order_type': self.order_types[order_type_int],
            'exchange': self.exchanges[exchange_int],
            'leverage': min(leverage, self.constraints.max_leverage)
        }
    
    def validate_action(self, action: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Validate crypto action"""
        # Check leverage
        leverage = action.get('leverage', 1.0)
        if leverage > self.constraints.max_leverage:
            return False
        
        # Check position size
        position_size = action.get('position_size', 0)
        if position_size > self.constraints.max_position_size:
            return False
        
        return True
    
    def execute_action(self, 
                      action: Dict[str, Any], 
                      market_state: Dict[str, Any]) -> ActionResult:
        
        if not self.validate_action(action, market_state):
            return ActionResult(
                success=False,
                action_type=action['action_type'],
                quantity=0,
                price=0,
                cost=0,
                error_message="Crypto action validation failed"
            )
        
        # Simulate crypto execution
        current_price = market_state.get('price', 100.0)
        position_size = action['position_size']
        leverage = action.get('leverage', 1.0)
        
        # Calculate quantity (with leverage)
        portfolio_value = market_state.get('portfolio_value', 100000)
        order_value = portfolio_value * position_size * leverage
        quantity = order_value / current_price
        
        # Crypto fees (typically lower than stocks)
        exchange = action.get('exchange', 'binance')
        fee_rates = {'binance': 0.001, 'coinbase': 0.005, 'kraken': 0.002}
        fee_rate = fee_rates.get(exchange, 0.001)
        
        total_cost = order_value * fee_rate
        
        result = ActionResult(
            success=True,
            action_type=action['action_type'],
            quantity=quantity,
            price=current_price,
            cost=total_cost
        )
        
        self.action_history.append(result)
        return result

class ActionSpaceFactory:
    """
    Factory class for creating action spaces
    
    Educational: This factory pattern makes it easy to create
    appropriate action spaces for different trading scenarios.
    """
    
    @staticmethod
    def create_action_space(asset_class: AssetClass,
                          action_space_type: str = "box",
                          constraints: Optional[ActionConstraints] = None,
                          **kwargs) -> BaseActionSpace:
        """
        Create an action space for the specified asset class
        
        Args:
            asset_class: The asset class to trade
            action_space_type: Type of action space ("discrete", "multidiscrete", "box")
            constraints: Trading constraints
            **kwargs: Additional parameters for specific action spaces
            
        Returns:
            Instance of the requested action space
        """
        
        if constraints is None:
            constraints = ActionConstraints()
        
        if asset_class == AssetClass.STOCK:
            if action_space_type == "discrete":
                return DiscreteActionSpace(asset_class, constraints)
            elif action_space_type == "multidiscrete":
                return MultiDiscreteActionSpace(asset_class, constraints)
            elif action_space_type == "box":
                return BoxActionSpace(asset_class, constraints)
            else:
                raise ValueError(f"Unknown action space type: {action_space_type}")
        
        elif asset_class == AssetClass.OPTION:
            return OptionActionSpace(constraints, **kwargs)
        
        elif asset_class == AssetClass.CRYPTO:
            return CryptoActionSpace(constraints, **kwargs)
        
        elif asset_class == AssetClass.FOREX:
            # Forex uses similar to box but with different constraints
            constraints.max_leverage = 100.0  # High leverage in forex
            return BoxActionSpace(asset_class, constraints)
        
        elif asset_class == AssetClass.COMMODITY:
            return BoxActionSpace(asset_class, constraints)
        
        elif asset_class == AssetClass.BOND:
            # Bonds typically use discrete actions
            return DiscreteActionSpace(asset_class, constraints)
        
        else:
            raise ValueError(f"Unknown asset class: {asset_class}")

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Simple Stock Trading:
   constraints = ActionConstraints(max_position_size=0.5)
   action_space = ActionSpaceFactory.create_action_space(
       AssetClass.STOCK, "discrete", constraints
   )

2. Advanced Crypto Trading:
   constraints = ActionConstraints(max_leverage=3.0, max_daily_trades=200)
   action_space = ActionSpaceFactory.create_action_space(
       AssetClass.CRYPTO, "box", constraints
   )

3. Options Trading:
   action_space = ActionSpaceFactory.create_action_space(
       AssetClass.OPTION,
       strike_prices=[90, 95, 100, 105, 110],
       expirations=[timedelta(days=7), timedelta(days=30)]
   )

4. Forex Trading:
   constraints = ActionConstraints(max_leverage=50.0)
   action_space = ActionSpaceFactory.create_action_space(
       AssetClass.FOREX, "box", constraints
   )

5. Custom Constraints:
   constraints = ActionConstraints(
       max_position_size=0.3,
       max_daily_trades=50,
       position_holding_period_min=5
   )
   action_space = ActionSpaceFactory.create_action_space(
       AssetClass.STOCK, "multidiscrete", constraints
   )

Key Design Principles:
- Action space should match the asset class characteristics
- Constraints should reflect realistic trading limits
- Different strategies need different action spaces
- Consider computational complexity vs. expressiveness
- Include risk management in action constraints

Educational Notes:
- Start simple and gradually increase complexity
- Test different action spaces with the same strategy
- Consider the trading frequency and costs
- Ensure actions are interpretable and executable
- Monitor action distribution during training
"""