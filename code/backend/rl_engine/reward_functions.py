"""
Reward Functions - Phase 3.1

This module implements various reward functions for reinforcement learning
in trading. Different reward functions encourage different trading behaviors.

Educational Note:
Reward function design is one of the most important aspects of RL.
The reward function shapes what the agent learns and should be carefully
designed to encourage desired trading behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RewardType(Enum):
    """Types of reward functions"""
    PROFIT = "profit"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    DRAWDOWN = "drawdown"
    INFORMATION_RATIO = "information_ratio"
    CALMAR = "calmar"
    CUSTOM = "custom"

@dataclass
class RewardConfig:
    """Configuration for reward functions"""
    # Basic parameters
    profit_weight: float = 0.4
    risk_weight: float = 0.3
    transaction_cost_weight: float = 0.2
    position_weight: float = 0.1
    
    # Risk parameters
    risk_free_rate: float = 0.02
    target_return: float = 0.10
    max_drawdown_limit: float = 0.20
    
    # Transaction costs
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    market_impact_factor: float = 0.0001
    
    # Position management
    max_position_size: float = 1.0
    position_holding_period_target: int = 10
    turnover_penalty: float = 0.001
    
    # Advanced parameters
    benchmark_return: Optional[float] = None
    volatility_target: float = 0.15
    correlation_penalty: float = 0.1

class BaseRewardFunction(ABC):
    """
    Base class for reward functions
    
    Educational: All reward functions should inherit from this class
    and implement the calculate_reward method.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.reward_history = []
        self.episode_stats = {}
        
    @abstractmethod
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        """
        Calculate reward for the current step
        
        Args:
            current_state: Current market and portfolio state
            previous_state: Previous state for comparison
            action: Action taken by the agent
            transaction_cost: Cost of executing the action
            
        Returns:
            Reward value (can be positive or negative)
        """
        pass
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        self.reward_history = []
        self.episode_stats = {}
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics"""
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        
        stats = {
            'total_reward': np.sum(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'positive_rewards': np.sum(rewards > 0),
            'negative_rewards': np.sum(rewards < 0),
            'reward_volatility': np.std(rewards) / np.abs(np.mean(rewards)) if np.mean(rewards) != 0 else 0
        }
        
        return stats

class ProfitRewardFunction(BaseRewardFunction):
    """
    Simple profit-based reward function
    
    Educational: This is the most basic reward function that only
    considers profit. It's simple but can lead to risky behavior.
    """
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        # Calculate profit
        current_pnl = current_state.get('total_pnl', 0)
        previous_pnl = previous_state.get('total_pnl', 0)
        profit = current_pnl - previous_pnl
        
        # Normalize by initial capital
        initial_capital = current_state.get('initial_capital', 100000)
        normalized_profit = profit / initial_capital
        
        # Transaction cost penalty
        cost_penalty = -transaction_cost / initial_capital
        
        # Position size penalty (discourage over-leveraging)
        position_size = abs(current_state.get('position_size', 0))
        max_position = current_state.get('max_position_size', 1)
        position_penalty = -self.config.position_weight * (position_size / max_position) ** 2
        
        # Total reward
        total_reward = (
            self.config.profit_weight * normalized_profit +
            self.config.transaction_cost_weight * cost_penalty +
            self.config.position_weight * position_penalty
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward

class SharpeRewardFunction(BaseRewardFunction):
    """
    Sharpe ratio-based reward function
    
    Educational: This reward function considers risk-adjusted returns.
    It encourages consistent returns rather than high volatility.
    """
    
    def __init__(self, config: RewardConfig, lookback_period: int = 20):
        super().__init__(config)
        self.lookback_period = lookback_period
        self.return_history = []
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        # Calculate return
        current_value = current_state.get('portfolio_value', 0)
        previous_value = previous_state.get('portfolio_value', 0)
        
        if previous_value == 0:
            return 0
        
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Add to return history
        self.return_history.append(portfolio_return)
        
        # Keep only recent history
        if len(self.return_history) > self.lookback_period:
            self.return_history = self.return_history[-self.lookback_period:]
        
        # Calculate Sharpe ratio
        if len(self.return_history) >= 2:
            returns = np.array(self.return_history)
            excess_returns = returns - self.config.risk_free_rate / 252  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(excess_returns)
                sharpe_annualized = sharpe * np.sqrt(252)
            else:
                sharpe_annualized = 0
        else:
            sharpe_annualized = 0
        
        # Transaction cost penalty
        initial_capital = current_state.get('initial_capital', 100000)
        cost_penalty = -transaction_cost / initial_capital
        
        # Total reward
        total_reward = (
            self.config.profit_weight * sharpe_annualized +
            self.config.transaction_cost_weight * cost_penalty
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward

class SortinoRewardFunction(BaseRewardFunction):
    """
    Sortino ratio-based reward function
    
    Educational: Similar to Sharpe but only penalizes downside volatility.
    This is often preferred for trading as it doesn't punish upside volatility.
    """
    
    def __init__(self, config: RewardConfig, lookback_period: int = 20):
        super().__init__(config)
        self.lookback_period = lookback_period
        self.return_history = []
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        # Calculate return
        current_value = current_state.get('portfolio_value', 0)
        previous_value = previous_state.get('portfolio_value', 0)
        
        if previous_value == 0:
            return 0
        
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Add to return history
        self.return_history.append(portfolio_return)
        
        # Keep only recent history
        if len(self.return_history) > self.lookback_period:
            self.return_history = self.return_history[-self.lookback_period:]
        
        # Calculate Sortino ratio
        if len(self.return_history) >= 2:
            returns = np.array(self.return_history)
            excess_returns = returns - self.config.risk_free_rate / 252
            
            # Only consider downside volatility
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                sortino = np.mean(excess_returns) / downside_deviation
                sortino_annualized = sortino * np.sqrt(252)
            else:
                sortino_annualized = 0
        else:
            sortino_annualized = 0
        
        # Transaction cost penalty
        initial_capital = current_state.get('initial_capital', 100000)
        cost_penalty = -transaction_cost / initial_capital
        
        # Total reward
        total_reward = (
            self.config.profit_weight * sortino_annualized +
            self.config.transaction_cost_weight * cost_penalty
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward

class DrawdownRewardFunction(BaseRewardFunction):
    """
    Drawdown-based reward function
    
    Educational: This reward function heavily penalizes drawdowns,
    making it suitable for risk-averse trading strategies.
    """
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.peak_value = 0
        self.max_drawdown = 0
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        current_value = current_state.get('portfolio_value', 0)
        initial_capital = current_state.get('initial_capital', 100000)
        
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        else:
            current_drawdown = 0
        
        # Calculate return
        previous_value = previous_state.get('portfolio_value', current_value)
        if previous_value > 0:
            portfolio_return = (current_value - previous_value) / previous_value
        else:
            portfolio_return = 0
        
        # Drawdown penalty (very strong)
        drawdown_penalty = -self.config.risk_weight * current_drawdown * 10
        
        # Max drawdown penalty
        if current_drawdown > self.config.max_drawdown_limit:
            max_dd_penalty = -1.0  # Strong penalty for exceeding limit
        else:
            max_dd_penalty = 0
        
        # Transaction cost penalty
        cost_penalty = -transaction_cost / initial_capital
        
        # Profit reward
        profit_reward = self.config.profit_weight * portfolio_return
        
        # Total reward
        total_reward = (
            profit_reward +
            drawdown_penalty +
            max_dd_penalty +
            cost_penalty
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        super().reset_episode()
        self.peak_value = 0
        self.max_drawdown = 0

class InformationRatioRewardFunction(BaseRewardFunction):
    """
    Information ratio-based reward function
    
    Educational: This reward function measures performance relative
    to a benchmark. It's useful for relative value strategies.
    """
    
    def __init__(self, config: RewardConfig, benchmark_returns: pd.Series):
        super().__init__(config)
        self.benchmark_returns = benchmark_returns
        self.active_returns = []
        self.current_step = 0
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        # Calculate portfolio return
        current_value = current_state.get('portfolio_value', 0)
        previous_value = previous_state.get('portfolio_value', 0)
        
        if previous_value == 0:
            return 0
        
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Get benchmark return for this step
        if self.current_step < len(self.benchmark_returns):
            benchmark_return = self.benchmark_returns.iloc[self.current_step]
        else:
            benchmark_return = 0
        
        # Calculate active return (excess over benchmark)
        active_return = portfolio_return - benchmark_return
        self.active_returns.append(active_return)
        
        # Calculate information ratio
        if len(self.active_returns) >= 2:
            active_returns_array = np.array(self.active_returns)
            
            if np.std(active_returns_array) > 0:
                information_ratio = np.mean(active_returns_array) / np.std(active_returns_array)
                ir_annualized = information_ratio * np.sqrt(252)
            else:
                ir_annualized = 0
        else:
            ir_annualized = 0
        
        # Transaction cost penalty
        initial_capital = current_state.get('initial_capital', 100000)
        cost_penalty = -transaction_cost / initial_capital
        
        # Total reward
        total_reward = (
            self.config.profit_weight * ir_annualized +
            self.config.transaction_cost_weight * cost_penalty
        )
        
        self.reward_history.append(total_reward)
        self.current_step += 1
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        super().reset_episode()
        self.active_returns = []
        self.current_step = 0

class CalmarRewardFunction(BaseRewardFunction):
    """
    Calmar ratio-based reward function
    
    Educational: Calmar ratio = Annual Return / Maximum Drawdown.
    This reward function balances returns with drawdown protection.
    """
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.peak_value = 0
        self.max_drawdown = 0
        self.start_value = 0
        self.start_time = None
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        current_value = current_state.get('portfolio_value', 0)
        current_time = current_state.get('timestamp', datetime.now())
        
        # Initialize start values
        if self.start_value == 0:
            self.start_value = current_value
            self.start_time = current_time
        
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        else:
            current_drawdown = 0
        
        # Calculate annualized return
        if self.start_time and current_time > self.start_time:
            days_elapsed = (current_time - self.start_time).days
            if days_elapsed > 0:
                total_return = (current_value - self.start_value) / self.start_value
                annualized_return = total_return * (365 / days_elapsed)
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # Calculate Calmar ratio
        if self.max_drawdown > 0:
            calmar_ratio = annualized_return / self.max_drawdown
        else:
            calmar_ratio = annualized_return  # No drawdown, just return
        
        # Drawdown penalty
        drawdown_penalty = -self.config.risk_weight * current_drawdown * 5
        
        # Transaction cost penalty
        initial_capital = current_state.get('initial_capital', 100000)
        cost_penalty = -transaction_cost / initial_capital
        
        # Total reward
        total_reward = (
            self.config.profit_weight * calmar_ratio +
            drawdown_penalty +
            cost_penalty
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        super().reset_episode()
        self.peak_value = 0
        self.max_drawdown = 0
        self.start_value = 0
        self.start_time = None

class CustomRewardFunction(BaseRewardFunction):
    """
    Custom reward function that combines multiple objectives
    
    Educational: This shows how to create sophisticated reward functions
    that balance multiple trading objectives.
    """
    
    def __init__(self, config: RewardConfig, custom_weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.custom_weights = custom_weights or {}
        self.return_history = []
        self.position_history = []
        self.transaction_costs = []
        
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        previous_state: Dict[str, Any],
                        action: np.ndarray,
                        transaction_cost: float) -> float:
        
        # Extract state variables
        current_value = current_state.get('portfolio_value', 0)
        previous_value = previous_state.get('portfolio_value', 0)
        position_size = current_state.get('position_size', 0)
        volatility = current_state.get('volatility', 0)
        
        # Calculate components
        if previous_value > 0:
            portfolio_return = (current_value - previous_value) / previous_value
        else:
            portfolio_return = 0
        
        # Risk-adjusted return component
        self.return_history.append(portfolio_return)
        if len(self.return_history) >= 10:
            recent_returns = np.array(self.return_history[-10:])
            if np.std(recent_returns) > 0:
                risk_adjusted_return = np.mean(recent_returns) / np.std(recent_returns)
            else:
                risk_adjusted_return = np.mean(recent_returns)
        else:
            risk_adjusted_return = portfolio_return
        
        # Position stability component
        self.position_history.append(position_size)
        if len(self.position_history) >= 5:
            recent_positions = np.array(self.position_history[-5:])
            position_stability = -np.std(recent_positions)  # Penalize position changes
        else:
            position_stability = 0
        
        # Transaction cost component
        self.transaction_costs.append(transaction_cost)
        if len(self.transaction_costs) >= 10:
            recent_costs = np.array(self.transaction_costs[-10:])
            cost_efficiency = -np.mean(recent_costs)
        else:
            cost_efficiency = -transaction_cost
        
        # Volatility targeting component
        if volatility > 0:
            volatility_penalty = -abs(volatility - self.config.volatility_target) * self.config.correlation_penalty
        else:
            volatility_penalty = 0
        
        # Combine components with custom weights
        total_reward = (
            self.custom_weights.get('return', 0.3) * risk_adjusted_return +
            self.custom_weights.get('stability', 0.2) * position_stability +
            self.custom_weights.get('cost', 0.2) * cost_efficiency +
            self.custom_weights.get('volatility', 0.1) * volatility_penalty +
            self.custom_weights.get('profit', 0.2) * portfolio_return
        )
        
        self.reward_history.append(total_reward)
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific variables"""
        super().reset_episode()
        self.return_history = []
        self.position_history = []
        self.transaction_costs = []

class RewardFunctionFactory:
    """
    Factory class for creating reward functions
    
    Educational: This factory pattern makes it easy to create
    different reward functions without changing the main code.
    """
    
    @staticmethod
    def create_reward_function(reward_type: RewardType, 
                             config: RewardConfig,
                             **kwargs) -> BaseRewardFunction:
        """
        Create a reward function of the specified type
        
        Args:
            reward_type: Type of reward function to create
            config: Configuration for the reward function
            **kwargs: Additional parameters for specific reward functions
            
        Returns:
            Instance of the requested reward function
        """
        
        if reward_type == RewardType.PROFIT:
            return ProfitRewardFunction(config)
        
        elif reward_type == RewardType.SHARPE:
            return SharpeRewardFunction(config, lookback_period=kwargs.get('lookback_period', 20))
        
        elif reward_type == RewardType.SORTINO:
            return SortinoRewardFunction(config, lookback_period=kwargs.get('lookback_period', 20))
        
        elif reward_type == RewardType.DRAWDOWN:
            return DrawdownRewardFunction(config)
        
        elif reward_type == RewardType.INFORMATION_RATIO:
            benchmark_returns = kwargs.get('benchmark_returns')
            if benchmark_returns is None:
                raise ValueError("benchmark_returns required for Information Ratio reward function")
            return InformationRatioRewardFunction(config, benchmark_returns)
        
        elif reward_type == RewardType.CALMAR:
            return CalmarRewardFunction(config)
        
        elif reward_type == RewardType.CUSTOM:
            custom_weights = kwargs.get('custom_weights')
            return CustomRewardFunction(config, custom_weights)
        
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Basic Profit Reward:
   config = RewardConfig()
   reward_fn = RewardFunctionFactory.create_reward_function(RewardType.PROFIT, config)

2. Risk-Adjusted Reward (Sharpe):
   config = RewardConfig(profit_weight=0.6, risk_weight=0.3)
   reward_fn = RewardFunctionFactory.create_reward_function(RewardType.SHARPE, config)

3. Drawdown-Focused Reward:
   config = RewardConfig(
       profit_weight=0.2,
       risk_weight=0.7,
       max_drawdown_limit=0.15
   )
   reward_fn = RewardFunctionFactory.create_reward_function(RewardType.DRAWDOWN, config)

4. Custom Multi-Objective Reward:
   custom_weights = {
       'return': 0.3,
       'stability': 0.2,
       'cost': 0.2,
       'volatility': 0.1,
       'profit': 0.2
   }
   reward_fn = RewardFunctionFactory.create_reward_function(
       RewardType.CUSTOM, config, custom_weights=custom_weights
   )

5. Benchmark-Relative Reward:
   benchmark_returns = pd.read_csv('benchmark_returns.csv')
   reward_fn = RewardFunctionFactory.create_reward_function(
       RewardType.INFORMATION_RATIO, config, benchmark_returns=benchmark_returns
   )

Key Design Principles:
- Balance profit and risk
- Penalize excessive transaction costs
- Encourage desired trading behaviors
- Consider the investment horizon
- Align with trading objectives

Educational Notes:
- Different reward functions produce different behaviors
- Test multiple reward functions to find the best fit
- Consider the risk tolerance of the strategy
- Monitor agent behavior during training
- Adjust weights based on observed performance
"""