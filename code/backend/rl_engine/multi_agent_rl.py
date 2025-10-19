"""
Multi-Agent Reinforcement Learning System - Phase 3.2

This module implements a multi-agent RL system where multiple specialized
agents work together to trade different assets and strategies.

Educational Note:
Multi-agent systems can capture complex market dynamics better than
single agents. Each agent specializes in a particular strategy or market
condition, and they collaborate through a meta-agent.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our RL components
from .trading_env import TradingEnvironment, AssetType
from .reward_functions import RewardFunctionFactory, RewardType, RewardConfig
from .action_spaces import ActionSpaceFactory, AssetClass, ActionConstraints

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of specialized trading agents"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_TRADING = "volatility_trading"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"

class MarketRegime(Enum):
    """Market regimes for agent selection"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_type: AgentType
    asset_class: AssetClass
    reward_type: RewardType
    action_space_type: str = "box"
    learning_rate: float = 0.001
    memory_size: int = 10000
    batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    training_freq: int = 4
    save_freq: int = 1000
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    dropout_rate: float = 0.2
    
    # Training parameters
    gamma: float = 0.99
    tau: float = 0.005  # For soft updates
    gradient_clip: float = 1.0

@dataclass
class AgentPerformance:
    """Performance metrics for an agent"""
    agent_id: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    avg_trade: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, returns: List[float], trades: List[float]):
        """Update performance metrics"""
        if not returns:
            return
        
        returns_array = np.array(returns)
        trades_array = np.array(trades)
        
        self.total_return = np.sum(returns_array)
        self.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        if self.volatility > 0:
            self.sharpe_ratio = self.total_return / self.volatility
        
        self.max_drawdown = self._calculate_max_drawdown(returns_array)
        self.win_rate = np.sum(trades_array > 0) / len(trades_array) if len(trades_array) > 0 else 0
        self.avg_trade = np.mean(trades_array) if len(trades_array) > 0 else 0
        self.num_trades = len(trades_array)
        self.last_updated = datetime.now()
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for trading agents
    
    Educational: DQN is a fundamental RL algorithm that learns
    the value of taking actions in states.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

class ReplayBuffer:
    """
    Experience replay buffer for DQN
    
    Educational: Experience replay breaks correlations and improves
    sample efficiency by reusing past experiences.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class TradingAgent:
    """
    Individual trading agent with DQN
    
    Educational: Each agent specializes in a particular trading style
    and learns to make optimal decisions for its domain.
    """
    
    def __init__(self, 
                 agent_id: str,
                 config: AgentConfig,
                 environment: TradingEnvironment):
        
        self.agent_id = agent_id
        self.config = config
        self.environment = environment
        
        # Initialize networks
        state_dim = self._get_state_dimension()
        action_dim = self._get_action_dimension()
        
        self.q_network = DQNNetwork(state_dim, action_dim, config)
        self.target_network = DQNNetwork(state_dim, action_dim, config)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.memory_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        
        # Performance tracking
        self.performance = AgentPerformance(agent_id)
        self.returns_history = []
        self.trades_history = []
        
        # Action space
        self.action_space = ActionSpaceFactory.create_action_space(
            config.asset_class, 
            config.action_space_type
        )
        
        # Reward function
        reward_config = RewardConfig()
        self.reward_function = RewardFunctionFactory.create_reward_function(
            config.reward_type, reward_config
        )
        
        logger.info(f"TradingAgent {agent_id} initialized ({config.agent_type.value})")
    
    def _get_state_dimension(self) -> int:
        """Calculate state dimension from environment"""
        # This would be dynamically calculated based on the environment
        # For now, return a reasonable default
        return 200  # Approximate state size
    
    def _get_action_dimension(self) -> int:
        """Calculate action dimension from action space"""
        if hasattr(self.action_space, 'action_space'):
            if hasattr(self.action_space.action_space, 'n'):
                return self.action_space.action_space.n
            elif hasattr(self.action_space.action_space, 'nvec'):
                return np.prod(self.action_space.action_space.nvec)
            elif hasattr(self.action_space.action_space, 'shape'):
                return np.prod(self.action_space.action_space.shape)
        
        # Default for box spaces
        return 10  # Reasonable default
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy
        
        Educational: Epsilon-greedy balances exploration and exploitation
        by randomly selecting actions with probability epsilon.
        """
        if training and np.random.random() < self.epsilon:
            # Random action for exploration
            if hasattr(self.action_space.action_space, 'sample'):
                return self.action_space.action_space.sample()
            else:
                return np.random.uniform(-1, 1, size=self._get_action_dimension())
        
        # Greedy action for exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            if hasattr(self.action_space.action_space, 'n'):
                action = q_values.argmax().item()
                return np.array([action])
            else:
                # For continuous actions, use the Q-values to guide the action
                return torch.tanh(q_values).squeeze().numpy()
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Educational: This is where the agent learns from experience
        by updating its Q-values using the Bellman equation.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_performance(self, reward: float, trade_pnl: float):
        """Update agent performance metrics"""
        self.returns_history.append(reward)
        self.trades_history.append(trade_pnl)
        
        # Keep only recent history
        if len(self.returns_history) > 1000:
            self.returns_history = self.returns_history[-1000:]
            self.trades_history = self.trades_history[-1000:]
        
        # Update performance metrics
        self.performance.update(self.returns_history, self.trades_history)
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'performance': self.performance
        }, filepath)
        logger.info(f"Agent {self.agent_id} model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.performance = checkpoint['performance']
        logger.info(f"Agent {self.agent_id} model loaded from {filepath}")

class MetaAgent:
    """
    Meta-agent that coordinates multiple specialized agents
    
    Educational: The meta-agent learns to allocate capital and select
    the best agents for current market conditions.
    """
    
    def __init__(self, agents: Dict[str, TradingAgent]):
        self.agents = agents
        self.agent_ids = list(agents.keys())
        
        # Meta-agent network for agent selection
        self.meta_network = None
        self.meta_optimizer = None
        
        # Capital allocation
        self.capital_allocations = {agent_id: 1.0 / len(agents) for agent_id in self.agent_ids}
        self.allocation_history = []
        
        # Performance tracking
        self.agent_performances = {agent_id: [] for agent_id in self.agent_ids}
        self.regime_history = []
        
        # Market regime detection
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_detector = self._init_regime_detector()
        
        logger.info(f"MetaAgent initialized with {len(agents)} specialized agents")
    
    def _init_regime_detector(self):
        """Initialize market regime detector"""
        # Simple regime detection based on recent performance
        return {
            'bull_threshold': 0.02,      # 2% daily return for bull
            'bear_threshold': -0.02,     # -2% daily return for bear
            'volatility_threshold': 0.03, # 3% daily volatility
            'lookback_period': 20
        }
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime
        
        Educational: Market regime detection helps select the most
        appropriate agents for current conditions.
        """
        if len(market_data) < self.regime_detector['lookback_period']:
            return MarketRegime.SIDEWAYS
        
        recent_data = market_data.tail(self.regime_detector['lookback_period'])
        returns = recent_data['close'].pct_change().dropna()
        
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Detect regime
        if volatility > self.regime_detector['volatility_threshold']:
            if avg_return < -0.05:
                return MarketRegime.CRISIS
            else:
                return MarketRegime.HIGH_VOLATILITY
        elif avg_return > self.regime_detector['bull_threshold']:
            return MarketRegime.BULL_MARKET
        elif avg_return < self.regime_detector['bear_threshold']:
            return MarketRegime.BEAR_MARKET
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS
    
    def select_active_agents(self, regime: MarketRegime) -> List[str]:
        """
        Select agents most suitable for current regime
        
        Educational: Different agents perform better in different market conditions.
        """
        regime_agent_mapping = {
            MarketRegime.BULL_MARKET: [AgentType.TREND_FOLLOWING, AgentType.MOMENTUM],
            MarketRegime.BEAR_MARKET: [AgentType.MEAN_REVERSION, AgentType.VOLATILITY_TRADING],
            MarketRegime.SIDEWAYS: [AgentType.MEAN_REVERSION, AgentType.MARKET_MAKING],
            MarketRegime.HIGH_VOLATILITY: [AgentType.VOLATILITY_TRADING, AgentType.SENTIMENT],
            MarketRegime.LOW_VOLATILITY: [AgentType.TREND_FOLLOWING, AgentType.MOMENTUM],
            MarketRegime.CRISIS: [AgentType.SENTIMENT, AgentType.LIQUIDITY],
            MarketRegime.RECOVERY: [AgentType.TREND_FOLLOWING, AgentType.MOMENTUM]
        }
        
        preferred_types = regime_agent_mapping.get(regime, [AgentType.TREND_FOLLOWING])
        
        # Select agents with preferred types
        active_agents = []
        for agent_id, agent in self.agents.items():
            if agent.config.agent_type in preferred_types:
                active_agents.append(agent_id)
        
        # If no preferred agents, select all
        if not active_agents:
            active_agents = self.agent_ids
        
        return active_agents
    
    def update_capital_allocation(self, performances: Dict[str, AgentPerformance]):
        """
        Update capital allocation based on agent performance
        
        Educational: Capital allocation should favor better-performing agents
        while maintaining some diversification.
        """
        # Calculate performance scores
        scores = {}
        for agent_id, perf in performances.items():
            # Composite score: return + sharpe - drawdown
            score = perf.total_return + perf.sharpe_ratio - perf.max_drawdown
            scores[agent_id] = max(score, 0.01)  # Ensure positive scores
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            new_allocations = {
                agent_id: score / total_score for agent_id, score in scores.items()
            }
        else:
            new_allocations = {agent_id: 1.0 / len(self.agents) for agent_id in self.agent_ids}
        
        # Smooth allocation changes (avoid drastic shifts)
        smoothing_factor = 0.1
        for agent_id in self.agent_ids:
            self.capital_allocations[agent_id] = (
                smoothing_factor * new_allocations[agent_id] +
                (1 - smoothing_factor) * self.capital_allocations[agent_id]
            )
        
        self.allocation_history.append(self.capital_allocations.copy())
    
    def get_consensus_action(self, 
                           states: Dict[str, np.ndarray],
                           regime: MarketRegime) -> Dict[str, np.ndarray]:
        """
        Get consensus action from active agents
        
        Educational: The meta-agent combines agent actions using
        capital allocation weights.
        """
        active_agents = self.select_active_agents(regime)
        consensus_actions = {}
        
        for agent_id in active_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                state = states.get(agent_id)
                
                if state is not None:
                    action = agent.select_action(state)
                    weight = self.capital_allocations[agent_id]
                    consensus_actions[agent_id] = action * weight
        
        return consensus_actions
    
    def update_agent_performances(self, agent_id: str, performance: AgentPerformance):
        """Update performance history for an agent"""
        self.agent_performances[agent_id].append(performance)
        
        # Keep only recent history
        if len(self.agent_performances[agent_id]) > 100:
            self.agent_performances[agent_id] = self.agent_performances[agent_id][-100:]

class MultiAgentTradingSystem:
    """
    Complete multi-agent trading system
    
    Educational: This system coordinates multiple specialized agents
    to create a robust, adaptive trading strategy.
    """
    
    def __init__(self, 
                 agents_config: List[AgentConfig],
                 symbols: List[str],
                 initial_capital: float = 1000000):
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize agents
        self.agents = {}
        self.environments = {}
        
        for i, config in enumerate(agents_config):
            agent_id = f"{config.agent_type.value}_{i}"
            
            # Create environment for each agent
            env = TradingEnvironment(
                symbol=symbols[i % len(symbols)],  # Cycle through symbols
                initial_cash=initial_capital / len(agents_config)
            )
            
            # Create agent
            agent = TradingAgent(agent_id, config, env)
            
            self.agents[agent_id] = agent
            self.environments[agent_id] = env
        
        # Initialize meta-agent
        self.meta_agent = MetaAgent(self.agents)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.training_losses = []
        
        # Performance tracking
        self.portfolio_values = []
        self.global_returns = []
        
        logger.info(f"MultiAgentTradingSystem initialized with {len(self.agents)} agents")
    
    def train_episode(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Train one episode across all agents
        
        Educational: Episodes represent complete trading periods.
        Agents learn from their experiences during each episode.
        """
        episode_results = {}
        
        # Reset all environments
        for agent_id, env in self.environments.items():
            if agent_id in market_data:
                env.set_market_data(market_data[agent_id.split('_')[0]])
        
        # Reset agents
        for agent in self.agents.values():
            agent.reward_function.reset_episode()
        
        # Get initial states
        states = {}
        for agent_id, env in self.environments.items():
            obs, _ = env.reset()
            states[agent_id] = self._flatten_observation(obs)
        
        # Detect market regime
        sample_symbol = list(market_data.keys())[0]
        current_regime = self.meta_agent.detect_market_regime(market_data[sample_symbol])
        self.meta_agent.current_regime = current_regime
        
        # Episode loop
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 1000:  # Max episode length
            # Get consensus actions
            actions = self.meta_agent.get_consensus_action(states, current_regime)
            
            # Execute actions for each agent
            next_states = {}
            rewards = {}
            dones = {}
            
            for agent_id, action in actions.items():
                if agent_id in self.environments:
                    env = self.environments[agent_id]
                    agent = self.agents[agent_id]
                    
                    # Execute action
                    obs, reward, done, truncated, info = env.step(action)
                    next_state = self._flatten_observation(obs)
                    
                    # Store experience
                    agent.remember(states[agent_id], action, reward, next_state, done)
                    
                    # Train agent
                    if agent.step_count % agent.config.training_freq == 0:
                        loss = agent.train_step()
                        if loss:
                            self.training_losses.append(loss)
                    
                    # Update target network
                    if agent.step_count % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                    
                    # Decay epsilon
                    agent.decay_epsilon()
                    
                    next_states[agent_id] = next_state
                    rewards[agent_id] = reward
                    dones[agent_id] = done or truncated
                    
                    agent.step_count += 1
            
            # Update states
            states = next_states
            
            # Check if all agents are done
            if all(dones.values()):
                done = True
            
            episode_steps += 1
            self.global_step += 1
        
        # Update performances
        for agent_id, agent in self.agents.items():
            if agent_id in self.environments:
                env = self.environments[agent_id]
                stats = env.get_portfolio_stats()
                agent.performance.total_return = stats.get('total_return', 0)
                agent.performance.sharpe_ratio = stats.get('sharpe_ratio', 0)
                agent.performance.max_drawdown = stats.get('max_drawdown', 0)
                
                self.meta_agent.update_agent_performances(agent_id, agent.performance)
        
        # Update capital allocation
        self.meta_agent.update_capital_allocation({
            agent_id: agent.performance for agent_id, agent in self.agents.items()
        })
        
        # Calculate global performance
        global_performance = self._calculate_global_performance()
        
        self.episode_count += 1
        
        episode_results = {
            'episode': self.episode_count,
            'global_return': global_performance['total_return'],
            'global_sharpe': global_performance['sharpe_ratio'],
            'regime': current_regime.value,
            'num_steps': episode_steps,
            'active_agents': len(self.meta_agent.select_active_agents(current_regime))
        }
        
        logger.info(f"Episode {self.episode_count} completed: {episode_results}")
        
        return episode_results
    
    def _flatten_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten observation dictionary to single array"""
        flattened = []
        
        if isinstance(obs, dict):
            for key in ['prices', 'features', 'portfolio', 'position']:
                if key in obs:
                    flattened.append(obs[key].flatten())
        
        return np.concatenate(flattened) if flattened else np.array([])
    
    def _calculate_global_performance(self) -> Dict[str, float]:
        """Calculate global portfolio performance"""
        total_value = 0
        total_pnl = 0
        returns = []
        
        for env in self.environments.values():
            stats = env.get_portfolio_stats()
            total_value += stats.get('portfolio_value', 0)
            total_pnl += stats.get('total_pnl', 0)
            returns.append(stats.get('total_return', 0))
        
        global_return = total_pnl / self.initial_capital
        
        if returns:
            global_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            global_sharpe = 0
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return': global_return,
            'sharpe_ratio': global_sharpe
        }
    
    def save_models(self, directory: str):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_id}.pth")
            agent.save_model(filepath)
        
        # Save meta-agent state
        meta_filepath = os.path.join(directory, "meta_agent.pth")
        torch.save({
            'capital_allocations': self.meta_agent.capital_allocations,
            'allocation_history': self.meta_agent.allocation_history,
            'agent_performances': self.meta_agent.agent_performances
        }, meta_filepath)
        
        logger.info(f"All models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load all agent models"""
        import os
        
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_id}.pth")
            if os.path.exists(filepath):
                agent.load_model(filepath)
        
        # Load meta-agent state
        meta_filepath = os.path.join(directory, "meta_agent.pth")
        if os.path.exists(meta_filepath):
            checkpoint = torch.load(meta_filepath)
            self.meta_agent.capital_allocations = checkpoint['capital_allocations']
            self.meta_agent.allocation_history = checkpoint['allocation_history']
            self.meta_agent.agent_performances = checkpoint['agent_performances']
        
        logger.info(f"All models loaded from {directory}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                'type': agent.config.agent_type.value,
                'performance': agent.performance.__dict__,
                'epsilon': agent.epsilon,
                'steps': agent.step_count,
                'buffer_size': len(agent.replay_buffer)
            }
        
        return {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'current_regime': self.meta_agent.current_regime.value,
            'capital_allocations': self.meta_agent.capital_allocations,
            'agents': agent_statuses,
            'global_performance': self._calculate_global_performance()
        }

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Create Multi-Agent System:
   configs = [
       AgentConfig(AgentType.TREND_FOLLOWING, AssetClass.STOCK, RewardType.SHARPE),
       AgentConfig(AgentType.MEAN_REVERSION, AssetClass.STOCK, RewardType.SORTINO),
       AgentConfig(AgentType.VOLATILITY_TRADING, AssetClass.OPTION, RewardType.CALMAR),
       AgentConfig(AgentType.SENTIMENT, AssetClass.STOCK, RewardType.PROFIT)
   ]
   
   system = MultiAgentTradingSystem(configs, ['AAPL', 'GOOGL'], 1000000)

2. Train System:
   market_data = {
       'AAPL': load_stock_data('AAPL'),
       'GOOGL': load_stock_data('GOOGL')
   }
   
   for episode in range(100):
       results = system.train_episode(market_data)
       print(f"Episode {episode}: Return = {results['global_return']:.2%}")

3. Monitor Performance:
   status = system.get_system_status()
   print(f"Current Regime: {status['current_regime']}")
   print(f"Capital Allocations: {status['capital_allocations']}")

4. Save/Load Models:
   system.save_models('models/')
   system.load_models('models/')

Key Concepts:
- Specialized agents for different strategies
- Meta-agent for coordination and capital allocation
- Market regime detection for agent selection
- Performance-based capital allocation
- Multi-agent learning and adaptation

Educational Notes:
- Different agents excel in different market conditions
- Meta-agent learns to allocate capital optimally
- System adapts to changing market dynamics
- Diversification across strategies reduces risk
- Continuous learning improves performance over time
"""