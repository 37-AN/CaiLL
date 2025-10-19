"""
Reinforcement Learning Engine - Phase 3

This module provides a comprehensive reinforcement learning system for algorithmic trading.
It implements multiple specialized agents, sophisticated reward functions, and continuous learning.

Components:
- TradingEnvironment: Gym-compatible environment for trading simulation
- RewardFunctions: Various reward functions for different trading objectives
- ActionSpaces: Different action spaces for various asset classes
- MultiAgentRL: Coordinated multi-agent trading system
- ContinuousLearning: Adaptive learning system for market changes

Educational Notes:
Reinforcement learning is particularly well-suited for trading because:
1. Trading has sequential decision-making structure
2. Market feedback provides natural reward signals
3. Agents can learn complex strategies without explicit programming
4. Adaptation to changing market conditions is possible

Key Principles:
1. Multiple specialized agents outperform single monolithic agents
2. Reward function design shapes agent behavior significantly
3. Continuous learning is essential for non-stationary markets
4. Risk management must be built into the learning process
5. Performance monitoring ensures agents continue to perform well
"""

from .trading_env import (
    TradingEnvironment,
    MarketState,
    Position,
    Portfolio,
    Transaction,
    AssetType,
    OrderType,
    PositionSide
)

from .reward_functions import (
    BaseRewardFunction,
    ProfitRewardFunction,
    SharpeRewardFunction,
    SortinoRewardFunction,
    DrawdownRewardFunction,
    InformationRatioRewardFunction,
    CalmarRewardFunction,
    CustomRewardFunction,
    RewardFunctionFactory,
    RewardType,
    RewardConfig
)

from .action_spaces import (
    BaseActionSpace,
    DiscreteActionSpace,
    MultiDiscreteActionSpace,
    BoxActionSpace,
    OptionActionSpace,
    CryptoActionSpace,
    ActionSpaceFactory,
    ActionType,
    AssetClass,
    ActionConstraints,
    ActionResult
)

from .multi_agent_rl import (
    TradingAgent,
    MetaAgent,
    MultiAgentTradingSystem,
    DQNNetwork,
    ReplayBuffer,
    AgentConfig,
    AgentPerformance,
    AgentType,
    MarketRegime
)

from .continuous_learning import (
    ContinuousLearningSystem,
    ConceptDriftDetector,
    PerformanceMonitor,
    AdaptiveLearningRate,
    ExperienceReplay,
    LearningStrategy
)

__all__ = [
    # Trading Environment
    'TradingEnvironment',
    'MarketState',
    'Position',
    'Portfolio',
    'Transaction',
    'AssetType',
    'OrderType',
    'PositionSide',
    
    # Reward Functions
    'BaseRewardFunction',
    'ProfitRewardFunction',
    'SharpeRewardFunction',
    'SortinoRewardFunction',
    'DrawdownRewardFunction',
    'InformationRatioRewardFunction',
    'CalmarRewardFunction',
    'CustomRewardFunction',
    'RewardFunctionFactory',
    'RewardType',
    'RewardConfig',
    
    # Action Spaces
    'BaseActionSpace',
    'DiscreteActionSpace',
    'MultiDiscreteActionSpace',
    'BoxActionSpace',
    'OptionActionSpace',
    'CryptoActionSpace',
    'ActionSpaceFactory',
    'ActionType',
    'AssetClass',
    'ActionConstraints',
    'ActionResult',
    
    # Multi-Agent RL
    'TradingAgent',
    'MetaAgent',
    'MultiAgentTradingSystem',
    'DQNNetwork',
    'ReplayBuffer',
    'AgentConfig',
    'AgentPerformance',
    'AgentType',
    'MarketRegime',
    
    # Continuous Learning
    'ContinuousLearningSystem',
    'ConceptDriftDetector',
    'PerformanceMonitor',
    'AdaptiveLearningRate',
    'ExperienceReplay',
    'LearningStrategy'
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Trading System"
__description__ = "Comprehensive reinforcement learning engine for algorithmic trading"

# Educational: Quick Start Guide
QUICK_START_GUIDE = """
Quick Start Guide for Reinforcement Learning Engine

1. Basic Trading Environment:
   from rl_engine import TradingEnvironment
   
   env = TradingEnvironment(
       symbol="AAPL",
       initial_cash=100000,
       commission_rate=0.001
   )
   
   market_data = pd.read_csv('AAPL_data.csv')
   env.set_market_data(market_data)

2. Single Agent Training:
   from rl_engine import AgentConfig, TradingAgent, RewardType
   
   config = AgentConfig(
       agent_type=AgentType.TREND_FOLLOWING,
       asset_class=AssetClass.STOCK,
       reward_type=RewardType.SHARPE
   )
   
   agent = TradingAgent("trend_agent", config, env)

3. Multi-Agent System:
   from rl_engine import MultiAgentTradingSystem
   
   configs = [
       AgentConfig(AgentType.TREND_FOLLOWING, AssetClass.STOCK, RewardType.SHARPE),
       AgentConfig(AgentType.MEAN_REVERSION, AssetClass.STOCK, RewardType.SORTINO),
       AgentConfig(AgentType.VOLATILITY_TRADING, AssetClass.OPTION, RewardType.CALMAR)
   ]
   
   system = MultiAgentTradingSystem(configs, ['AAPL', 'GOOGL'], 1000000)

4. Continuous Learning:
   from rl_engine import ContinuousLearningSystem, LearningStrategy
   
   learner = ContinuousLearningSystem(
       system,
       learning_strategy=LearningStrategy.ADAPTIVE_LEARNING
   )
   
   # Start continuous learning
   import asyncio
   asyncio.create_task(learner.start_continuous_learning())

5. Custom Reward Function:
   from rl_engine import RewardConfig, RewardFunctionFactory, RewardType
   
   config = RewardConfig(
       profit_weight=0.4,
       risk_weight=0.4,
       transaction_cost_weight=0.2
   )
   
   reward_fn = RewardFunctionFactory.create_reward_function(
       RewardType.CUSTOM, config
   )

Key Features:
- Gym-compatible trading environment
- Multiple specialized trading agents
- Sophisticated reward functions
- Different action spaces for various assets
- Multi-agent coordination
- Continuous learning and adaptation
- Performance monitoring
- Model versioning and rollback

Educational Notes:
- Start with simple environments and gradually increase complexity
- Monitor agent performance closely during training
- Use appropriate reward functions for your trading objectives
- Consider risk management in all aspects of the system
- Continuous learning is essential for real-world trading
"""

# Module-level logging
import logging
logger = logging.getLogger(__name__)
logger.info("RL Engine initialized - Phase 3 Complete")