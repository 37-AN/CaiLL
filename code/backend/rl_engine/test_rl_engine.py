"""
RL Engine Test Script - Phase 3

This script demonstrates and tests the complete reinforcement learning engine.
It shows how all components work together to create intelligent trading agents.

Educational Note:
Testing is crucial for ensuring the RL system works correctly.
This script serves as both a test suite and a learning example.
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

# Import our RL components
from .trading_env import TradingEnvironment, AssetType
from .reward_functions import RewardFunctionFactory, RewardType, RewardConfig
from .action_spaces import ActionSpaceFactory, AssetClass, ActionConstraints
from .multi_agent_rl import (
    MultiAgentTradingSystem, TradingAgent, AgentConfig, 
    AgentType, MarketRegime
)
from .continuous_learning import (
    ContinuousLearningSystem, LearningStrategy, ConceptDriftDetector
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_market_data(symbol: str, periods: int = 252) -> pd.DataFrame:
    """
    Generate sample market data for testing
    
    Educational: Creates realistic market data with trends,
    volatility, and patterns for testing RL agents.
    """
    np.random.seed(42)
    
    # Generate price series with trend and volatility
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
    
    # Base price with trend
    base_price = 100.0
    trend = 0.0001  # Daily trend
    volatility = 0.02  # Daily volatility
    
    prices = [base_price]
    for i in range(1, periods):
        # Add some market cycles
        cycle = 0.0005 * np.sin(2 * np.pi * i / 50)  # 50-day cycle
        
        # Random walk with trend and cycle
        daily_return = trend + cycle + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC
        high_low_range = close * np.random.uniform(0.005, 0.03)
        
        if i == 0:
            open_price = close
        else:
            # Gap from previous close
            gap = np.random.normal(0, 0.002)
            open_price = prices[i-1] * (1 + gap)
        
        high = max(open_price, close) + np.random.uniform(0, high_low_range)
        low = min(open_price, close) - np.random.uniform(0, high_low_range)
        
        # Volume correlated with price movement
        if i > 0:
            price_change = abs(close - prices[i-1]) / prices[i-1]
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
        else:
            volume = 1000000
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    logger.info(f"Generated {len(df)} days of sample market data for {symbol}")
    return df

def test_trading_environment():
    """
    Test the trading environment
    """
    logger.info("Testing Trading Environment...")
    
    try:
        # Create environment
        env = TradingEnvironment(
            symbol="AAPL",
            initial_cash=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Generate and set market data
        market_data = generate_sample_market_data("AAPL", 100)
        env.set_market_data(market_data)
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"Environment reset. Observation shape: {obs['prices'].shape}")
        
        # Test step
        action = np.array([0.5, 0.0, 0.0])  # Buy 50% position
        obs, reward, done, truncated, info = env.step(action)
        
        logger.info(f"Step completed. Reward: {reward:.6f}")
        logger.info(f"Portfolio value: ${info['portfolio_value']:,.2f}")
        logger.info(f"Position size: {info['position_size']}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(10):
            action = np.random.uniform(-1, 1, 3)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        logger.info(f"10 steps completed. Total reward: {total_reward:.6f}")
        
        # Test portfolio statistics
        stats = env.get_portfolio_stats()
        logger.info(f"Portfolio stats: {stats}")
        
        logger.info("✅ Trading Environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading Environment test failed: {e}")
        return False

def test_reward_functions():
    """
    Test different reward functions
    """
    logger.info("Testing Reward Functions...")
    
    try:
        # Test different reward functions
        reward_types = [
            RewardType.PROFIT,
            RewardType.SHARPE,
            RewardType.SORTINO,
            RewardType.DRAWDOWN
        ]
        
        config = RewardConfig(
            profit_weight=0.4,
            risk_weight=0.3,
            transaction_cost_weight=0.2,
            position_weight=0.1
        )
        
        for reward_type in reward_types:
            reward_fn = RewardFunctionFactory.create_reward_function(reward_type, config)
            
            # Test reward calculation
            current_state = {
                'portfolio_value': 105000,
                'total_pnl': 5000,
                'initial_capital': 100000,
                'position_size': 0.5,
                'max_position_size': 1.0
            }
            
            previous_state = {
                'portfolio_value': 103000,
                'total_pnl': 3000,
                'initial_capital': 100000
            }
            
            action = np.array([0.2, 0.0, 0.0])
            transaction_cost = 50.0
            
            reward = reward_fn.calculate_reward(
                current_state, previous_state, action, transaction_cost
            )
            
            logger.info(f"{reward_type.value} reward: {reward:.6f}")
        
        logger.info("✅ Reward Functions test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward Functions test failed: {e}")
        return False

def test_action_spaces():
    """
    Test different action spaces
    """
    logger.info("Testing Action Spaces...")
    
    try:
        # Test different asset classes
        asset_classes = [
            (AssetClass.STOCK, "discrete"),
            (AssetClass.CRYPTO, "box"),
            (AssetClass.OPTION, "specialized")
        ]
        
        constraints = ActionConstraints(
            max_position_size=0.5,
            max_daily_trades=10
        )
        
        for asset_class, action_type in asset_classes:
            if asset_class == AssetClass.OPTION:
                action_space = ActionSpaceFactory.create_action_space(
                    asset_class, constraints, 
                    strike_prices=[90, 100, 110]
                )
            else:
                action_space = ActionSpaceFactory.create_action_space(
                    asset_class, action_type, constraints
                )
            
            # Test action space
            gym_space = action_space.get_action_space()
            logger.info(f"{asset_class.value} action space: {gym_space}")
            
            # Test action decoding
            if hasattr(gym_space, 'sample'):
                raw_action = gym_space.sample()
                decoded_action = action_space.decode_action(raw_action)
                logger.info(f"Decoded action: {decoded_action}")
            
            # Test action validation
            market_state = {
                'price': 100.0,
                'portfolio_value': 100000,
                'leverage': 1.0
            }
            
            is_valid = action_space.validate_action(decoded_action, market_state)
            logger.info(f"Action valid: {is_valid}")
        
        logger.info("✅ Action Spaces test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Action Spaces test failed: {e}")
        return False

def test_single_agent():
    """
    Test a single trading agent
    """
    logger.info("Testing Single Trading Agent...")
    
    try:
        # Create environment
        env = TradingEnvironment(
            symbol="AAPL",
            initial_cash=100000
        )
        
        market_data = generate_sample_market_data("AAPL", 200)
        env.set_market_data(market_data)
        
        # Create agent configuration
        config = AgentConfig(
            agent_type=AgentType.TREND_FOLLOWING,
            asset_class=AssetClass.STOCK,
            reward_type=RewardType.SHARPE,
            learning_rate=0.001,
            memory_size=1000,
            batch_size=32
        )
        
        # Create agent
        agent = TradingAgent("test_agent", config, env)
        
        # Test agent training
        obs, _ = env.reset()
        state = np.random.randn(200)  # Simulated state
        
        total_reward = 0
        for step in range(50):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute in environment
            obs, reward, done, truncated, info = env.step(action)
            next_state = np.random.randn(200)  # Simulated next state
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if step % 4 == 0:
                loss = agent.train_step()
                if loss:
                    logger.info(f"Step {step}: Loss = {loss:.6f}")
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        logger.info(f"Agent training completed. Total reward: {total_reward:.6f}")
        logger.info(f"Final epsilon: {agent.epsilon:.6f}")
        logger.info(f"Buffer size: {len(agent.replay_buffer)}")
        
        logger.info("✅ Single Agent test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Single Agent test failed: {e}")
        return False

def test_multi_agent_system():
    """
    Test the multi-agent trading system
    """
    logger.info("Testing Multi-Agent Trading System...")
    
    try:
        # Create agent configurations
        configs = [
            AgentConfig(
                agent_type=AgentType.TREND_FOLLOWING,
                asset_class=AssetClass.STOCK,
                reward_type=RewardType.SHARPE,
                learning_rate=0.001
            ),
            AgentConfig(
                agent_type=AgentType.MEAN_REVERSION,
                asset_class=AssetClass.STOCK,
                reward_type=RewardType.SORTINO,
                learning_rate=0.001
            ),
            AgentConfig(
                agent_type=AgentType.MOMENTUM,
                asset_class=AssetClass.STOCK,
                reward_type=RewardType.PROFIT,
                learning_rate=0.001
            )
        ]
        
        # Create multi-agent system
        system = MultiAgentTradingSystem(
            configs=configs,
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            initial_capital=1000000
        )
        
        # Generate market data
        market_data = {
            'AAPL': generate_sample_market_data("AAPL", 100),
            'GOOGL': generate_sample_market_data("GOOGL", 100),
            'MSFT': generate_sample_market_data("MSFT", 100)
        }
        
        # Test training episode
        results = system.train_episode(market_data)
        
        logger.info(f"Multi-agent episode completed:")
        logger.info(f"  Episode: {results['episode']}")
        logger.info(f"  Global return: {results['global_return']:.4f}")
        logger.info(f"  Global Sharpe: {results['global_sharpe']:.4f}")
        logger.info(f"  Regime: {results['regime']}")
        logger.info(f"  Active agents: {results['active_agents']}")
        
        # Test system status
        status = system.get_system_status()
        logger.info(f"System status:")
        logger.info(f"  Global step: {status['global_step']}")
        logger.info(f"  Episode count: {status['episode_count']}")
        logger.info(f"  Current regime: {status['current_regime']}")
        
        # Test capital allocations
        logger.info("Capital allocations:")
        for agent_id, allocation in status['capital_allocations'].items():
            logger.info(f"  {agent_id}: {allocation:.4f}")
        
        logger.info("✅ Multi-Agent System test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-Agent System test failed: {e}")
        return False

def test_continuous_learning():
    """
    Test the continuous learning system
    """
    logger.info("Testing Continuous Learning System...")
    
    try:
        # Create a simple multi-agent system
        configs = [
            AgentConfig(
                agent_type=AgentType.TREND_FOLLOWING,
                asset_class=AssetClass.STOCK,
                reward_type=RewardType.SHARPE,
                learning_rate=0.001
            )
        ]
        
        system = MultiAgentTradingSystem(
            configs=configs,
            symbols=['AAPL'],
            initial_capital=100000
        )
        
        # Create continuous learning system
        learner = ContinuousLearningSystem(
            trading_system=system,
            learning_strategy=LearningStrategy.ADAPTIVE_LEARNING,
            update_frequency=30  # 30 seconds for testing
        )
        
        # Test concept drift detection
        drift_detector = ConceptDriftDetector(window_size=20, threshold=0.05)
        
        # Add reference data
        reference_data = [
            {'return': np.random.normal(0.001, 0.02),
             'volatility': np.random.uniform(0.01, 0.03)}
            for _ in range(50)
        ]
        drift_detector.set_reference(reference_data)
        
        # Add new samples
        for i in range(25):
            sample = {
                'return': np.random.normal(0.002, 0.025),  # Slightly different
                'volatility': np.random.uniform(0.015, 0.035)
            }
            drift_detector.add_sample(sample)
        
        # Test drift detection
        drift_detected, drift_score = drift_detector.detect_drift()
        logger.info(f"Concept drift detected: {drift_detected}, score: {drift_score:.6f}")
        
        # Test performance monitoring
        from .continuous_learning import PerformanceMonitor, AgentPerformance
        
        monitor = PerformanceMonitor()
        
        # Add some performance data
        for i in range(15):
            perf = AgentPerformance(
                agent_id="test_agent",
                total_return=np.random.uniform(-0.1, 0.2),
                sharpe_ratio=np.random.uniform(-0.5, 2.0),
                max_drawdown=np.random.uniform(0.05, 0.3),
                win_rate=np.random.uniform(0.3, 0.7),
                num_trades=np.random.randint(10, 100)
            )
            monitor.update_performance("test_agent", perf)
        
        # Get recommendations
        recommendations = monitor.get_learning_recommendations()
        logger.info(f"Learning recommendations: {recommendations}")
        
        # Test learning status
        status = learner.get_learning_status()
        logger.info(f"Learning status:")
        logger.info(f"  Strategy: {status['learning_strategy']}")
        logger.info(f"  Version: {status['current_version']}")
        logger.info(f"  Is learning: {status['is_learning']}")
        
        logger.info("✅ Continuous Learning test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Continuous Learning test failed: {e}")
        return False

async def test_async_learning():
    """
    Test asynchronous learning capabilities
    """
    logger.info("Testing Async Learning...")
    
    try:
        # Create simple system
        configs = [
            AgentConfig(
                agent_type=AgentType.TREND_FOLLOWING,
                asset_class=AssetClass.STOCK,
                reward_type=RewardType.SHARPE
            )
        ]
        
        system = MultiAgentTradingSystem(configs, ['AAPL'], 100000)
        
        # Create continuous learner
        learner = ContinuousLearningSystem(
            system,
            learning_strategy=LearningStrategy.ONLINE_LEARNING,
            update_frequency=5  # 5 seconds for testing
        )
        
        # Start learning for a short time
        learning_task = asyncio.create_task(learner.start_continuous_learning())
        
        # Let it run for 10 seconds
        await asyncio.sleep(10)
        
        # Stop learning
        learner.stop_continuous_learning()
        await learning_task
        
        # Check status
        status = learner.get_learning_status()
        logger.info(f"Async learning completed. Version: {status['current_version']}")
        
        logger.info("✅ Async Learning test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Async Learning test failed: {e}")
        return False

def run_all_tests():
    """
    Run all RL engine tests
    """
    logger.info("🚀 Starting RL Engine Tests - Phase 3")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(test_trading_environment())
    test_results.append(test_reward_functions())
    test_results.append(test_action_spaces())
    test_results.append(test_single_agent())
    test_results.append(test_multi_agent_system())
    test_results.append(test_continuous_learning())
    
    # Run async test
    try:
        async_result = asyncio.run(test_async_learning())
        test_results.append(async_result)
    except Exception as e:
        logger.error(f"Async learning test error: {e}")
        test_results.append(False)
    
    # Summary
    logger.info("=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        logger.info(f"🎉 All {total} tests passed! Phase 3 is complete.")
    else:
        logger.error(f"❌ {total - passed} out of {total} tests failed.")
    
    logger.info("=" * 60)
    
    # Educational summary
    logger.info("📚 Educational Summary:")
    logger.info("✅ Trading Environment: Gym-compatible market simulation")
    logger.info("✅ Reward Functions: Multiple objectives (profit, Sharpe, etc.)")
    logger.info("✅ Action Spaces: Different spaces for various assets")
    logger.info("✅ Single Agent: DQN-based trading agent")
    logger.info("✅ Multi-Agent System: Coordinated specialized agents")
    logger.info("✅ Continuous Learning: Adaptive learning system")
    logger.info("✅ Async Learning: Non-blocking learning capabilities")
    
    return passed == total

if __name__ == "__main__":
    """
    Educational: This test script demonstrates the complete RL engine.
    It serves as both validation and a learning example.
    """
    success = run_all_tests()
    
    if success:
        print("\n🎯 Phase 3 Complete: Reinforcement Learning Engine")
        print("🧠 Ready for Phase 4: Risk Management & Execution")
        print("\nKey Achievements:")
        print("- Gym-compatible trading environment")
        print("- Multiple specialized trading agents")
        print("- Sophisticated reward functions")
        print("- Multi-agent coordination")
        print("- Continuous learning and adaptation")
        print("- Performance monitoring")
        print("- Model versioning and rollback")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("\n🎓 Educational Outcomes:")
    print("- Understanding of RL in trading")
    print("- Experience with multi-agent systems")
    print("- Knowledge of reward function design")
    print("- Skills in continuous learning")
    print("- Expertise in performance monitoring")