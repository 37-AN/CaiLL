"""
Continuous Learning System - Phase 3.3

This module implements continuous learning capabilities that allow the RL agents
to adapt to changing market conditions and improve their performance over time.

Educational Note:
Continuous learning is essential for trading systems because markets are
non-stationary. Strategies that work today may not work tomorrow, so agents
must continuously adapt and learn from new data.
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
import json
import pickle
from pathlib import Path
import warnings

# Import our RL components
from .multi_agent_rl import MultiAgentTradingSystem, TradingAgent, AgentPerformance
from .trading_env import TradingEnvironment
from .reward_functions import RewardFunctionFactory, RewardType

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Types of continuous learning strategies"""
    ONLINE_LEARNING = "online_learning"
    PERIODIC_RETRAINING = "periodic_retraining"
    ADAPTIVE_LEARNING = "adaptive_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE_LEARNING = "ensemble_learning"
    META_LEARNING = "meta_learning"

class ConceptDriftDetector:
    """
    Detects concept drift in market data
    
    Educational: Concept drift occurs when the statistical properties
    of the target variable change over time. In trading, this means
    market dynamics change and strategies need to adapt.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = None
        self.current_window = deque(maxlen=window_size)
        self.drift_scores = []
        self.drift_detected = False
        
    def add_sample(self, sample: Dict[str, float]):
        """Add new sample to the current window"""
        self.current_window.append(sample)
        
    def set_reference(self, reference_data: List[Dict[str, float]]):
        """Set reference window for comparison"""
        self.reference_window = reference_data[-self.window_size:]
        
    def detect_drift(self) -> Tuple[bool, float]:
        """
        Detect if concept drift has occurred
        
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        if len(self.current_window) < self.window_size or self.reference_window is None:
            return False, 0.0
        
        # Calculate drift score using KL divergence
        current_stats = self._calculate_window_stats(list(self.current_window))
        reference_stats = self._calculate_window_stats(self.reference_window)
        
        drift_score = self._calculate_kl_divergence(reference_stats, current_stats)
        self.drift_scores.append(drift_score)
        
        # Detect drift
        drift_detected = drift_score > self.threshold
        self.drift_detected = drift_detected
        
        return drift_detected, drift_score
    
    def _calculate_window_stats(self, window: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical properties of a window"""
        if not window:
            return {}
        
        stats = {}
        for key in window[0].keys():
            values = [sample[key] for sample in window if key in sample]
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def _calculate_kl_divergence(self, ref_stats: Dict, cur_stats: Dict) -> float:
        """Calculate KL divergence between reference and current statistics"""
        if not ref_stats or not cur_stats:
            return 0.0
        
        divergences = []
        
        for key in ref_stats.keys():
            if key in cur_stats:
                ref_mean, ref_std = ref_stats[key]['mean'], ref_stats[key]['std']
                cur_mean, cur_std = cur_stats[key]['mean'], cur_stats[key]['std']
                
                if ref_std > 0 and cur_std > 0:
                    # Approximate KL divergence for normal distributions
                    kl = (np.log(ref_std / cur_std) + 
                          (cur_std**2 + (cur_mean - ref_mean)**2) / (2 * ref_std**2) - 0.5)
                    divergences.append(abs(kl))
        
        return np.mean(divergences) if divergences else 0.0

class PerformanceMonitor:
    """
    Monitors agent performance and triggers learning when needed
    
    Educational: Performance monitoring helps identify when agents
    are underperforming and need retraining or adaptation.
    """
    
    def __init__(self, 
                 performance_window: int = 50,
                 degradation_threshold: float = -0.1,
                 improvement_threshold: float = 0.05):
        
        self.performance_window = performance_window
        self.degradation_threshold = degradation_threshold
        self.improvement_threshold = improvement_threshold
        
        self.performance_history = defaultdict(lambda: deque(maxlen=performance_window))
        self.baseline_performance = {}
        self.alerts = []
        
    def update_performance(self, agent_id: str, performance: AgentPerformance):
        """Update performance for an agent"""
        self.performance_history[agent_id].append(performance)
        
        # Set baseline if not set
        if agent_id not in self.baseline_performance and len(self.performance_history[agent_id]) >= 10:
            recent_performances = list(self.performance_history[agent_id])[:10]
            avg_return = np.mean([p.total_return for p in recent_performances])
            avg_sharpe = np.mean([p.sharpe_ratio for p in recent_performances])
            
            self.baseline_performance[agent_id] = {
                'return': avg_return,
                'sharpe': avg_sharpe,
                'timestamp': datetime.now()
            }
    
    def check_performance_degradation(self, agent_id: str) -> bool:
        """Check if agent performance has degraded"""
        if agent_id not in self.baseline_performance:
            return False
        
        if len(self.performance_history[agent_id]) < 10:
            return False
        
        baseline = self.baseline_performance[agent_id]
        recent_performances = list(self.performance_history[agent_id])[-10:]
        
        avg_return = np.mean([p.total_return for p in recent_performances])
        return_change = (avg_return - baseline['return']) / baseline['return']
        
        return return_change < self.degradation_threshold
    
    def check_improvement_opportunity(self, agent_id: str) -> bool:
        """Check if there's an opportunity for improvement"""
        if len(self.performance_history[agent_id]) < 20:
            return False
        
        recent_performances = list(self.performance_history[agent_id])
        first_half = recent_performances[:10]
        second_half = recent_performances[-10:]
        
        first_return = np.mean([p.total_return for p in first_half])
        second_return = np.mean([p.total_return for p in second_half])
        
        improvement = (second_return - first_return) / abs(first_return) if first_return != 0 else 0
        
        return improvement > self.improvement_threshold
    
    def get_learning_recommendations(self) -> Dict[str, List[str]]:
        """Get learning recommendations for all agents"""
        recommendations = {}
        
        for agent_id in self.performance_history.keys():
            agent_recommendations = []
            
            if self.check_performance_degradation(agent_id):
                agent_recommendations.append("retrain")
            
            if self.check_improvement_opportunity(agent_id):
                agent_recommendations.append("fine_tune")
            
            # Check for high volatility
            recent_performances = list(self.performance_history[agent_id])[-10:]
            if recent_performances:
                avg_volatility = np.mean([p.volatility for p in recent_performances])
                if avg_volatility > 0.3:  # High volatility threshold
                    agent_recommendations.append("adjust_risk")
            
            if agent_recommendations:
                recommendations[agent_id] = agent_recommendations
        
        return recommendations

class AdaptiveLearningRate:
    """
    Adaptive learning rate scheduler
    
    Educational: Learning rate adaptation helps agents converge
    faster and avoid getting stuck in local optima.
    """
    
    def __init__(self, 
                 initial_lr: float = 0.001,
                 min_lr: float = 0.00001,
                 max_lr: float = 0.01,
                 patience: int = 10,
                 factor: float = 0.5):
        
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.loss_history = []
        
    def update(self, loss: float) -> float:
        """Update learning rate based on loss"""
        self.loss_history.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
            # Increase learning rate when improving
            self.current_lr = min(self.current_lr * 1.1, self.max_lr)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                # Decrease learning rate when not improving
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.num_bad_epochs = 0
        
        return self.current_lr
    
    def reset(self):
        """Reset scheduler state"""
        self.current_lr = self.initial_lr
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.loss_history = []

class ExperienceReplay:
    """
    Advanced experience replay with prioritization
    
    Educational: Prioritized experience replay focuses on
    learning from the most informative experiences.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done, error: float = None):
        """Add experience with priority"""
        max_priority = error if error is not None else self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
        self.max_priority = max(self.max_priority, max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritization"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get samples
        samples = [self.buffer[i] for i in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices: List[int], errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6  # Small epsilon to avoid zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])

class ContinuousLearningSystem:
    """
    Main continuous learning system that coordinates all learning components
    
    Educational: This system manages the entire continuous learning process,
    from detecting market changes to adapting agent strategies.
    """
    
    def __init__(self, 
                 trading_system: MultiAgentTradingSystem,
                 learning_strategy: LearningStrategy = LearningStrategy.ADAPTIVE_LEARNING,
                 update_frequency: int = 100,
                 save_frequency: int = 1000):
        
        self.trading_system = trading_system
        self.learning_strategy = learning_strategy
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        
        # Learning components
        self.concept_drift_detector = ConceptDriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_lr = AdaptiveLearningRate()
        
        # Learning state
        self.learning_step = 0
        self.last_update = datetime.now()
        self.learning_history = []
        
        # Model versioning
        self.model_versions = {}
        self.current_version = 0
        
        # Async learning
        self.learning_queue = asyncio.Queue()
        self.is_learning = False
        
        logger.info(f"ContinuousLearningSystem initialized with {learning_strategy.value}")
    
    async def start_continuous_learning(self):
        """Start the continuous learning loop"""
        self.is_learning = True
        
        while self.is_learning:
            try:
                # Check if learning is needed
                if self._should_trigger_learning():
                    await self._execute_learning_cycle()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop_continuous_learning(self):
        """Stop the continuous learning loop"""
        self.is_learning = False
        logger.info("Continuous learning stopped")
    
    def _should_trigger_learning(self) -> bool:
        """Check if learning should be triggered"""
        # Check time-based trigger
        time_since_update = datetime.now() - self.last_update
        if time_since_update.total_seconds() > self.update_frequency:
            return True
        
        # Check performance-based trigger
        recommendations = self.performance_monitor.get_learning_recommendations()
        if any(recommendations.values()):
            return True
        
        # Check concept drift
        drift_detected, drift_score = self.concept_drift_detector.detect_drift()
        if drift_detected:
            logger.info(f"Concept drift detected (score: {drift_score:.4f})")
            return True
        
        return False
    
    async def _execute_learning_cycle(self):
        """Execute a complete learning cycle"""
        logger.info("Starting learning cycle")
        
        try:
            # Update performance monitoring
            await self._update_performance_monitoring()
            
            # Detect learning needs
            learning_needs = self._assess_learning_needs()
            
            # Execute learning based on strategy
            if self.learning_strategy == LearningStrategy.ONLINE_LEARNING:
                await self._online_learning(learning_needs)
            elif self.learning_strategy == LearningStrategy.PERIODIC_RETRAINING:
                await self._periodic_retraining(learning_needs)
            elif self.learning_strategy == LearningStrategy.ADAPTIVE_LEARNING:
                await self._adaptive_learning(learning_needs)
            elif self.learning_strategy == LearningStrategy.TRANSFER_LEARNING:
                await self._transfer_learning(learning_needs)
            elif self.learning_strategy == LearningStrategy.ENSEMBLE_LEARNING:
                await self._ensemble_learning(learning_needs)
            elif self.learning_strategy == LearningStrategy.META_LEARNING:
                await self._meta_learning(learning_needs)
            
            # Update model version
            self.current_version += 1
            self._save_model_version()
            
            # Update learning history
            self._update_learning_history(learning_needs)
            
            self.last_update = datetime.now()
            self.learning_step += 1
            
            logger.info(f"Learning cycle completed (version {self.current_version})")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def _update_performance_monitoring(self):
        """Update performance monitoring for all agents"""
        for agent_id, agent in self.trading_system.agents.items():
            self.performance_monitor.update_performance(agent_id, agent.performance)
    
    def _assess_learning_needs(self) -> Dict[str, List[str]]:
        """Assess what learning is needed for each agent"""
        needs = {}
        
        for agent_id in self.trading_system.agents.keys():
            agent_needs = []
            
            # Check performance degradation
            if self.performance_monitor.check_performance_degradation(agent_id):
                agent_needs.append("retrain")
            
            # Check improvement opportunity
            if self.performance_monitor.check_improvement_opportunity(agent_id):
                agent_needs.append("fine_tune")
            
            # Check concept drift
            if self.concept_drift_detector.drift_detected:
                agent_needs.append("adapt")
            
            # Check exploration rate
            agent = self.trading_system.agents[agent_id]
            if agent.epsilon > 0.5:  # Still exploring a lot
                agent_needs.append("explore")
            
            if agent_needs:
                needs[agent_id] = agent_needs
        
        return needs
    
    async def _online_learning(self, learning_needs: Dict[str, List[str]]):
        """Execute online learning strategy"""
        for agent_id, needs in learning_needs.items():
            agent = self.trading_system.agents[agent_id]
            
            # Adjust learning rate based on performance
            if agent.training_losses:
                current_loss = agent.training_losses[-1]
                new_lr = self.adaptive_lr.update(current_loss)
                
                # Update optimizer learning rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            # Additional training steps
            if "retrain" in needs or "fine_tune" in needs:
                additional_steps = 100
                for _ in range(additional_steps):
                    loss = agent.train_step()
                    if loss and agent.training_losses:
                        agent.training_losses.append(loss)
    
    async def _periodic_retraining(self, learning_needs: Dict[str, List[str]]):
        """Execute periodic retraining strategy"""
        for agent_id, needs in learning_needs.items():
            if "retrain" in needs:
                agent = self.trading_system.agents[agent_id]
                
                # Reset epsilon for more exploration
                agent.epsilon = min(0.5, agent.epsilon * 2)
                
                # Retrain with more episodes
                retrain_episodes = 10
                for _ in range(retrain_episodes):
                    # Simulate additional training episodes
                    await self._simulate_training_episode(agent)
    
    async def _adaptive_learning(self, learning_needs: Dict[str, List[str]]):
        """Execute adaptive learning strategy"""
        for agent_id, needs in learning_needs.items():
            agent = self.trading_system.agents[agent_id]
            
            # Adapt based on specific needs
            if "retrain" in needs:
                # Full retraining with reset
                agent.epsilon = agent.config.epsilon_start
                self.adaptive_lr.reset()
                
            elif "fine_tune" in needs:
                # Fine-tuning with lower learning rate
                current_lr = agent.optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.5
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
            elif "adapt" in needs:
                # Adapt to concept drift
                agent.epsilon = min(0.3, agent.epsilon * 1.5)
                
            elif "explore" in needs:
                # Increase exploration
                agent.epsilon = min(0.8, agent.epsilon * 1.2)
    
    async def _transfer_learning(self, learning_needs: Dict[str, List[str]]):
        """Execute transfer learning strategy"""
        # Find best performing agent
        best_agent_id = max(
            self.trading_system.agents.keys(),
            key=lambda aid: self.trading_system.agents[aid].performance.total_return
        )
        
        best_agent = self.trading_system.agents[best_agent_id]
        
        for agent_id, needs in learning_needs.items():
            if agent_id != best_agent_id and "retrain" in needs:
                agent = self.trading_system.agents[agent_id]
                
                # Transfer knowledge from best agent
                agent.q_network.load_state_dict(best_agent.q_network.state_dict())
                
                # Fine-tune with lower learning rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= 0.1
    
    async def _ensemble_learning(self, learning_needs: Dict[str, List[str]]):
        """Execute ensemble learning strategy"""
        # Create ensemble of best performing agents
        performing_agents = []
        for agent_id, agent in self.trading_system.agents.items():
            if agent.performance.total_return > 0:
                performing_agents.append(agent)
        
        if len(performing_agents) >= 2:
            # Average the weights of performing agents
            avg_state_dict = {}
            
            for agent in performing_agents:
                state_dict = agent.q_network.state_dict()
                for key in state_dict.keys():
                    if key not in avg_state_dict:
                        avg_state_dict[key] = 0
                    avg_state_dict[key] += state_dict[key] / len(performing_agents)
            
            # Update underperforming agents with ensemble weights
            for agent_id, needs in learning_needs.items():
                if "retrain" in needs:
                    agent = self.trading_system.agents[agent_id]
                    agent.q_network.load_state_dict(avg_state_dict)
    
    async def _meta_learning(self, learning_needs: Dict[str, List[str]]):
        """Execute meta-learning strategy"""
        # Learn how to learn - adapt learning strategies based on past performance
        for agent_id, needs in learning_needs.items():
            agent = self.trading_system.agents[agent_id]
            
            # Analyze past learning patterns
            if len(self.learning_history) > 10:
                recent_history = self.learning_history[-10:]
                
                # Find most successful learning actions
                successful_actions = []
                for record in recent_history:
                    if record.get('success', False):
                        successful_actions.extend(record.get('actions', []))
                
                if successful_actions:
                    # Apply most successful actions
                    most_common = max(set(successful_actions), key=successful_actions.count)
                    
                    if most_common == "increase_lr":
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] *= 1.2
                    elif most_common == "decrease_lr":
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] *= 0.8
                    elif most_common == "increase_epsilon":
                        agent.epsilon = min(0.8, agent.epsilon * 1.2)
                    elif most_common == "decrease_epsilon":
                        agent.epsilon = max(0.01, agent.epsilon * 0.8)
    
    async def _simulate_training_episode(self, agent: TradingAgent):
        """Simulate a training episode for an agent"""
        # This would typically involve running the agent in a simulation
        # For now, we'll just do additional training steps
        for _ in range(50):
            loss = agent.train_step()
            if loss:
                agent.training_losses.append(loss)
    
    def _save_model_version(self):
        """Save current model version"""
        version_dir = Path(f"models/version_{self.current_version}")
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all agents
        for agent_id, agent in self.trading_system.agents.items():
            model_path = version_dir / f"{agent_id}.pth"
            agent.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'learning_step': self.learning_step,
            'performance': {
                agent_id: agent.performance.__dict__
                for agent_id, agent in self.trading_system.agents.items()
            },
            'learning_strategy': self.learning_strategy.value
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.model_versions[self.current_version] = str(version_dir)
        
        logger.info(f"Model version {self.current_version} saved")
    
    def _update_learning_history(self, learning_needs: Dict[str, List[str]]):
        """Update learning history"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'version': self.current_version,
            'learning_needs': learning_needs,
            'actions': [],
            'success': False
        }
        
        # Determine actions taken
        for needs in learning_needs.values():
            record['actions'].extend(needs)
        
        # Determine success (will be updated based on future performance)
        self.learning_history.append(record)
        
        # Keep only recent history
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
    
    def add_market_sample(self, sample: Dict[str, float]):
        """Add market sample for concept drift detection"""
        self.concept_drift_detector.add_sample(sample)
    
    def set_reference_market_data(self, market_data: List[Dict[str, float]]):
        """Set reference market data for concept drift detection"""
        self.concept_drift_detector.set_reference(market_data)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        return {
            'learning_strategy': self.learning_strategy.value,
            'learning_step': self.learning_step,
            'current_version': self.current_version,
            'last_update': self.last_update.isoformat(),
            'is_learning': self.is_learning,
            'concept_drift_detected': self.concept_drift_detector.drift_detected,
            'drift_score': self.concept_drift_detector.drift_scores[-1] if self.concept_drift_detector.drift_scores else 0,
            'learning_recommendations': self.performance_monitor.get_learning_recommendations(),
            'model_versions': list(self.model_versions.keys()),
            'recent_learning_history': self.learning_history[-5:] if self.learning_history else []
        }
    
    def rollback_to_version(self, version: int):
        """Rollback to a previous model version"""
        if version not in self.model_versions:
            raise ValueError(f"Version {version} not found")
        
        version_dir = Path(self.model_versions[version])
        
        # Load all agents from version
        for agent_id, agent in self.trading_system.agents.items():
            model_path = version_dir / f"{agent_id}.pth"
            if model_path.exists():
                agent.load_model(str(model_path))
        
        self.current_version = version
        logger.info(f"Rolled back to version {version}")

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Setup Continuous Learning:
   system = MultiAgentTradingSystem(configs, symbols, capital)
   continuous_learner = ContinuousLearningSystem(
       system,
       learning_strategy=LearningStrategy.ADAPTIVE_LEARNING,
       update_frequency=300  # 5 minutes
   )

2. Start Continuous Learning:
   import asyncio
   
   async def run_learning():
       await continuous_learner.start_continuous_learning()
   
   # Run in background
   learning_task = asyncio.create_task(run_learning())

3. Monitor Learning:
   status = continuous_learner.get_learning_status()
   print(f"Current version: {status['current_version']}")
   print(f"Concept drift: {status['concept_drift_detected']}")
   print(f"Recommendations: {status['learning_recommendations']}")

4. Manual Learning Trigger:
   # Add market data for drift detection
   market_sample = {'return': 0.02, 'volatility': 0.03, 'volume': 1000000}
   continuous_learner.add_market_sample(market_sample)
   
   # Trigger learning cycle
   await continuous_learner._execute_learning_cycle()

5. Model Management:
   # Save current version
   continuous_learner._save_model_version()
   
   # Rollback to previous version
   continuous_learner.rollback_to_version(5)

Key Concepts:
- Concept drift detection for market changes
- Performance monitoring for degradation detection
- Adaptive learning rates for better convergence
- Prioritized experience replay for efficient learning
- Model versioning for rollback capabilities
- Multiple learning strategies for different scenarios

Educational Notes:
- Markets are non-stationary - continuous learning is essential
- Different learning strategies work for different situations
- Performance monitoring helps identify when adaptation is needed
- Model versioning provides safety nets for experimentation
- Async learning allows trading to continue during training
- Concept drift detection helps anticipate market changes
"""