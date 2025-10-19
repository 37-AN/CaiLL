"""
Circuit Breakers System - AI Trading System

This module implements circuit breakers and safety mechanisms to protect
the trading system from extreme market conditions and technical failures.

Educational Note:
Circuit breakers are essential safety mechanisms that automatically
stop trading when predefined risk thresholds are breached. They protect
capital during market crises and system failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    PORTFOLIO_LOSS = "portfolio_loss"
    POSITION_LOSS = "position_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    TECHNICAL = "technical"
    MARKET_WIDE = "market_wide"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""
    name: str
    breaker_type: CircuitBreakerType
    threshold: float
    lookback_period: int  # in minutes
    cooldown_period: int  # in minutes
    auto_recovery: bool = True
    action_on_trigger: str = "stop_trading"  # stop_trading, reduce_positions, alert_only
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertMessage:
    """Alert message structure"""
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerStatus:
    """Current status of a circuit breaker"""
    name: str
    state: CircuitBreakerState
    current_value: float
    threshold: float
    last_triggered: Optional[datetime]
    last_reset: Optional[datetime]
    trigger_count: int
    cooldown_until: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker(ABC):
    """Abstract base class for circuit breakers"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.NORMAL
        self.current_value = 0.0
        self.last_triggered = None
        self.last_reset = datetime.now()
        self.trigger_count = 0
        self.cooldown_until = None
        self.metadata = {}
        
    @abstractmethod
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate the metric this circuit breaker monitors"""
        pass
    
    @abstractmethod
    def should_trigger(self, metric_value: float) -> bool:
        """Determine if circuit breaker should trigger"""
        pass
    
    def check(self, data: Dict[str, Any]) -> bool:
        """Check if circuit breaker should trigger"""
        if not self.config.enabled:
            return False
        
        # Check cooldown period
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False
        
        # Calculate metric
        self.current_value = self.calculate_metric(data)
        
        # Check if should trigger
        if self.should_trigger(self.current_value):
            self.trigger()
            return True
        
        # Check for recovery
        if self.state == CircuitBreakerState.TRIGGERED and self.config.auto_recovery:
            if self.should_recover(data):
                self.recover()
        
        return False
    
    def trigger(self):
        """Trigger the circuit breaker"""
        self.state = CircuitBreakerState.TRIGGERED
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        self.cooldown_until = datetime.now() + timedelta(minutes=self.config.cooldown_period)
        
        logger.warning(f"Circuit breaker '{self.config.name}' triggered at {self.current_value:.4f}")
    
    def recover(self):
        """Recover from triggered state"""
        self.state = CircuitBreakerState.RECOVERING
        self.last_reset = datetime.now()
        logger.info(f"Circuit breaker '{self.config.name}' recovering")
    
    def reset(self):
        """Reset circuit breaker to normal state"""
        self.state = CircuitBreakerState.NORMAL
        self.current_value = 0.0
        self.last_reset = datetime.now()
        logger.info(f"Circuit breaker '{self.config.name}' reset to normal")
    
    def should_recover(self, data: Dict[str, Any]) -> bool:
        """Check if circuit breaker should recover"""
        # Default recovery logic: metric must be below threshold for some time
        metric_value = self.calculate_metric(data)
        return not self.should_trigger(metric_value)


class PortfolioLossBreaker(CircuitBreaker):
    """
    Portfolio Loss Circuit Breaker
    
    Educational Note:
    Stops trading when portfolio losses exceed threshold.
    This is the most basic and important risk control.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate portfolio loss percentage"""
        portfolio_value = data.get('portfolio_value', 0)
        initial_value = data.get('initial_portfolio_value', portfolio_value)
        
        if initial_value == 0:
            return 0.0
        
        return (initial_value - portfolio_value) / initial_value
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if loss exceeds threshold"""
        return metric_value > self.config.threshold


class PositionLossBreaker(CircuitBreaker):
    """
    Position Loss Circuit Breaker
    
    Educational Note:
    Monitors individual position losses to prevent
    catastrophic losses from single positions.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate maximum position loss"""
        positions = data.get('positions', {})
        max_loss = 0.0
        
        for symbol, position_data in positions.items():
            current_value = position_data.get('current_value', 0)
            entry_value = position_data.get('entry_value', current_value)
            
            if entry_value > 0:
                loss = (entry_value - current_value) / entry_value
                max_loss = max(max_loss, loss)
        
        return max_loss
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if any position loss exceeds threshold"""
        return metric_value > self.config.threshold


class DrawdownBreaker(CircuitBreaker):
    """
    Drawdown Circuit Breaker
    
    Educational Note:
    Stops trading when drawdown from peak exceeds threshold.
    Drawdown is a better measure of risk than simple losses.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate current drawdown"""
        portfolio_values = data.get('portfolio_history', [])
        
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = max(portfolio_values)
        current = portfolio_values[-1]
        
        if peak == 0:
            return 0.0
        
        return (peak - current) / peak
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if drawdown exceeds threshold"""
        return metric_value > self.config.threshold


class VolatilityBreaker(CircuitBreaker):
    """
    Volatility Circuit Breaker
    
    Educational Note:
    Stops trading when volatility becomes extreme.
    High volatility often precedes market crashes.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate portfolio volatility"""
        returns = data.get('portfolio_returns', [])
        
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if volatility exceeds threshold"""
        return metric_value > self.config.threshold


class LeverageBreaker(CircuitBreaker):
    """
    Leverage Circuit Breaker
    
    Educational Note:
    Prevents excessive leverage that could lead to
    margin calls and forced liquidation.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate portfolio leverage"""
        total_value = data.get('total_position_value', 0)
        portfolio_value = data.get('portfolio_value', 0)
        
        if portfolio_value == 0:
            return 0.0
        
        return total_value / portfolio_value
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if leverage exceeds threshold"""
        return metric_value > self.config.threshold


class ConcentrationBreaker(CircuitBreaker):
    """
    Concentration Risk Circuit Breaker
    
    Educational Note:
    Prevents excessive concentration in single positions
    or sectors, which increases portfolio risk.
    """
    
    def calculate_metric(self, data: Dict[str, Any]) -> float:
        """Calculate maximum position concentration"""
        positions = data.get('positions', {})
        total_value = data.get('total_position_value', 0)
        
        if total_value == 0:
            return 0.0
        
        max_concentration = 0.0
        for position_data in positions.values():
            position_value = position_data.get('current_value', 0)
            concentration = position_value / total_value
            max_concentration = max(max_concentration, concentration)
        
        return max_concentration
    
    def should_trigger(self, metric_value: float) -> bool:
        """Trigger if concentration exceeds threshold"""
        return metric_value > self.config.threshold


class CircuitBreakerManager:
    """
    Circuit Breaker Manager
    
    Educational Note:
    Coordinates all circuit breakers and manages their states.
    Provides centralized risk monitoring and alerting.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.alert_handlers: List[Callable] = []
        self.trading_enabled = True
        self.last_check = datetime.now()
        self.check_interval = 60  # seconds
        
        # Register default circuit breakers
        self._register_default_breakers()
    
    def _register_default_breakers(self):
        """Register default circuit breakers with conservative settings"""
        
        default_configs = [
            CircuitBreakerConfig(
                name="Portfolio Loss",
                breaker_type=CircuitBreakerType.PORTFOLIO_LOSS,
                threshold=0.15,  # 15% portfolio loss
                lookback_period=1,
                cooldown_period=60,
                auto_recovery=True,
                action_on_trigger="stop_trading"
            ),
            CircuitBreakerConfig(
                name="Position Loss",
                breaker_type=CircuitBreakerType.POSITION_LOSS,
                threshold=0.25,  # 25% position loss
                lookback_period=5,
                cooldown_period=30,
                auto_recovery=True,
                action_on_trigger="reduce_positions"
            ),
            CircuitBreakerConfig(
                name="Drawdown",
                breaker_type=CircuitBreakerType.DRAWDOWN,
                threshold=0.20,  # 20% drawdown
                lookback_period=1440,  # 24 hours
                cooldown_period=120,
                auto_recovery=True,
                action_on_trigger="stop_trading"
            ),
            CircuitBreakerConfig(
                name="Volatility",
                breaker_type=CircuitBreakerType.VOLATILITY,
                threshold=0.50,  # 50% annual volatility
                lookback_period=60,
                cooldown_period=30,
                auto_recovery=True,
                action_on_trigger="reduce_positions"
            ),
            CircuitBreakerConfig(
                name="Leverage",
                breaker_type=CircuitBreakerType.LEVERAGE,
                threshold=3.0,  # 3x leverage
                lookback_period=1,
                cooldown_period=15,
                auto_recovery=True,
                action_on_trigger="reduce_positions"
            ),
            CircuitBreakerConfig(
                name="Concentration",
                breaker_type=CircuitBreakerType.CONCENTRATION,
                threshold=0.40,  # 40% in single position
                lookback_period=1,
                cooldown_period=30,
                auto_recovery=True,
                action_on_trigger="alert_only"
            )
        ]
        
        for config in default_configs:
            self.register_circuit_breaker(config)
    
    def register_circuit_breaker(self, config: CircuitBreakerConfig):
        """Register a new circuit breaker"""
        
        breaker_classes = {
            CircuitBreakerType.PORTFOLIO_LOSS: PortfolioLossBreaker,
            CircuitBreakerType.POSITION_LOSS: PositionLossBreaker,
            CircuitBreakerType.DRAWDOWN: DrawdownBreaker,
            CircuitBreakerType.VOLATILITY: VolatilityBreaker,
            CircuitBreakerType.LEVERAGE: LeverageBreaker,
            CircuitBreakerType.CONCENTRATION: ConcentrationBreaker
        }
        
        breaker_class = breaker_classes.get(config.breaker_type)
        if not breaker_class:
            raise ValueError(f"Unknown circuit breaker type: {config.breaker_type}")
        
        breaker = breaker_class(config)
        self.circuit_breakers[config.name] = breaker
        
        logger.info(f"Registered circuit breaker: {config.name}")
    
    def add_alert_handler(self, handler: Callable[[AlertMessage], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def send_alert(self, level: AlertLevel, message: str, source: str, data: Dict = None):
        """Send alert to all handlers"""
        alert = AlertMessage(
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            data=data or {}
        )
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def check_circuit_breakers(self, data: Dict[str, Any]) -> List[str]:
        """Check all circuit breakers and return triggered ones"""
        
        triggered_breakers = []
        
        for name, breaker in self.circuit_breakers.items():
            try:
                if breaker.check(data):
                    triggered_breakers.append(name)
                    
                    # Send alert
                    self.send_alert(
                        AlertLevel.CRITICAL,
                        f"Circuit breaker '{name}' triggered: {breaker.current_value:.4f} > {breaker.config.threshold:.4f}",
                        "CircuitBreakerManager",
                        {
                            'breaker_name': name,
                            'current_value': breaker.current_value,
                            'threshold': breaker.config.threshold,
                            'action': breaker.config.action_on_trigger
                        }
                    )
                    
                    # Execute action
                    self._execute_breaker_action(breaker)
                    
            except Exception as e:
                logger.error(f"Error checking circuit breaker {name}: {e}")
                self.send_alert(
                    AlertLevel.WARNING,
                    f"Error checking circuit breaker '{name}': {str(e)}",
                    "CircuitBreakerManager"
                )
        
        # Update trading status
        self._update_trading_status()
        
        return triggered_breakers
    
    def _execute_breaker_action(self, breaker: CircuitBreaker):
        """Execute action when circuit breaker triggers"""
        
        action = breaker.config.action_on_trigger
        
        if action == "stop_trading":
            self.trading_enabled = False
            logger.critical(f"Trading stopped due to {breaker.config.name} circuit breaker")
            
        elif action == "reduce_positions":
            # This would signal the trading system to reduce positions
            logger.warning(f"Position reduction triggered by {breaker.config.name} circuit breaker")
            
        elif action == "alert_only":
            # Only send alert, don't change trading state
            pass
    
    def _update_trading_status(self):
        """Update overall trading status based on circuit breakers"""
        
        # Check if any critical breakers are triggered
        critical_triggered = any(
            breaker.state == CircuitBreakerState.TRIGGERED and
            breaker.config.action_on_trigger == "stop_trading"
            for breaker in self.circuit_breakers.values()
        )
        
        if critical_triggered:
            self.trading_enabled = False
        else:
            # Check if all breakers are in normal state
            all_normal = all(
                breaker.state == CircuitBreakerState.NORMAL
                for breaker in self.circuit_breakers.values()
            )
            
            if all_normal and not self.trading_enabled:
                self.trading_enabled = True
                logger.info("Trading re-enabled - all circuit breakers normal")
    
    def get_status(self) -> Dict[str, CircuitBreakerStatus]:
        """Get status of all circuit breakers"""
        
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = CircuitBreakerStatus(
                name=name,
                state=breaker.state,
                current_value=breaker.current_value,
                threshold=breaker.config.threshold,
                last_triggered=breaker.last_triggered,
                last_reset=breaker.last_reset,
                trigger_count=breaker.trigger_count,
                cooldown_until=breaker.cooldown_until,
                metadata=breaker.metadata
            )
        
        return status
    
    def reset_breaker(self, name: str):
        """Manually reset a circuit breaker"""
        
        if name in self.circuit_breakers:
            self.circuit_breakers[name].reset()
            logger.info(f"Manually reset circuit breaker: {name}")
        else:
            logger.warning(f"Circuit breaker not found: {name}")
    
    def enable_breaker(self, name: str):
        """Enable a circuit breaker"""
        
        if name in self.circuit_breakers:
            self.circuit_breakers[name].config.enabled = True
            logger.info(f"Enabled circuit breaker: {name}")
    
    def disable_breaker(self, name: str):
        """Disable a circuit breaker"""
        
        if name in self.circuit_breakers:
            self.circuit_breakers[name].config.enabled = False
            logger.warning(f"Disabled circuit breaker: {name}")
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        return self.trading_enabled
    
    async def start_monitoring(self, data_provider: Callable[[], Dict[str, Any]]):
        """Start continuous monitoring"""
        
        logger.info("Starting circuit breaker monitoring")
        
        while True:
            try:
                # Get current data
                data = data_provider()
                
                # Check circuit breakers
                triggered = self.check_circuit_breakers(data)
                
                if triggered:
                    logger.warning(f"Triggered circuit breakers: {triggered}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring: {e}")
                await asyncio.sleep(self.check_interval)


def console_alert_handler(alert: AlertMessage):
    """Simple console alert handler"""
    
    level_symbols = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.CRITICAL: "🚨",
        AlertLevel.EMERGENCY: "🆘"
    }
    
    symbol = level_symbols.get(alert.level, "📢")
    timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{symbol} [{timestamp}] {alert.level.value.upper()}: {alert.message}")
    
    if alert.data:
        print(f"   Data: {alert.data}")


def explain_circuit_breakers():
    """
    Educational explanation of circuit breakers
    """
    
    print("=== Circuit Breakers Educational Guide ===\n")
    
    concepts = {
        'Circuit Breaker': "Automatic safety mechanism that stops trading when risk thresholds are breached",
        
        'Portfolio Loss': "Stops trading when overall portfolio losses exceed a set percentage",
        
        'Position Loss': "Monitors individual positions to prevent catastrophic single-position losses",
        
        'Drawdown': "Measures decline from portfolio peak, better than simple loss measurement",
        
        'Volatility': "Detects extreme market conditions that often precede crashes",
        
        'Leverage': "Prevents excessive borrowing that could lead to margin calls",
        
        'Concentration': "Limits exposure to single positions or sectors",
        
        'Cooldown Period': "Time after triggering when breaker cannot trigger again",
        
        'Auto Recovery': "Automatic re-enabling of trading when conditions normalize"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Circuit Breaker Best Practices ===")
    print("1. Set conservative thresholds - better to be safe than sorry")
    print("2. Use multiple breakers for different types of risk")
    print("3. Implement appropriate cooldown periods")
    print("4. Monitor breaker performance and adjust thresholds")
    print("5. Have manual override capabilities for emergencies")
    print("6. Test breakers regularly with historical data")
    print("7. Document all breaker configurations and changes")
    
    print("\n=== Real-World Examples ===")
    print("• 2020 COVID Crash: Many funds stopped trading after 15-20% losses")
    print("• 2008 Financial Crisis: Volatility breakers triggered across markets")
    print("• Flash Crashes: Circuit breakers prevent automated trading cascades")
    print("• Individual Stocks: Position loss breakers prevent total loss from single stocks")


if __name__ == "__main__":
    # Example usage
    explain_circuit_breakers()
    
    # Create circuit breaker manager
    manager = CircuitBreakerManager()
    
    # Add console alert handler
    manager.add_alert_handler(console_alert_handler)
    
    # Simulate market data
    sample_data = {
        'portfolio_value': 85000,  # Down 15% from 100k
        'initial_portfolio_value': 100000,
        'positions': {
            'AAPL': {'current_value': 30000, 'entry_value': 40000},  # Down 25%
            'MSFT': {'current_value': 25000, 'entry_value': 25000},
            'GOOGL': {'current_value': 30000, 'entry_value': 35000}   # Down 14%
        },
        'total_position_value': 85000,
        'portfolio_history': [100000, 95000, 90000, 85000],
        'portfolio_returns': [-0.05, -0.05, -0.055]
    }
    
    print("\n=== Testing Circuit Breakers ===")
    print("Sample data: Portfolio down 15%, AAPL down 25%")
    
    # Check circuit breakers
    triggered = manager.check_circuit_breakers(sample_data)
    print(f"Triggered breakers: {triggered}")
    
    # Get status
    status = manager.get_status()
    print(f"\nTrading enabled: {manager.is_trading_allowed()}")
    
    for name, breaker_status in status.items():
        print(f"\n{name}:")
        print(f"  State: {breaker_status.state.value}")
        print(f"  Current: {breaker_status.current_value:.2%}")
        print(f"  Threshold: {breaker_status.threshold:.2%}")
        print(f"  Trigger count: {breaker_status.trigger_count}")
    
    # Test recovery
    print("\n=== Testing Recovery ===")
    recovery_data = {
        'portfolio_value': 95000,  # Recovered to 5% loss
        'initial_portfolio_value': 100000,
        'positions': {
            'AAPL': {'current_value': 38000, 'entry_value': 40000},  # Down 5%
            'MSFT': {'current_value': 27000, 'entry_value': 25000},
            'GOOGL': {'current_value': 30000, 'entry_value': 35000}
        },
        'total_position_value': 95000,
        'portfolio_history': [100000, 95000, 90000, 85000, 90000, 95000],
        'portfolio_returns': [-0.05, -0.05, -0.055, 0.059, 0.056]
    }
    
    # Wait a moment and check again
    import time
    time.sleep(2)
    
    triggered = manager.check_circuit_breakers(recovery_data)
    print(f"Triggered breakers after recovery: {triggered}")
    print(f"Trading enabled: {manager.is_trading_allowed()}")