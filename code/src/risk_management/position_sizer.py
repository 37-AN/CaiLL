"""
Position Sizing System - AI Trading System

This module implements various position sizing algorithms to determine
optimal trade sizes based on risk tolerance, account size, and market conditions.

Educational Note:
Position sizing is one of the most critical aspects of trading success.
Even the best strategy will fail with improper position sizing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class PositionSizingMethod(Enum):
    """Different position sizing approaches"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"
    FIXED_AMOUNT = "fixed_amount"
    PERCENT_VOLATILITY = "percent_volatility"


@dataclass
class PositionSize:
    """Result of position sizing calculation"""
    shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    method_used: str
    confidence: float
    metadata: Dict


@dataclass
class TradeParameters:
    """Parameters for position sizing calculation"""
    symbol: str
    entry_price: float
    stop_loss: float
    account_value: float
    existing_positions: Dict[str, float]
    volatility: float
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    max_position_pct: float = 0.25
    min_position_pct: float = 0.01


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms"""
    
    @abstractmethod
    def calculate_position_size(self, params: TradeParameters) -> PositionSize:
        """Calculate optimal position size"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this position sizing method"""
        pass


class FixedFractionalSizer(PositionSizer):
    """
    Fixed Fractional Position Sizing
    
    Educational Note:
    Fixed fractional sizing risks a fixed percentage of capital on each trade.
    This is the most common and recommended approach for most traders.
    
    Formula: Position Size = (Account Value × Risk %) / (Entry Price - Stop Loss)
    """
    
    def __init__(self, risk_per_trade: float = 0.02):
        self.risk_per_trade = risk_per_trade  # 2% risk per trade by default
    
    def calculate_position_size(self, params: TradeParameters) -> PositionSize:
        """Calculate position size using fixed fractional method"""
        
        # Calculate risk amount
        risk_amount = params.account_value * self.risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(params.entry_price - params.stop_loss)
        
        if risk_per_share == 0:
            return PositionSize(
                shares=0,
                position_value=0,
                risk_amount=0,
                risk_percent=0,
                method_used=self.get_method_name(),
                confidence=0.0,
                metadata={'error': 'No stop loss provided'}
            )
        
        # Calculate number of shares
        shares = int(risk_amount / risk_per_share)
        position_value = shares * params.entry_price
        
        # Apply position limits
        max_position_value = params.account_value * params.max_position_pct
        min_position_value = params.account_value * params.min_position_pct
        
        if position_value > max_position_value:
            shares = int(max_position_value / params.entry_price)
            position_value = shares * params.entry_price
        elif position_value < min_position_value and position_value > 0:
            shares = int(min_position_value / params.entry_price)
            position_value = shares * params.entry_price
        
        actual_risk = shares * risk_per_share
        actual_risk_percent = actual_risk / params.account_value
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=actual_risk_percent,
            method_used=self.get_method_name(),
            confidence=0.8,
            metadata={
                'risk_per_trade': self.risk_per_trade,
                'risk_per_share': risk_per_share,
                'max_position_pct': params.max_position_pct
            }
        )
    
    def get_method_name(self) -> str:
        return "Fixed Fractional"


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion Position Sizing
    
    Educational Note:
    The Kelly Criterion maximizes long-term wealth growth.
    It's mathematically optimal but can be aggressive.
    Many traders use fractional Kelly (1/2 or 1/4 Kelly) for safety.
    
    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of capital to wager
    - b = odds received (win/loss ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction  # Use 1/4 Kelly for safety
    
    def calculate_position_size(self, params: TradeParameters) -> PositionSize:
        """Calculate position size using Kelly Criterion"""
        
        if not all([params.win_rate, params.avg_win, params.avg_loss]):
            # Fall back to fixed fractional if Kelly parameters not available
            fallback = FixedFractionalSizer(0.02)
            result = fallback.calculate_position_size(params)
            result.method_used = f"{self.get_method_name()} (Fallback)"
            return result
        
        # Calculate Kelly percentage
        win_rate = params.win_rate
        loss_rate = 1 - win_rate
        win_loss_ratio = params.avg_win / params.avg_loss if params.avg_loss > 0 else 1
        
        # Kelly formula: f* = (bp - q) / b
        kelly_pct = (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        kelly_pct *= self.kelly_fraction
        
        # Ensure positive Kelly (no bet if negative expectation)
        kelly_pct = max(0, kelly_pct)
        
        # Calculate position size
        risk_amount = params.account_value * kelly_pct
        risk_per_share = abs(params.entry_price - params.stop_loss)
        
        if risk_per_share == 0:
            return PositionSize(
                shares=0,
                position_value=0,
                risk_amount=0,
                risk_percent=0,
                method_used=self.get_method_name(),
                confidence=0.0,
                metadata={'error': 'No stop loss provided'}
            )
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * params.entry_price
        
        # Apply position limits
        max_position_value = params.account_value * params.max_position_pct
        if position_value > max_position_value:
            shares = int(max_position_value / params.entry_price)
            position_value = shares * params.entry_price
        
        actual_risk = shares * risk_per_share
        actual_risk_percent = actual_risk / params.account_value
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=actual_risk_percent,
            method_used=self.get_method_name(),
            confidence=0.9 if kelly_pct > 0 else 0.0,
            metadata={
                'kelly_pct': kelly_pct,
                'kelly_fraction': self.kelly_fraction,
                'win_rate': win_rate,
                'win_loss_ratio': win_loss_ratio
            }
        )
    
    def get_method_name(self) -> str:
        return "Kelly Criterion"


class VolatilityTargetSizer(PositionSizer):
    """
    Volatility Target Position Sizing
    
    Educational Note:
    This method adjusts position size based on asset volatility.
    Higher volatility = smaller position, lower volatility = larger position.
    This creates a more consistent portfolio volatility profile.
    
    Formula: Position Size = Target Volatility / Asset Volatility
    """
    
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility  # 15% annual volatility target
    
    def calculate_position_size(self, params: TradeParameters) -> PositionSize:
        """Calculate position size using volatility targeting"""
        
        # Normalize volatility to annual if needed
        if params.volatility < 1:  # Assuming daily volatility if < 1
            annual_volatility = params.volatility * np.sqrt(252)
        else:
            annual_volatility = params.volatility
        
        if annual_volatility == 0:
            return PositionSize(
                shares=0,
                position_value=0,
                risk_amount=0,
                risk_percent=0,
                method_used=self.get_method_name(),
                confidence=0.0,
                metadata={'error': 'Zero volatility'}
            )
        
        # Calculate volatility scaling factor
        vol_scale = self.target_volatility / annual_volatility
        
        # Calculate position value
        position_value = params.account_value * vol_scale
        
        # Apply position limits
        max_position_value = params.account_value * params.max_position_pct
        min_position_value = params.account_value * params.min_position_pct
        
        position_value = np.clip(position_value, min_position_value, max_position_value)
        
        # Calculate shares
        shares = int(position_value / params.entry_price)
        actual_position_value = shares * params.entry_price
        
        # Estimate risk using 2 standard deviations
        risk_amount = actual_position_value * (2 * annual_volatility)
        risk_percent = risk_amount / params.account_value
        
        return PositionSize(
            shares=shares,
            position_value=actual_position_value,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            method_used=self.get_method_name(),
            confidence=0.7,
            metadata={
                'target_volatility': self.target_volatility,
                'asset_volatility': annual_volatility,
                'vol_scale': vol_scale
            }
        )
    
    def get_method_name(self) -> str:
        return "Volatility Target"


class RiskParitySizer(PositionSizer):
    """
    Risk Parity Position Sizing
    
    Educational Note:
    Risk parity allocates capital so that each position contributes
    equally to portfolio risk. This creates a more balanced risk profile.
    """
    
    def __init__(self, risk_budget: float = 0.1):
        self.risk_budget = risk_budget  # 10% risk budget per position
    
    def calculate_position_size(self, params: TradeParameters) -> PositionSize:
        """Calculate position size using risk parity"""
        
        # Calculate portfolio risk contribution target
        target_risk_contribution = params.account_value * self.risk_budget
        
        # Estimate position risk using volatility
        if params.volatility < 1:
            annual_volatility = params.volatility * np.sqrt(252)
        else:
            annual_volatility = params.volatility
        
        # Calculate position size for equal risk contribution
        position_value = target_risk_contribution / annual_volatility
        
        # Apply position limits
        max_position_value = params.account_value * params.max_position_pct
        min_position_value = params.account_value * params.min_position_pct
        
        position_value = np.clip(position_value, min_position_value, max_position_value)
        
        # Calculate shares
        shares = int(position_value / params.entry_price)
        actual_position_value = shares * params.entry_price
        
        # Calculate actual risk
        risk_amount = actual_position_value * annual_volatility
        risk_percent = risk_amount / params.account_value
        
        return PositionSize(
            shares=shares,
            position_value=actual_position_value,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            method_used=self.get_method_name(),
            confidence=0.75,
            metadata={
                'risk_budget': self.risk_budget,
                'target_risk_contribution': target_risk_contribution,
                'annual_volatility': annual_volatility
            }
        )
    
    def get_method_name(self) -> str:
        return "Risk Parity"


class PositionSizingManager:
    """
    Position Sizing Manager
    
    Educational Note:
    This manager coordinates multiple position sizing methods and can
    switch between them based on market conditions or strategy performance.
    """
    
    def __init__(self):
        self.sizers: Dict[PositionSizingMethod, PositionSizer] = {
            PositionSizingMethod.FIXED_FRACTIONAL: FixedFractionalSizer(),
            PositionSizingMethod.KELLY_CRITERION: KellyCriterionSizer(),
            PositionSizingMethod.VOLATILITY_TARGET: VolatilityTargetSizer(),
            PositionSizingMethod.RISK_PARITY: RiskParitySizer()
        }
        self.default_method = PositionSizingMethod.FIXED_FRACTIONAL
        self.performance_history: List[Dict] = []
    
    def calculate_position_size(
        self,
        params: TradeParameters,
        method: Optional[PositionSizingMethod] = None
    ) -> PositionSize:
        """Calculate position size using specified or default method"""
        
        if method is None:
            method = self.default_method
        
        sizer = self.sizers.get(method)
        if not sizer:
            raise ValueError(f"Unknown position sizing method: {method}")
        
        return sizer.calculate_position_size(params)
    
    def get_consensus_size(
        self,
        params: TradeParameters,
        methods: Optional[List[PositionSizingMethod]] = None
    ) -> PositionSize:
        """Get consensus position size from multiple methods"""
        
        if methods is None:
            methods = [
                PositionSizingMethod.FIXED_FRACTIONAL,
                PositionSizingMethod.VOLATILITY_TARGET,
                PositionSizingMethod.RISK_PARITY
            ]
        
        results = []
        for method in methods:
            try:
                result = self.calculate_position_size(params, method)
                results.append(result)
            except Exception as e:
                print(f"Error calculating position size with {method}: {e}")
        
        if not results:
            return PositionSize(
                shares=0,
                position_value=0,
                risk_amount=0,
                risk_percent=0,
                method_used="Consensus Failed",
                confidence=0.0,
                metadata={'error': 'All methods failed'}
            )
        
        # Calculate weighted average based on confidence
        total_confidence = sum(r.confidence for r in results)
        if total_confidence == 0:
            # Use simple average if no confidence
            avg_shares = int(np.mean([r.shares for r in results]))
            avg_position_value = np.mean([r.position_value for r in results])
            avg_risk_amount = np.mean([r.risk_amount for r in results])
            avg_risk_percent = np.mean([r.risk_percent for r in results])
        else:
            # Weighted average by confidence
            weights = [r.confidence / total_confidence for r in results]
            avg_shares = int(np.average([r.shares for r in results], weights=weights))
            avg_position_value = np.average([r.position_value for r in results], weights=weights)
            avg_risk_amount = np.average([r.risk_amount for r in results], weights=weights)
            avg_risk_percent = np.average([r.risk_percent for r in results], weights=weights)
        
        return PositionSize(
            shares=avg_shares,
            position_value=avg_position_value,
            risk_amount=avg_risk_amount,
            risk_percent=avg_risk_percent,
            method_used="Consensus",
            confidence=total_confidence / len(results),
            metadata={
                'methods_used': [r.method_used for r in results],
                'individual_results': [
                    {
                        'method': r.method_used,
                        'shares': r.shares,
                        'confidence': r.confidence
                    } for r in results
                ]
            }
        )
    
    def update_method_performance(self, method: PositionSizingMethod, performance: Dict):
        """Update performance tracking for position sizing methods"""
        
        self.performance_history.append({
            'method': method.value,
            'timestamp': pd.Timestamp.now(),
            'performance': performance
        })
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_best_method(self, lookback_days: int = 30) -> PositionSizingMethod:
        """Get best performing position sizing method"""
        
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent_performance = [
            p for p in self.performance_history
            if p['timestamp'] > cutoff_date
        ]
        
        if not recent_performance:
            return self.default_method
        
        # Calculate average performance by method
        method_performance = {}
        for record in recent_performance:
            method = record['method']
            perf = record['performance'].get('sharpe_ratio', 0)
            
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(perf)
        
        # Find best method
        best_method = self.default_method
        best_avg_perf = float('-inf')
        
        for method, performances in method_performance.items():
            avg_perf = np.mean(performances)
            if avg_perf > best_avg_perf:
                best_avg_perf = avg_perf
                best_method = PositionSizingMethod(method)
        
        return best_method


# Educational helper functions
def explain_position_sizing_concepts():
    """
    Educational explanation of position sizing concepts
    """
    
    concepts = {
        'Position Sizing': "The process of determining how many shares or contracts to trade based on risk tolerance and account size.",
        
        'Risk per Trade': "The maximum amount of money you're willing to lose on a single trade, typically 1-3% of account value.",
        
        'Fixed Fractional': "Risk a fixed percentage of capital on each trade. Simple, conservative, and widely recommended.",
        
        'Kelly Criterion': "Mathematically optimal position sizing for maximum long-term growth. Can be aggressive, so fractional Kelly is often used.",
        
        'Volatility Target': "Adjust position size based on asset volatility to maintain consistent portfolio risk.",
        
        'Risk Parity': "Allocate capital so each position contributes equally to portfolio risk.",
        
        'Maximum Drawdown': "The largest peak-to-trough decline in portfolio value. Position sizing helps control this.",
        
        'Correlation': "How different assets move in relation to each other. Important for portfolio-level position sizing."
    }
    
    print("=== Position Sizing Educational Guide ===\n")
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Key Principles ===")
    print("1. Never risk more than you can afford to lose")
    print("2. Consistent position sizing is more important than perfect entry timing")
    print("3. Lower volatility = larger position, Higher volatility = smaller position")
    print("4. Always consider portfolio-level risk, not just individual trades")
    print("5. Backtest position sizing strategies before implementation")


if __name__ == "__main__":
    # Example usage
    explain_position_sizing_concepts()
    
    # Create sample trade parameters
    params = TradeParameters(
        symbol="AAPL",
        entry_price=150.0,
        stop_loss=145.0,
        account_value=100000,
        existing_positions={},
        volatility=0.02,
        win_rate=0.55,
        avg_win=100,
        avg_loss=80
    )
    
    # Test different position sizing methods
    manager = PositionSizingManager()
    
    print("\n=== Position Sizing Comparison ===")
    for method in PositionSizingMethod:
        try:
            result = manager.calculate_position_size(params, method)
            print(f"\n{result.method_used}:")
            print(f"  Shares: {result.shares}")
            print(f"  Position Value: ${result.position_value:,.2f}")
            print(f"  Risk Amount: ${result.risk_amount:,.2f}")
            print(f"  Risk %: {result.risk_percent:.2%}")
            print(f"  Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"Error with {method.value}: {e}")
    
    # Test consensus
    consensus = manager.get_consensus_size(params)
    print(f"\nConsensus Result:")
    print(f"  Shares: {consensus.shares}")
    print(f"  Position Value: ${consensus.position_value:,.2f}")
    print(f"  Risk %: {consensus.risk_percent:.2%}")
    print(f"  Confidence: {consensus.confidence:.2f}")