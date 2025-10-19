"""
Risk Calculator - AI Trading System

This module implements comprehensive risk calculation methods including
VaR, CVaR, maximum drawdown, correlation analysis, and portfolio risk metrics.

Educational Note:
Understanding and quantifying risk is the foundation of successful trading.
This module provides the mathematical tools to measure and manage risk effectively.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RiskMetricType(Enum):
    """Different types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics"""
    var_1d: float
    var_5d: float
    var_30d: float
    cvar_1d: float
    cvar_5d: float
    cvar_30d: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    beta: Optional[float]
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float]
    correlation_matrix: Optional[pd.DataFrame]
    beta_to_market: Optional[float]
    tracking_error: Optional[float]
    downside_deviation: float
    upside_capture: float
    downside_capture: float
    var_confidence: float = 0.95
    calculation_date: datetime = None
    
    def __post_init__(self):
        if self.calculation_date is None:
            self.calculation_date = datetime.now()


@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_value: float
    weight: float
    var_contribution: float
    cvar_contribution: float
    marginal_var: float
    component_var: float
    beta: float
    volatility: float
    correlation_to_portfolio: float
    expected_shortfall: float


class RiskCalculator:
    """
    Comprehensive Risk Calculator
    
    Educational Note:
    This calculator implements industry-standard risk metrics used by
    hedge funds and institutional traders. Each metric provides different
    insights into portfolio risk characteristics.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Educational Note:
        VaR answers: "What is the maximum loss I can expect with X% confidence
        over a given time horizon?" It's a standard risk measure in finance.
        
        Methods:
        - Historical: Uses actual historical return distribution
        - Parametric: Assumes normal distribution
        - Monte Carlo: Simulates many scenarios
        """
        
        if confidence is None:
            confidence = self.confidence_level
            
        if method == "historical":
            return self._historical_var(returns, confidence)
        elif method == "parametric":
            return self._parametric_var(returns, confidence)
        elif method == "monte_carlo":
            return self._monte_carlo_var(returns, confidence)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Historical VaR - uses actual historical distribution"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Parametric VaR - assumes normal distribution"""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mean + z_score * std
    
    def _monte_carlo_var(self, returns: pd.Series, confidence: float, n_simulations: int = 10000) -> float:
        """Monte Carlo VaR - simulates many scenarios"""
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mean, std, n_simulations)
        return np.percentile(simulated_returns, (1 - confidence) * 100)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
        method: str = "historical"
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        
        Educational Note:
        CVaR (also called Expected Shortfall) answers: "If I exceed my VaR,
        what is my average expected loss?" It addresses VaR's limitation of
        not telling you how bad losses can be beyond the VaR threshold.
        """
        
        if confidence is None:
            confidence = self.confidence_level
            
        var = self.calculate_var(returns, confidence, method)
        
        if method == "historical":
            # Average of returns that are worse than VaR
            tail_losses = returns[returns <= var]
            return tail_losses.mean() if len(tail_losses) > 0 else var
        else:
            # For parametric methods, use analytical formula
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            phi_z = stats.norm.pdf(z_score)
            return mean - std * (phi_z / (1 - confidence))
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """
        Calculate Maximum Drawdown and Duration
        
        Educational Note:
        Maximum drawdown measures the largest peak-to-trough decline.
        Duration measures how long it took to recover. This is crucial for
        understanding psychological and capital recovery challenges.
        """
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find duration of maximum drawdown
        max_dd_end = drawdown.idxmin()
        max_dd_start = equity_curve[:max_dd_end].idxmax()
        
        # Calculate recovery duration
        recovery_mask = equity_curve[max_dd_end:] >= equity_curve[max_dd_start]
        if recovery_mask.any():
            recovery_date = equity_curve[max_dd_end:][recovery_mask].index[0]
            duration = (recovery_date - max_dd_start).days
        else:
            duration = (equity_curve.index[-1] - max_dd_start).days
        
        return max_dd, duration
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        
        Educational Note:
        Volatility is the most basic risk measure. Higher volatility
        means larger price swings and potentially higher risk.
        """
        
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Assuming daily returns
        return vol
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (systematic risk)
        
        Educational Note:
        Beta measures an asset's sensitivity to market movements.
        Beta = 1: moves with market
        Beta > 1: more volatile than market
        Beta < 1: less volatile than market
        Beta < 0: moves opposite to market
        """
        
        # Align the series
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0  # Default beta
        
        asset = aligned_data.iloc[:, 0]
        market = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(asset, market)[0, 1]
        market_variance = np.var(market)
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio
        
        Educational Note:
        Sharpe Ratio measures risk-adjusted returns.
        Higher is better. Values > 1 are considered good,
        > 2 are very good, > 3 are excellent.
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / 252  # Daily adjustment
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        
        if annualize:
            sharpe *= np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio
        
        Educational Note:
        Sortino Ratio is similar to Sharpe but only penalizes
        downside volatility, making it more suitable for asymmetric
        return distributions.
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns - risk_free_rate / 252
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_deviation
        
        if annualize:
            sortino *= np.sqrt(252)
        
        return sortino
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        equity_curve: pd.Series
    ) -> float:
        """
        Calculate Calmar Ratio
        
        Educational Note:
        Calmar Ratio = Annual Return / Maximum Drawdown
        It measures return relative to worst-case loss experience.
        Higher values indicate better risk-adjusted performance.
        """
        
        # Calculate annualized return
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # Calculate maximum drawdown
        max_dd, _ = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_portfolio_risk(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        market_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Educational Note:
        This provides a complete risk profile of the portfolio,
        including absolute and relative risk measures.
        """
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        equity_curve = (1 + portfolio_returns).cumprod()
        
        # Calculate VaR at different horizons
        var_1d = self.calculate_var(portfolio_returns, self.confidence_level)
        var_5d = self.calculate_var(portfolio_returns, self.confidence_level) * np.sqrt(5)
        var_30d = self.calculate_var(portfolio_returns, self.confidence_level) * np.sqrt(30)
        
        # Calculate CVaR
        cvar_1d = self.calculate_cvar(portfolio_returns, self.confidence_level)
        cvar_5d = self.calculate_cvar(portfolio_returns, self.confidence_level) * np.sqrt(5)
        cvar_30d = self.calculate_cvar(portfolio_returns, self.confidence_level) * np.sqrt(30)
        
        # Calculate maximum drawdown
        max_dd, dd_duration = self.calculate_max_drawdown(equity_curve)
        
        # Calculate volatility
        volatility = self.calculate_volatility(portfolio_returns)
        
        # Calculate risk ratios
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        sortino = self.calculate_sortino_ratio(portfolio_returns)
        calmar = self.calculate_calmar_ratio(portfolio_returns, equity_curve)
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Calculate beta and information ratio if market returns provided
        beta = None
        information_ratio = None
        tracking_error = None
        
        if market_returns is not None:
            beta = self.calculate_beta(portfolio_returns, market_returns)
            excess_returns = portfolio_returns - market_returns
            if excess_returns.std() > 0:
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Calculate additional metrics
        downside_deviation = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        
        # Upside/Downside capture (requires market returns)
        upside_capture = 1.0
        downside_capture = 1.0
        
        if market_returns is not None:
            up_market = market_returns > 0
            down_market = market_returns < 0
            
            if up_market.sum() > 0:
                upside_capture = (
                    portfolio_returns[up_market].mean() / 
                    market_returns[up_market].mean()
                )
            
            if down_market.sum() > 0:
                downside_capture = (
                    portfolio_returns[down_market].mean() / 
                    market_returns[down_market].mean()
                )
        
        return RiskMetrics(
            var_1d=var_1d,
            var_5d=var_5d,
            var_30d=var_30d,
            cvar_1d=cvar_1d,
            cvar_5d=cvar_5d,
            cvar_30d=cvar_30d,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            correlation_matrix=correlation_matrix,
            beta_to_market=beta,
            tracking_error=tracking_error,
            downside_deviation=downside_deviation,
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            var_confidence=self.confidence_level
        )
    
    def calculate_position_risk_contributions(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_var: float
    ) -> List[PositionRisk]:
        """
        Calculate risk contributions of individual positions
        
        Educational Note:
        This shows how each position contributes to overall portfolio risk.
        It's essential for understanding portfolio diversification and
        identifying concentration risks.
        """
        
        n_assets = len(weights)
        covariance_matrix = returns.cov() * 252  # Annualized
        
        # Calculate marginal VaR for each position
        marginal_var = np.zeros(n_assets)
        for i in range(n_assets):
            marginal_var[i] = (weights @ covariance_matrix[i]) / np.sqrt(weights @ covariance_matrix @ weights.T)
        
        # Calculate component VaR
        component_var = weights * marginal_var
        
        # Calculate position-level metrics
        position_risks = []
        for i in range(n_assets):
            symbol = returns.columns[i]
            position_value = weights[i]  # Assuming normalized weights
            weight = weights[i]
            
            # Calculate individual asset metrics
            asset_returns = returns.iloc[:, i]
            asset_volatility = self.calculate_volatility(asset_returns)
            asset_var = self.calculate_var(asset_returns, self.confidence_level)
            asset_cvar = self.calculate_cvar(asset_returns, self.confidence_level)
            
            # Calculate correlation to portfolio
            portfolio_returns = (returns * weights).sum(axis=1)
            correlation = returns.iloc[:, i].corr(portfolio_returns)
            
            # Calculate beta (using market as proxy if available)
            beta = correlation * (asset_volatility / portfolio_returns.std() * np.sqrt(252))
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_value=position_value,
                weight=weight,
                var_contribution=component_var[i],
                cvar_contribution=component_var[i] * 1.5,  # Approximation
                marginal_var=marginal_var[i],
                component_var=component_var[i],
                beta=beta,
                volatility=asset_volatility,
                correlation_to_portfolio=correlation,
                expected_shortfall=asset_cvar
            )
            
            position_risks.append(position_risk)
        
        return position_risks
    
    def stress_test_portfolio(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        scenarios: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Perform stress testing on portfolio
        
        Educational Note:
        Stress testing shows how portfolio would perform under
        extreme market conditions. It's a crucial risk management
        tool for understanding tail risk.
        """
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        stress_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply stress scenario
            stressed_returns = portfolio_returns.copy()
            
            if 'market_shock' in scenario_params:
                shock = scenario_params['market_shock']
                stressed_returns += shock
            
            if 'volatility_multiplier' in scenario_params:
                vol_mult = scenario_params['volatility_multiplier']
                mean = stressed_returns.mean()
                stressed_returns = (stressed_returns - mean) * vol_mult + mean
            
            if 'correlation_breakdown' in scenario_params:
                # Simplified correlation stress
                correlation_stress = scenario_params['correlation_breakdown']
                stressed_returns *= correlation_stress
            
            # Calculate stressed metrics
            stressed_var = self.calculate_var(stressed_returns, self.confidence_level)
            stressed_cvar = self.calculate_cvar(stressed_returns, self.confidence_level)
            stressed_vol = self.calculate_volatility(stressed_returns)
            
            # Calculate equity curve and drawdown
            stressed_equity = (1 + stressed_returns).cumprod()
            stressed_max_dd, _ = self.calculate_max_drawdown(stressed_equity)
            
            stress_results[scenario_name] = {
                'var': stressed_var,
                'cvar': stressed_cvar,
                'volatility': stressed_vol,
                'max_drawdown': stressed_max_dd,
                'worst_return': stressed_returns.min(),
                'scenario_params': scenario_params
            }
        
        return stress_results


def explain_risk_metrics():
    """
    Educational explanation of risk metrics
    """
    
    explanations = {
        'Value at Risk (VaR)': {
            'concept': 'Maximum expected loss with given confidence over time horizon',
            'example': '1-day 95% VaR of $1,000 means 95% chance of losing less than $1,000 in one day',
            'limitations': 'Doesnt tell you how bad losses can be beyond VaR',
            'use_case': 'Setting position limits and capital requirements'
        },
        
        'Conditional VaR (CVaR)': {
            'concept': 'Average expected loss when VaR is exceeded',
            'example': 'If VaR is $1,000 and CVaR is $1,500, average loss in worst 5% is $1,500',
            'limitations': 'Requires more data and assumptions',
            'use_case': 'Stress testing and extreme risk management'
        },
        
        'Maximum Drawdown': {
            'concept': 'Largest peak-to-trough decline in portfolio value',
            'example': 'Portfolio went from $100,000 to $70,000, max drawdown is 30%',
            'limitations': 'Path-dependent, can be misleading for volatile assets',
            'use_case': 'Risk tolerance assessment and performance evaluation'
        },
        
        'Sharpe Ratio': {
            'concept': 'Risk-adjusted return measure (excess return per unit of risk)',
            'example': 'Sharpe of 2 means 2 units of return per unit of risk',
            'limitations': 'Penalizes upside and downside volatility equally',
            'use_case': 'Comparing strategies and performance evaluation'
        },
        
        'Beta': {
            'concept': 'Sensitivity to market movements',
            'example': 'Beta of 1.5 means 1.5% move for every 1% market move',
            'limitations': 'Historical measure, assumes linear relationship',
            'use_case': 'Portfolio construction and hedging'
        },
        
        'Correlation': {
            'concept': 'How assets move in relation to each other',
            'example': 'Correlation of 0.8 means strong positive relationship',
            'limitations': 'Can change during market stress',
            'use_case': 'Diversification analysis and portfolio optimization'
        }
    }
    
    print("=== Risk Metrics Educational Guide ===\n")
    
    for metric, details in explanations.items():
        print(f"{metric}:")
        print(f"  Concept: {details['concept']}")
        print(f"  Example: {details['example']}")
        print(f"  Limitations: {details['limitations']}")
        print(f"  Use Case: {details['use_case']}\n")
    
    print("=== Risk Management Best Practices ===")
    print("1. Use multiple risk metrics - no single measure tells the whole story")
    print("2. Monitor both absolute and relative risk")
    print("3. Understand the limitations of each metric")
    print("4. Combine quantitative analysis with qualitative judgment")
    print("5. Regularly stress test portfolios under extreme scenarios")
    print("6. Consider correlation breakdowns during market crises")


if __name__ == "__main__":
    # Example usage
    explain_risk_metrics()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate correlated returns for 3 assets
    n_assets = 3
    correlation_matrix = np.array([
        [1.0, 0.7, 0.3],
        [0.7, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    # Generate random returns with correlation
    mean_returns = np.array([0.0005, 0.0003, 0.0007])  # Daily means
    volatilities = np.array([0.02, 0.015, 0.025])  # Daily volatilities
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
    returns_df = pd.DataFrame(returns, index=dates, columns=['AAPL', 'MSFT', 'GOOGL'])
    
    # Create portfolio weights
    weights = np.array([0.4, 0.3, 0.3])
    
    # Create market returns (simplified)
    market_returns = pd.Series(
        np.random.normal(0.0004, 0.01, len(dates)),
        index=dates,
        name='Market'
    )
    
    # Calculate risk metrics
    calculator = RiskCalculator(confidence_level=0.95)
    risk_metrics = calculator.calculate_portfolio_risk(returns_df, weights, market_returns)
    
    print("\n=== Portfolio Risk Analysis ===")
    print(f"1-Day VaR (95%): {risk_metrics.var_1d:.2%}")
    print(f"5-Day VaR (95%): {risk_metrics.var_5d:.2%}")
    print(f"30-Day VaR (95%): {risk_metrics.var_30d:.2%}")
    print(f"1-Day CVaR (95%): {risk_metrics.cvar_1d:.2%}")
    print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"Drawdown Duration: {risk_metrics.max_drawdown_duration} days")
    print(f"Annual Volatility: {risk_metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")
    print(f"Beta to Market: {risk_metrics.beta:.2f}")
    
    # Calculate position risk contributions
    portfolio_var = calculator.calculate_var((returns_df * weights).sum(axis=1), 0.95)
    position_risks = calculator.calculate_position_risk_contributions(returns_df, weights, portfolio_var)
    
    print("\n=== Position Risk Contributions ===")
    for pos_risk in position_risks:
        print(f"\n{pos_risk.symbol}:")
        print(f"  Weight: {pos_risk.weight:.1%}")
        print(f"  VaR Contribution: {pos_risk.var_contribution:.2%}")
        print(f"  Beta: {pos_risk.beta:.2f}")
        print(f"  Volatility: {pos_risk.volatility:.2%}")
        print(f"  Correlation to Portfolio: {pos_risk.correlation_to_portfolio:.2f}")
    
    # Stress testing
    scenarios = {
        'Market Crash': {
            'market_shock': -0.05,  # 5% market drop
            'volatility_multiplier': 2.0
        },
        'Volatility Spike': {
            'volatility_multiplier': 3.0
        },
        'Correlation Breakdown': {
            'correlation_breakdown': 1.5
        }
    }
    
    stress_results = calculator.stress_test_portfolio(returns_df, weights, scenarios)
    
    print("\n=== Stress Test Results ===")
    for scenario, results in stress_results.items():
        print(f"\n{scenario}:")
        print(f"  VaR: {results['var']:.2%}")
        print(f"  CVaR: {results['cvar']:.2%}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Worst Return: {results['worst_return']:.2%}")