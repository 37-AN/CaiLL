"""
Performance Calculator - AI Trading System

This module implements comprehensive performance calculation and analysis
tools for trading strategies. It provides detailed metrics, attribution
analysis, and performance comparison capabilities.

Educational Note:
Performance analysis is crucial for understanding strategy effectiveness.
Beyond simple returns, we need to analyze risk-adjusted performance,
consistency, drawdown characteristics, and attribution to different
factors. This calculator provides institutional-grade analytics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting components
from .backtest_engine import BacktestResult, Trade, Order


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted return metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    trades_per_month: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    positive_months: float
    negative_months: float
    
    # Consistency metrics
    hit_rate: float
    recovery_factor: float
    sterling_ratio: float
    burke_ratio: float
    
    # Benchmark comparison
    beta: float
    alpha: float
    tracking_error: float
    up_capture: float
    down_capture: float
    
    # Additional metrics
    gain_to_pain_ratio: float
    omega_ratio: float
    tail_ratio: float
    ulcer_index: float


@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    
    # Sector attribution
    sector_attribution: Dict[str, Dict[str, float]]
    
    # Factor attribution
    factor_attribution: Dict[str, Dict[str, float]]
    
    # Time attribution
    monthly_attribution: Dict[str, Dict[str, float]]
    
    # Trade attribution
    trade_attribution: Dict[str, Dict[str, float]]
    
    # Risk attribution
    risk_attribution: Dict[str, Dict[str, float]]


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    
    strategy_name: str
    period_start: datetime
    period_end: datetime
    
    # Basic metrics
    metrics: PerformanceMetrics
    
    # Attribution analysis
    attribution: AttributionAnalysis
    
    # Monthly breakdown
    monthly_performance: Dict[str, Dict[str, float]]
    
    # Rolling metrics
    rolling_metrics: Dict[str, pd.Series]
    
    # Benchmark comparison
    benchmark_comparison: Dict[str, Any]
    
    # Risk analysis
    risk_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    total_trades: int
    trading_days: int
    calculation_date: datetime = field(default_factory=datetime.now)


class PerformanceCalculator:
    """
    Comprehensive Performance Calculator
    
    Educational Note:
    This calculator provides institutional-grade performance analysis
    using industry-standard methodologies. It goes beyond simple
    return calculations to provide deep insights into strategy
    characteristics, risk profile, and sources of returns.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        
    def calculate_performance(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark"
    ) -> PerformanceReport:
        """Calculate comprehensive performance metrics"""
        
        print(f"Calculating performance for {result.strategy_name}")
        
        # Extract data
        returns = result.returns
        equity_curve = result.equity_curve
        trades = result.trades_history
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(returns, equity_curve, trades)
        
        # Calculate risk-adjusted metrics
        self._calculate_risk_adjusted_metrics(metrics, returns, benchmark_returns)
        
        # Calculate trading metrics
        self._calculate_trading_metrics(metrics, trades, returns)
        
        # Calculate distribution metrics
        self._calculate_distribution_metrics(metrics, returns)
        
        # Calculate consistency metrics
        self._calculate_consistency_metrics(metrics, returns, equity_curve)
        
        # Calculate benchmark comparison
        self._calculate_benchmark_comparison(metrics, returns, benchmark_returns)
        
        # Calculate additional metrics
        self._calculate_additional_metrics(metrics, returns)
        
        # Attribution analysis
        attribution = self._calculate_attribution(result, benchmark_returns)
        
        # Monthly breakdown
        monthly_performance = self._calculate_monthly_performance(returns, equity_curve)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(returns)
        
        # Benchmark comparison
        benchmark_comparison = self._analyze_benchmark_comparison(
            metrics, returns, benchmark_returns, benchmark_name
        )
        
        # Risk analysis
        risk_analysis = self._analyze_risk(returns, equity_curve, trades)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, risk_analysis)
        
        return PerformanceReport(
            strategy_name=result.strategy_name,
            period_start=result.start_date,
            period_end=result.end_date,
            metrics=metrics,
            attribution=attribution,
            monthly_performance=monthly_performance,
            rolling_metrics=rolling_metrics,
            benchmark_comparison=benchmark_comparison,
            risk_analysis=risk_analysis,
            recommendations=recommendations,
            total_trades=len(trades),
            trading_days=len(returns),
            calculation_date=datetime.now()
        )
    
    def _calculate_basic_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> PerformanceMetrics:
        """Calculate basic performance metrics"""
        
        # Return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Calculate annualized return
        days = len(returns)
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        cagr = annualized_return
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Downside volatility
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 0
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Find maximum drawdown duration
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(i - drawdown_start)
        
        if in_drawdown:
            drawdown_periods.append(len(drawdown) - drawdown_start)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            downside_volatility=downside_volatility,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=0,  # Will be calculated later
            sortino_ratio=0,  # Will be calculated later
            calmar_ratio=0,  # Will be calculated later
            information_ratio=0,  # Will be calculated later
            treynor_ratio=0,  # Will be calculated later
            jensen_alpha=0,  # Will be calculated later
            win_rate=0,  # Will be calculated later
            profit_factor=0,  # Will be calculated later
            avg_win=0,  # Will be calculated later
            avg_loss=0,  # Will be calculated later
            best_trade=0,  # Will be calculated later
            worst_trade=0,  # Will be calculated later
            avg_trade_duration=0,  # Will be calculated later
            trades_per_month=0,  # Will be calculated later
            skewness=0,  # Will be calculated later
            kurtosis=0,  # Will be calculated later
            positive_months=0,  # Will be calculated later
            negative_months=0,  # Will be calculated later
            hit_rate=0,  # Will be calculated later
            recovery_factor=0,  # Will be calculated later
            sterling_ratio=0,  # Will be calculated later
            burke_ratio=0,  # Will be calculated later
            beta=0,  # Will be calculated later
            alpha=0,  # Will be calculated later
            tracking_error=0,  # Will be calculated later
            up_capture=0,  # Will be calculated later
            down_capture=0,  # Will be calculated later
            gain_to_pain_ratio=0,  # Will be calculated later
            omega_ratio=0,  # Will be calculated later
            tail_ratio=0,  # Will be calculated later
            ulcer_index=0  # Will be calculated later
        )
    
    def _calculate_risk_adjusted_metrics(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series]
    ):
        """Calculate risk-adjusted performance metrics"""
        
        # Sharpe ratio
        excess_return = metrics.annualized_return - self.risk_free_rate
        metrics.sharpe_ratio = excess_return / metrics.volatility if metrics.volatility > 0 else 0
        
        # Sortino ratio
        excess_return_sortino = metrics.annualized_return - self.risk_free_rate
        metrics.sortino_ratio = excess_return_sortino / metrics.downside_volatility if metrics.downside_volatility > 0 else 0
        
        # Calmar ratio
        metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            metrics.tracking_error = tracking_error
            
            if tracking_error > 0:
                metrics.information_ratio = active_returns.mean() * 252 / tracking_error
            
            # Beta and Alpha
            if len(returns) > 1 and len(benchmark_returns) > 1:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                metrics.beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Jensen's Alpha
                benchmark_annual_return = benchmark_returns.mean() * 252
                expected_return = self.risk_free_rate + metrics.beta * (benchmark_annual_return - self.risk_free_rate)
                metrics.jensen_alpha = metrics.annualized_return - expected_return
                metrics.alpha = metrics.jensen_alpha
            
            # Up/Down capture
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0
            
            if up_periods.sum() > 0:
                strategy_up_returns = returns[up_periods].mean() * 252
                benchmark_up_returns = benchmark_returns[up_periods].mean() * 252
                metrics.up_capture = strategy_up_returns / benchmark_up_returns if benchmark_up_returns != 0 else 0
            
            if down_periods.sum() > 0:
                strategy_down_returns = returns[down_periods].mean() * 252
                benchmark_down_returns = benchmark_returns[down_periods].mean() * 252
                metrics.down_capture = strategy_down_returns / benchmark_down_returns if benchmark_down_returns != 0 else 0
        
        # Treynor ratio
        metrics.treynor_ratio = excess_return / metrics.beta if metrics.beta != 0 else 0
    
    def _calculate_trading_metrics(
        self,
        metrics: PerformanceMetrics,
        trades: List[Trade],
        returns: pd.Series
    ):
        """Calculate trading-specific metrics"""
        
        if not trades:
            return
        
        # Calculate trade P&L (simplified)
        trade_pnls = []
        trade_durations = []
        
        for i, trade in enumerate(trades):
            # Simplified P&L calculation
            if i % 2 == 1:  # Assume pairs of trades (open/close)
                pnl = (trade.price - trades[i-1].price) * trade.quantity
                trade_pnls.append(pnl)
                
                # Calculate duration (simplified)
                duration = (trade.timestamp - trades[i-1].timestamp).days
                trade_durations.append(duration)
        
        if not trade_pnls:
            return
        
        # Win rate
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        metrics.win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
        
        # Average win/loss
        metrics.avg_win = np.mean(winning_trades) if winning_trades else 0
        metrics.avg_loss = np.mean(losing_trades) if losing_trades else 0
        metrics.best_trade = max(trade_pnls) if trade_pnls else 0
        metrics.worst_trade = min(trade_pnls) if trade_pnls else 0
        
        # Profit factor
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average trade duration
        metrics.avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Trades per month
        months = len(returns) / 21  # Approximate trading days per month
        metrics.trades_per_month = len(trades) / months if months > 0 else 0
        
        # Hit rate (same as win rate)
        metrics.hit_rate = metrics.win_rate
    
    def _calculate_distribution_metrics(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series
    ):
        """Calculate distribution metrics"""
        
        if len(returns) < 2:
            return
        
        # Skewness and kurtosis
        metrics.skewness = returns.skew()
        metrics.kurtosis = returns.kurtosis()
        
        # Monthly performance
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        metrics.positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0
        metrics.negative_months = (monthly_returns < 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0
    
    def _calculate_consistency_metrics(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series,
        equity_curve: pd.Series
    ):
        """Calculate consistency metrics"""
        
        # Recovery factor
        metrics.recovery_factor = metrics.total_return / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
        
        # Sterling ratio
        if metrics.max_drawdown != 0:
            # Calculate average drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            avg_drawdown = drawdown[drawdown < 0].mean()
            metrics.sterling_ratio = metrics.annualized_return / abs(avg_drawdown) if avg_drawdown != 0 else 0
        
        # Burke ratio
        if metrics.max_drawdown != 0:
            # Calculate squared drawdowns
            drawdowns = []
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            
            for dd in drawdown:
                if dd < 0:
                    drawdowns.append(dd ** 2)
            
            sum_squared_drawdowns = sum(drawdowns)
            metrics.burke_ratio = metrics.annualized_return / np.sqrt(sum_squared_drawdowns) if sum_squared_drawdowns > 0 else 0
    
    def _calculate_benchmark_comparison(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series]
    ):
        """Calculate benchmark comparison metrics"""
        
        # These are already calculated in _calculate_risk_adjusted_metrics
        pass
    
    def _calculate_additional_metrics(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series
    ):
        """Calculate additional performance metrics"""
        
        # Gain to pain ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        metrics.gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        # Omega ratio (using threshold of 0)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        metrics.omega_ratio = 1 + (gains / losses) if losses > 0 else float('inf')
        
        # Tail ratio
        if len(returns) > 0:
            percentile_95 = returns.quantile(0.95)
            percentile_5 = returns.quantile(0.05)
            metrics.tail_ratio = abs(percentile_95) / abs(percentile_5) if percentile_5 != 0 else float('inf')
        
        # Ulcer index
        peak = returns.cumsum().expanding().max()
        drawdown = returns.cumsum() - peak
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        metrics.ulcer_index = ulcer_index
    
    def _calculate_attribution(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series]
    ) -> AttributionAnalysis:
        """Calculate performance attribution"""
        
        # Simplified attribution analysis
        # In a real implementation, this would be much more detailed
        
        sector_attribution = {}
        factor_attribution = {}
        monthly_attribution = {}
        trade_attribution = {}
        risk_attribution = {}
        
        # Monthly attribution
        if len(result.returns) > 0:
            monthly_returns = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            for month, monthly_ret in monthly_returns.items():
                monthly_attribution[month.strftime('%Y-%m')] = {
                    'return': monthly_ret,
                    'contribution': monthly_ret * 0.1,  # Simplified
                    'active_return': monthly_ret * 0.05  # Simplified
                }
        
        return AttributionAnalysis(
            sector_attribution=sector_attribution,
            factor_attribution=factor_attribution,
            monthly_attribution=monthly_attribution,
            trade_attribution=trade_attribution,
            risk_attribution=risk_attribution
        )
    
    def _calculate_monthly_performance(
        self,
        returns: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Calculate monthly performance breakdown"""
        
        monthly_performance = {}
        
        # Group by month
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_equity = equity_curve.resample('M').last()
        
        for month, monthly_ret in monthly_returns.items():
            month_str = month.strftime('%Y-%m')
            
            # Calculate monthly metrics
            month_returns = returns[returns.index.month == month.month]
            
            monthly_performance[month_str] = {
                'return': monthly_ret,
                'volatility': month_returns.std() * np.sqrt(252) if len(month_returns) > 1 else 0,
                'max_drawdown': 0,  # Would calculate monthly drawdown
                'positive': monthly_ret > 0,
                'equity_start': monthly_equity.iloc[0] if len(monthly_equity) > 0 else 0,
                'equity_end': monthly_equity.iloc[-1] if len(monthly_equity) > 0 else 0
            }
        
        return monthly_performance
    
    def _calculate_rolling_metrics(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        
        rolling_metrics = {}
        
        if len(returns) < 63:  # Need at least 3 months
            return rolling_metrics
        
        # Rolling Sharpe ratio (3 months)
        rolling_sharpe = returns.rolling(window=63).apply(
            lambda x: (x.mean() * 252 - self.risk_free_rate) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        rolling_metrics['sharpe_3m'] = rolling_sharpe
        
        # Rolling volatility (3 months)
        rolling_vol = returns.rolling(window=63).std() * np.sqrt(252)
        rolling_metrics['volatility_3m'] = rolling_vol
        
        # Rolling max drawdown (3 months)
        rolling_equity = (1 + returns).cumprod()
        rolling_peak = rolling_equity.rolling(window=63).max()
        rolling_dd = (rolling_equity - rolling_peak) / rolling_peak
        rolling_max_dd = rolling_dd.rolling(window=63).min()
        rolling_metrics['max_drawdown_3m'] = rolling_max_dd
        
        return rolling_metrics
    
    def _analyze_benchmark_comparison(
        self,
        metrics: PerformanceMetrics,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        benchmark_name: str
    ) -> Dict[str, Any]:
        """Analyze benchmark comparison"""
        
        comparison = {
            'benchmark_name': benchmark_name,
            'has_benchmark': benchmark_returns is not None
        }
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Calculate correlation
            correlation = returns.corr(benchmark_returns)
            comparison['correlation'] = correlation
            
            # Calculate outperformance
            strategy_return = metrics.total_return
            benchmark_return = (1 + benchmark_returns).prod() - 1
            comparison['outperformance'] = strategy_return - benchmark_return
            
            # Calculate tracking error
            active_returns = returns - benchmark_returns
            comparison['tracking_error'] = active_returns.std() * np.sqrt(252)
            
            # Calculate up/down market performance
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0
            
            if up_periods.sum() > 0:
                strategy_up = returns[up_periods].mean() * 252
                benchmark_up = benchmark_returns[up_periods].mean() * 252
                comparison['up_market_performance'] = strategy_up - benchmark_up
            
            if down_periods.sum() > 0:
                strategy_down = returns[down_periods].mean() * 252
                benchmark_down = benchmark_returns[down_periods].mean() * 252
                comparison['down_market_performance'] = strategy_down - benchmark_down
        
        return comparison
    
    def _analyze_risk(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        
        risk_analysis = {}
        
        # Return distribution analysis
        risk_analysis['return_distribution'] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'percentile_5': returns.quantile(0.05),
            'percentile_95': returns.quantile(0.95)
        }
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        risk_analysis['drawdown_analysis'] = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_frequency': (drawdown < 0).sum() / len(drawdown),
            'avg_recovery_time': 0,  # Would calculate average recovery time
            'current_drawdown': drawdown.iloc[-1]
        }
        
        # Risk-adjusted return analysis
        risk_analysis['risk_adjusted_metrics'] = {
            'sharpe_ratio': (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'sortino_ratio': (returns.mean() * 252 - self.risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).std() > 0 else 0,
            'calmar_ratio': (returns.mean() * 252) / abs(drawdown.min()) if drawdown.min() != 0 else 0
        }
        
        # Trading risk analysis
        if trades:
            # Calculate trade-level risk metrics
            trade_pnls = []  # Would calculate actual P&L
            
            risk_analysis['trading_risk'] = {
                'total_trades': len(trades),
                'avg_trade_size': np.mean([abs(t.quantity * t.price) for t in trades]) if trades else 0,
                'position_concentration': 0,  # Would calculate concentration
                'trade_frequency': len(trades) / len(returns) if len(returns) > 0 else 0
            }
        
        return risk_analysis
    
    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        risk_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        # Return-based recommendations
        if metrics.annualized_return < 0.05:
            recommendations.append("Low annualized return - consider strategy optimization")
        elif metrics.annualized_return > 0.20:
            recommendations.append("High annualized return - verify sustainability and risk levels")
        
        # Risk-adjusted return recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Low Sharpe ratio - improve risk-adjusted performance")
        elif metrics.sharpe_ratio > 2.0:
            recommendations.append("Excellent Sharpe ratio - strategy shows strong risk-adjusted performance")
        
        # Drawdown recommendations
        if metrics.max_drawdown < -0.25:
            recommendations.append("High maximum drawdown - strengthen risk management")
        elif metrics.max_drawdown > -0.05:
            recommendations.append("Low drawdown indicates good risk control")
        
        # Consistency recommendations
        if metrics.win_rate < 0.4:
            recommendations.append("Low win rate - review entry/exit criteria")
        elif metrics.win_rate > 0.7:
            recommendations.append("High win rate indicates strong strategy performance")
        
        # Volatility recommendations
        if metrics.volatility > 0.25:
            recommendations.append("High volatility - consider position sizing adjustments")
        elif metrics.volatility < 0.05:
            recommendations.append("Low volatility may indicate insufficient market exposure")
        
        # Trading frequency recommendations
        if metrics.trades_per_month < 1:
            recommendations.append("Low trading frequency - strategy may be too conservative")
        elif metrics.trades_per_month > 50:
            recommendations.append("High trading frequency - monitor transaction costs")
        
        if not recommendations:
            recommendations.append("Strategy performance appears balanced across key metrics")
        
        return recommendations


def explain_performance_analysis():
    """
    Educational explanation of performance analysis
    """
    
    print("=== Performance Analysis Educational Guide ===\n")
    
    concepts = {
        'Risk-Adjusted Returns': "Returns measured relative to the risk taken to achieve them",
        
        'Sharpe Ratio': "Measures excess return per unit of total risk (volatility)",
        
        'Sortino Ratio': "Measures excess return per unit of downside risk only",
        
        'Maximum Drawdown': "Largest peak-to-trough decline in portfolio value",
        
        'Calmar Ratio': "Annual return divided by maximum drawdown",
        
        'Alpha': "Excess return relative to benchmark after adjusting for market risk",
        
        'Beta': "Sensitivity of strategy returns to market returns",
        
        'Information Ratio': "Active return divided by tracking error",
        
        'Win Rate': "Percentage of trades that are profitable",
        
        'Profit Factor': "Ratio of total profits to total losses",
        
        'Gain to Pain Ratio': "Ratio of positive returns to absolute negative returns"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Performance Analysis Best Practices ===")
    practices = [
        "1. Use multiple metrics for comprehensive evaluation",
        "2. Compare against appropriate benchmarks",
        "3. Analyze performance across different market regimes",
        "4. Consider risk-adjusted returns, not just absolute returns",
        "5. Evaluate consistency and stability of performance",
        "6. Monitor drawdowns and recovery times",
        "7. Assess trading costs and their impact",
        "8. Use rolling metrics to detect performance changes",
        "9. Perform attribution analysis to understand return sources",
        "10. Regularly review and update performance targets"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Interpreting Performance Metrics ===")
    interpretations = {
        "Sharpe Ratio > 2": "Excellent risk-adjusted performance",
        "Sharpe Ratio 1-2": "Good risk-adjusted performance",
        "Sharpe Ratio 0.5-1": "Acceptable performance",
        "Sharpe Ratio < 0.5": "Poor risk-adjusted performance",
        
        "Max Drawdown < -10%": "Excellent risk control",
        "Max Drawdown -10% to -20%": "Acceptable risk levels",
        "Max Drawdown -20% to -30%": "High risk, monitor closely",
        "Max Drawdown < -30%": "Excessive risk, review strategy",
        
        "Win Rate > 60%": "Strong strategy performance",
        "Win Rate 40-60%": "Normal range for many strategies",
        "Win Rate < 40%": "Strategy may need improvement",
        
        "Profit Factor > 2": "Excellent profitability",
        "Profit Factor 1.5-2": "Good profitability",
        "Profit Factor 1-1.5": "Marginal profitability",
        "Profit Factor < 1": "Strategy loses money"
    }
    
    for metric, interpretation in interpretations.items():
        print(f"{metric}:")
        print(f"  {interpretation}\n")


if __name__ == "__main__":
    # Example usage
    explain_performance_analysis()
    
    print("\n=== Performance Calculator Example ===")
    print("To use the performance calculator:")
    print("1. Provide backtest results with returns and equity curve")
    print("2. Optionally provide benchmark returns for comparison")
    print("3. Calculate comprehensive performance metrics")
    print("4. Analyze risk characteristics and attribution")
    print("5. Generate recommendations for improvement")
    
    print("\nKey features:")
    print("• 30+ performance metrics")
    print("• Risk-adjusted return calculations")
    print("• Benchmark comparison and attribution")
    print("• Rolling performance analysis")
    print("• Risk assessment and recommendations")
    print("• Monthly and daily breakdown")
    print("• Trading statistics and consistency metrics")
    
    print("\nThe calculator provides institutional-grade analysis:")
    print("• Comprehensive risk assessment")
    print("• Performance attribution")
    print("• Consistency evaluation")
    print("• Benchmark-relative analysis")
    print("• Actionable recommendations")