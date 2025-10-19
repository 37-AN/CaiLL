"""
Walk-Forward Analysis - AI Trading System

This module implements walk-forward analysis, a robust method for validating
trading strategies across different time periods to ensure they work well
out-of-sample and are not overfitted to historical data.

Educational Note:
Walk-forward analysis is the gold standard for strategy validation.
It simulates how a strategy would perform in real-time by:
1. Training on historical data
2. Testing on out-of-sample data
3. Rolling forward through time
4. Avoiding lookahead bias

This approach provides much more realistic performance estimates
than simple backtesting on a single time period.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import uuid
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting components
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, TradingStrategy


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    
    # Time parameters
    start_date: datetime
    end_date: datetime
    
    # Window sizes
    training_window: int  # Training period in days
    testing_window: int   # Testing period in days
    step_size: int        # Step size for rolling forward (days)
    
    # Optimization parameters
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    parameter_ranges: Dict[str, Tuple[Any, Any, Any]] = field(default_factory=dict)  # param: (min, max, step)
    
    # Validation parameters
    min_training_periods: int = 2  # Minimum training periods required
    confidence_level: float = 0.95  # For confidence intervals
    
    # Performance parameters
    benchmark: Optional[str] = None
    transaction_costs: bool = True
    
    # Parallel processing
    use_parallel: bool = True
    max_workers: int = 4


@dataclass
class WalkForwardPeriod:
    """Individual walk-forward period"""
    
    period_id: int
    training_start: datetime
    training_end: datetime
    testing_start: datetime
    testing_end: datetime
    
    # Optimization results
    optimal_parameters: Dict[str, Any]
    training_performance: Dict[str, float]
    
    # Testing results
    testing_performance: Dict[str, float]
    backtest_result: Optional[BacktestResult] = None
    
    # Metadata
    optimization_time: float = 0.0
    testing_time: float = 0.0
    notes: str = ""


@dataclass
class WalkForwardResult:
    """Results of walk-forward analysis"""
    
    config: WalkForwardConfig
    strategy_name: str
    
    # Period results
    periods: List[WalkForwardPeriod]
    
    # Aggregate performance
    aggregate_performance: Dict[str, float]
    
    # Parameter stability
    parameter_stability: Dict[str, float]
    parameter_evolution: Dict[str, List[Any]]
    
    # Performance consistency
    performance_consistency: Dict[str, float]
    hit_rate: float
    
    # Statistical validation
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_values: Dict[str, float]
    
    # Benchmark comparison
    benchmark_comparison: Dict[str, float]
    
    # Risk analysis
    risk_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    total_time: float = 0.0
    successful_periods: int = 0
    failed_periods: int = 0


class ParameterOptimizer:
    """
    Parameter optimizer for walk-forward analysis
    
    Educational Note:
    This optimizer finds the best parameters for each training period
    using various optimization methods. The goal is to maximize the
    chosen performance metric while avoiding overfitting.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.optimization_history: List[Dict] = []
    
    async def optimize_parameters(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        training_start: datetime,
        training_end: datetime
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize strategy parameters for given training period"""
        
        print(f"Optimizing parameters for {training_start.date()} to {training_end.date()}")
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations()
        
        # Evaluate each combination
        results = []
        
        for params in parameter_combinations:
            try:
                # Create strategy with parameters
                strategy = strategy_class("temp_strategy", params)
                
                # Create backtest config for training period
                bt_config = BacktestConfig(
                    start_date=training_start,
                    end_date=training_end,
                    initial_cash=100000,
                    commission_rate=0.001,
                    slippage_rate=0.0001
                )
                
                # Run backtest
                engine = BacktestEngine(bt_config)
                
                # Filter data for training period
                training_data = self._filter_data_by_period(data, training_start, training_end)
                
                # Run quick backtest for optimization
                result = await self._quick_backtest(strategy, training_data, bt_config)
                
                if result:
                    performance = self._extract_performance_metrics(result)
                    results.append({
                        'parameters': params.copy(),
                        'performance': performance,
                        'metric_value': performance.get(self.config.optimization_metric, 0)
                    })
                
            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                continue
        
        # Select best parameters
        if not results:
            return {}, {}
        
        # Sort by optimization metric
        results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        best_result = results[0]
        optimal_params = best_result['parameters']
        performance = best_result['performance']
        
        # Store optimization history
        self.optimization_history.append({
            'training_start': training_start,
            'training_end': training_end,
            'optimal_parameters': optimal_params,
            'performance': performance,
            'total_combinations': len(parameter_combinations),
            'successful_combinations': len(results)
        })
        
        print(f"Optimization complete. Best {self.config.optimization_metric}: {best_result['metric_value']:.4f}")
        
        return optimal_params, performance
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test"""
        
        if not self.config.parameter_ranges:
            return [{}]  # No parameters to optimize
        
        # Generate parameter values
        param_values = {}
        for param, (min_val, max_val, step) in self.config.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                param_values[param] = list(range(min_val, max_val + 1, step))
            else:
                # For float parameters
                num_steps = int((max_val - min_val) / step) + 1
                param_values[param] = [min_val + i * step for i in range(num_steps)]
        
        # Generate all combinations
        import itertools
        
        param_names = list(param_values.keys())
        param_value_lists = list(param_values.values())
        
        combinations = []
        for combination in itertools.product(*param_value_lists):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _filter_data_by_period(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Filter data for specific time period"""
        
        filtered_data = {}
        
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_data[symbol] = df[mask].copy()
        
        return filtered_data
    
    async def _quick_backtest(
        self,
        strategy: TradingStrategy,
        data: Dict[str, pd.DataFrame],
        config: BacktestConfig
    ) -> Optional[BacktestResult]:
        """Run quick backtest for parameter optimization"""
        
        try:
            engine = BacktestEngine(config)
            result = await engine.run_backtest(strategy, data)
            return result
        except Exception as e:
            print(f"Quick backtest error: {e}")
            return None
    
    def _extract_performance_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Extract key performance metrics from backtest result"""
        
        return {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor
        }


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis Engine
    
    Educational Note:
    This engine implements the complete walk-forward analysis process:
    1. Split data into training/testing periods
    2. Optimize parameters on training data
    3. Test optimized parameters on out-of-sample data
    4. Roll forward through time
    5. Aggregate results and analyze consistency
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.optimizer = ParameterOptimizer(config)
        self.analysis_periods: List[WalkForwardPeriod] = []
        
    async def run_analysis(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame]
    ) -> WalkForwardResult:
        """Run complete walk-forward analysis"""
        
        print(f"Starting walk-forward analysis for {strategy_class.__name__}")
        print(f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        print(f"Training window: {self.config.training_window} days")
        print(f"Testing window: {self.config.testing_window} days")
        print(f"Step size: {self.config.step_size} days")
        
        start_time = datetime.now()
        
        # Generate analysis periods
        periods = self._generate_analysis_periods()
        
        print(f"Generated {len(periods)} analysis periods")
        
        # Run analysis for each period
        successful_periods = []
        failed_periods = []
        
        for i, period in enumerate(periods):
            print(f"\n--- Period {i+1}/{len(periods)} ---")
            
            try:
                # Optimize parameters on training data
                optimal_params, training_perf = await self._optimize_period(
                    strategy_class, data, period
                )
                
                # Test on out-of-sample data
                testing_result, testing_perf = await self._test_period(
                    strategy_class, data, period, optimal_params
                )
                
                # Update period results
                period.optimal_parameters = optimal_params
                period.training_performance = training_perf
                period.testing_performance = testing_perf
                period.backtest_result = testing_result
                
                successful_periods.append(period)
                
                print(f"Period {i+1} completed successfully")
                print(f"Training Sharpe: {training_perf.get('sharpe_ratio', 0):.3f}")
                print(f"Testing Sharpe: {testing_perf.get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                print(f"Period {i+1} failed: {e}")
                failed_periods.append(period)
                continue
        
        # Calculate aggregate results
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        result = await self._calculate_aggregate_results(
            successful_periods, failed_periods, total_time
        )
        
        print(f"\nWalk-forward analysis completed in {total_time:.1f} seconds")
        print(f"Successful periods: {len(successful_periods)}/{len(periods)}")
        print(f"Aggregate Sharpe ratio: {result.aggregate_performance.get('sharpe_ratio', 0):.3f}")
        print(f"Hit rate: {result.hit_rate:.1%}")
        
        return result
    
    def _generate_analysis_periods(self) -> List[WalkForwardPeriod]:
        """Generate walk-forward analysis periods"""
        
        periods = []
        current_date = self.config.start_date
        period_id = 1
        
        while True:
            # Calculate training period
            training_start = current_date
            training_end = training_start + timedelta(days=self.config.training_window)
            
            # Calculate testing period
            testing_start = training_end
            testing_end = testing_start + timedelta(days=self.config.testing_window)
            
            # Check if we're within the overall date range
            if testing_end > self.config.end_date:
                # Adjust final testing period to end at config.end_date
                testing_end = self.config.end_date
                
                # If testing period is too short, skip
                if (testing_end - testing_start).days < 10:
                    break
            
            # Create period
            period = WalkForwardPeriod(
                period_id=period_id,
                training_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end
            )
            
            periods.append(period)
            
            # Move to next period
            current_date += timedelta(days=self.config.step_size)
            period_id += 1
            
            # Check if we should stop
            if training_start >= self.config.end_date:
                break
        
        return periods
    
    async def _optimize_period(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        period: WalkForwardPeriod
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize parameters for a specific period"""
        
        start_time = datetime.now()
        
        # Filter training data
        training_data = self._filter_data_by_period(
            data, period.training_start, period.training_end
        )
        
        # Optimize parameters
        optimal_params, performance = await self.optimizer.optimize_parameters(
            strategy_class, training_data, period.training_start, period.training_end
        )
        
        # Record optimization time
        period.optimization_time = (datetime.now() - start_time).total_seconds()
        
        return optimal_params, performance
    
    async def _test_period(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        period: WalkForwardPeriod,
        optimal_params: Dict[str, Any]
    ) -> Tuple[Optional[BacktestResult], Dict[str, float]]:
        """Test optimized parameters on out-of-sample data"""
        
        start_time = datetime.now()
        
        # Filter testing data
        testing_data = self._filter_data_by_period(
            data, period.testing_start, period.testing_end
        )
        
        # Create strategy with optimal parameters
        strategy = strategy_class(f"strategy_{period.period_id}", optimal_params)
        
        # Create backtest config
        bt_config = BacktestConfig(
            start_date=period.testing_start,
            end_date=period.testing_end,
            initial_cash=100000,
            commission_rate=0.001,
            slippage_rate=0.0001,
            benchmark=self.config.benchmark
        )
        
        # Run backtest
        engine = BacktestEngine(bt_config)
        result = await engine.run_backtest(strategy, testing_data)
        
        # Record testing time
        period.testing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract performance metrics
        performance = self._extract_performance_metrics(result)
        
        return result, performance
    
    def _filter_data_by_period(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Filter data for specific time period"""
        
        filtered_data = {}
        
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_data[symbol] = df[mask].copy()
        
        return filtered_data
    
    def _extract_performance_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Extract key performance metrics"""
        
        return {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'var_95': result.var_95,
            'cvar_95': result.cvar_95
        }
    
    async def _calculate_aggregate_results(
        self,
        successful_periods: List[WalkForwardPeriod],
        failed_periods: List[WalkForwardPeriod],
        total_time: float
    ) -> WalkForwardResult:
        """Calculate aggregate results from all periods"""
        
        if not successful_periods:
            raise ValueError("No successful periods to analyze")
        
        # Extract performance data
        training_metrics = {}
        testing_metrics = {}
        
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
            training_values = [p.training_performance.get(metric, 0) for p in successful_periods]
            testing_values = [p.testing_performance.get(metric, 0) for p in successful_periods]
            
            training_metrics[metric] = {
                'mean': np.mean(training_values),
                'std': np.std(training_values),
                'min': np.min(training_values),
                'max': np.max(training_values)
            }
            
            testing_metrics[metric] = {
                'mean': np.mean(testing_values),
                'std': np.std(testing_values),
                'min': np.min(testing_values),
                'max': np.max(testing_values)
            }
        
        # Calculate parameter stability
        parameter_stability = self._calculate_parameter_stability(successful_periods)
        
        # Calculate parameter evolution
        parameter_evolution = self._calculate_parameter_evolution(successful_periods)
        
        # Calculate performance consistency
        performance_consistency = self._calculate_performance_consistency(testing_metrics)
        
        # Calculate hit rate (periods with positive return)
        positive_returns = [p for p in successful_periods if p.testing_performance.get('total_return', 0) > 0]
        hit_rate = len(positive_returns) / len(successful_periods)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(successful_periods)
        
        # Calculate statistical significance
        p_values = self._calculate_statistical_significance(successful_periods)
        
        # Risk analysis
        risk_analysis = self._analyze_risk_characteristics(successful_periods)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            testing_metrics, parameter_stability, hit_rate
        )
        
        return WalkForwardResult(
            config=self.config,
            strategy_name=successful_periods[0].backtest_result.strategy_name if successful_periods[0].backtest_result else "Unknown",
            periods=successful_periods,
            aggregate_performance=testing_metrics,
            parameter_stability=parameter_stability,
            parameter_evolution=parameter_evolution,
            performance_consistency=performance_consistency,
            hit_rate=hit_rate,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            benchmark_comparison={},  # Would calculate if benchmark provided
            risk_analysis=risk_analysis,
            recommendations=recommendations,
            total_time=total_time,
            successful_periods=len(successful_periods),
            failed_periods=len(failed_periods)
        )
    
    def _calculate_parameter_stability(self, periods: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Calculate parameter stability across periods"""
        
        if not periods:
            return {}
        
        # Get all parameter names
        all_params = set()
        for period in periods:
            all_params.update(period.optimal_parameters.keys())
        
        stability = {}
        
        for param in all_params:
            values = []
            for period in periods:
                if param in period.optimal_parameters:
                    values.append(period.optimal_parameters[param])
            
            if len(values) > 1:
                # Calculate coefficient of variation (lower = more stable)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                stability[param] = 1 / (1 + cv)  # Convert to stability score (0-1)
            else:
                stability[param] = 1.0
        
        return stability
    
    def _calculate_parameter_evolution(self, periods: List[WalkForwardPeriod]) -> Dict[str, List[Any]]:
        """Track how parameters evolve over time"""
        
        evolution = {}
        
        # Get all parameter names
        all_params = set()
        for period in periods:
            all_params.update(period.optimal_parameters.keys())
        
        for param in all_params:
            evolution[param] = []
            for period in periods:
                if param in period.optimal_parameters:
                    evolution[param].append(period.optimal_parameters[param])
                else:
                    evolution[param].append(None)
        
        return evolution
    
    def _calculate_performance_consistency(self, testing_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate performance consistency metrics"""
        
        consistency = {}
        
        for metric, stats in testing_metrics.items():
            if stats['std'] > 0:
                # Consistency = 1 - coefficient of variation
                consistency[metric] = 1 - (stats['std'] / abs(stats['mean'])) if stats['mean'] != 0 else 0
            else:
                consistency[metric] = 1.0
        
        return consistency
    
    def _calculate_confidence_intervals(self, periods: List[WalkForwardPeriod]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        
        confidence_level = self.config.confidence_level
        alpha = 1 - confidence_level
        
        intervals = {}
        
        # Extract Sharpe ratios
        sharpe_values = [p.testing_performance.get('sharpe_ratio', 0) for p in periods]
        
        if len(sharpe_values) > 1:
            mean_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            
            # Calculate t-critical value
            from scipy import stats
            t_critical = stats.t.ppf(1 - alpha/2, len(sharpe_values) - 1)
            
            margin_error = t_critical * (std_sharpe / np.sqrt(len(sharpe_values)))
            
            intervals['sharpe_ratio'] = (
                mean_sharpe - margin_error,
                mean_sharpe + margin_error
            )
        
        return intervals
    
    def _calculate_statistical_significance(self, periods: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Calculate p-values for statistical significance"""
        
        p_values = {}
        
        # Test if Sharpe ratio is significantly different from zero
        sharpe_values = [p.testing_performance.get('sharpe_ratio', 0) for p in periods]
        
        if len(sharpe_values) > 1:
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(sharpe_values, 0)
            p_values['sharpe_ratio'] = p_val
        
        return p_values
    
    def _analyze_risk_characteristics(self, periods: List[WalkForwardPeriod]) -> Dict[str, Any]:
        """Analyze risk characteristics across periods"""
        
        # Extract risk metrics
        max_drawdowns = [p.testing_performance.get('max_drawdown', 0) for p in periods]
        var_95s = [p.testing_performance.get('var_95', 0) for p in periods]
        
        risk_analysis = {
            'max_drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'max': np.max(max_drawdowns),
                'worst_periods': [i for i, dd in enumerate(max_drawdowns) if dd <= np.percentile(max_drawdowns, 10)]
            },
            'var_95_stats': {
                'mean': np.mean(var_95s),
                'std': np.std(var_95s),
                'max': np.max(var_95s)
            },
            'risk_adjusted_consistency': self._calculate_risk_adjusted_consistency(periods)
        }
        
        return risk_analysis
    
    def _calculate_risk_adjusted_consistency(self, periods: List[WalkForwardPeriod]) -> float:
        """Calculate risk-adjusted consistency metric"""
        
        # Calculate average return and risk
        returns = [p.testing_performance.get('total_return', 0) for p in periods]
        risks = [abs(p.testing_performance.get('max_drawdown', 0)) for p in periods]
        
        if not returns or not risks:
            return 0.0
        
        # Calculate risk-adjusted returns
        risk_adjusted_returns = [r / (risk + 0.01) for r, risk in zip(returns, risks)]  # Add small constant to avoid division by zero
        
        # Consistency is 1 - coefficient of variation
        mean_ra = np.mean(risk_adjusted_returns)
        std_ra = np.std(risk_adjusted_returns)
        
        return 1 - (std_ra / abs(mean_ra)) if mean_ra != 0 else 0.0
    
    def _generate_recommendations(
        self,
        testing_metrics: Dict[str, Dict],
        parameter_stability: Dict[str, float],
        hit_rate: float
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Performance recommendations
        avg_sharpe = testing_metrics.get('sharpe_ratio', {}).get('mean', 0)
        if avg_sharpe < 0.5:
            recommendations.append("Low Sharpe ratio suggests strategy may not be profitable")
        elif avg_sharpe > 1.5:
            recommendations.append("High Sharpe ratio indicates strong risk-adjusted performance")
        
        # Consistency recommendations
        sharpe_consistency = testing_metrics.get('sharpe_ratio', {}).get('std', 0)
        if sharpe_consistency > 1.0:
            recommendations.append("High Sharpe ratio variability suggests inconsistent performance")
        
        # Hit rate recommendations
        if hit_rate < 0.4:
            recommendations.append("Low hit rate (< 40%) suggests strategy may need refinement")
        elif hit_rate > 0.7:
            recommendations.append("High hit rate (> 70%) indicates strong consistency")
        
        # Parameter stability recommendations
        unstable_params = [p for p, stability in parameter_stability.items() if stability < 0.5]
        if unstable_params:
            recommendations.append(f"Parameters {unstable_params} show low stability - consider fixing them")
        
        # Risk recommendations
        avg_max_dd = testing_metrics.get('max_drawdown', {}).get('mean', 0)
        if avg_max_dd < -0.2:
            recommendations.append("High maximum drawdown suggests excessive risk")
        
        if not recommendations:
            recommendations.append("Strategy shows good overall performance and consistency")
        
        return recommendations


def explain_walk_forward_analysis():
    """
    Educational explanation of walk-forward analysis
    """
    
    print("=== Walk-Forward Analysis Educational Guide ===\n")
    
    concepts = {
        'Walk-Forward Analysis': "Robust validation method that optimizes parameters on training data and tests on out-of-sample data",
        
        'Training Window': "Historical period used to optimize strategy parameters",
        
        'Testing Window': "Out-of-sample period used to validate optimized parameters",
        
        'Step Size': "Number of days to move forward for each new period",
        
        'Parameter Stability': "Consistency of optimal parameters across different periods",
        
        'Performance Consistency': "Consistency of strategy performance across periods",
        
        'Hit Rate': "Percentage of periods with positive returns",
        
        'Lookahead Bias': "Using future information in backtesting, which walk-forward avoids",
        
        'Overfitting': "Creating strategy that works too well on historical data but fails in live trading"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Walk-Forward vs Simple Backtesting ===")
    differences = {
        "Validation Method": "Simple: Single period test | Walk-Forward: Multiple rolling tests",
        "Parameter Selection": "Simple: Optimized on full dataset | Walk-Forward: Optimized on training only",
        "Realism": "Simple: Prone to overfitting | Walk-Forward: More realistic performance estimates",
        "Statistical Validity": "Simple: Weak validation | Walk-Forward: Strong statistical validation",
        "Parameter Stability": "Simple: Not assessed | Walk-Forward: Explicitly measured",
        "Performance Consistency": "Simple: Single estimate | Walk-Forward: Distribution of performance"
    }
    
    for aspect, comparison in differences.items():
        print(f"{aspect}:")
        print(f"  {comparison}\n")
    
    print("=== Walk-Forward Best Practices ===")
    practices = [
        "1. Use sufficient training data (at least 2-3 years)",
        "2. Keep testing periods reasonable (3-6 months)",
        "3. Use appropriate step sizes (1-3 months)",
        "4. Ensure enough periods for statistical significance",
        "5. Analyze parameter stability carefully",
        "6. Consider transaction costs and slippage",
        "7. Test across different market regimes",
        "8. Validate with out-of-sample data",
        "9. Monitor performance consistency",
        "10. Be cautious of over-optimization"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Interpreting Walk-Forward Results ===")
    interpretations = {
        "High Hit Rate (> 60%)": "Strategy performs consistently across different periods",
        "Parameter Stability > 0.7": "Optimal parameters are consistent, indicating robust strategy",
        "Sharpe Ratio Consistency": "Low variability in Sharpe ratio suggests reliable performance",
        "Performance Decline": "If testing performance is much lower than training, watch for overfitting",
        "Parameter Evolution": "Changing optimal parameters may indicate adapting market conditions",
        "Risk Analysis": "Consistent risk metrics suggest stable risk management"
    }
    
    for metric, interpretation in interpretations.items():
        print(f"{metric}:")
        print(f"  {interpretation}\n")


if __name__ == "__main__":
    # Example usage
    explain_walk_forward_analysis()
    
    print("\n=== Walk-Forward Analysis Example ===")
    print("To use walk-forward analysis:")
    print("1. Create WalkForwardConfig with time windows")
    print("2. Define parameter ranges to optimize")
    print("3. Implement your trading strategy class")
    print("4. Prepare historical market data")
    print("5. Run the analysis")
    print("6. Analyze parameter stability and performance consistency")
    
    # Example configuration
    config = WalkForwardConfig(
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2023, 12, 31),
        training_window=252 * 2,  # 2 years training
        testing_window=63,        # 3 months testing
        step_size=30,             # 1 month step
        optimization_metric="sharpe_ratio",
        parameter_ranges={
            'lookback_period': (10, 50, 10),
            'entry_threshold': (0.1, 0.5, 0.1),
            'exit_threshold': (0.05, 0.3, 0.05)
        }
    )
    
    print(f"\nSample configuration:")
    print(f"  Analysis period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Training window: {config.training_window} days")
    print(f"  Testing window: {config.testing_window} days")
    print(f"  Step size: {config.step_size} days")
    print(f"  Optimization metric: {config.optimization_metric}")
    print(f"  Parameters to optimize: {list(config.parameter_ranges.keys())}")
    
    print("\nWalk-forward analysis provides:")
    print("• Robust out-of-sample validation")
    print("• Parameter stability analysis")
    print("• Performance consistency metrics")
    print("• Statistical significance testing")
    print("• Realistic performance expectations")
    print("• Risk assessment across market conditions")