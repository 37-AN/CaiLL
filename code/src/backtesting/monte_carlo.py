"""
Monte Carlo Simulation - AI Trading System

This module implements Monte Carlo simulation for analyzing strategy performance
uncertainty and risk. It helps understand the range of possible outcomes
and the probability of achieving different performance levels.

Educational Note:
Monte Carlo simulation is essential for understanding strategy risk.
Instead of a single backtest result, you get a distribution of possible
outcomes, helping you answer questions like:
- What's the probability of losing money?
- What's the range of possible returns?
- How sensitive is the strategy to market conditions?
- What are the worst-case scenarios?

This is crucial for proper risk management and position sizing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting components
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, TradingStrategy


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    
    # Simulation parameters
    num_simulations: int = 1000
    time_horizon: int = 252  # Trading days (1 year)
    
    # Randomness parameters
    random_seed: Optional[int] = None
    bootstrap_samples: bool = True
    block_bootstrap: bool = True
    block_size: int = 20  # For block bootstrap
    
    # Market condition scenarios
    include_scenarios: bool = True
    scenario_weights: Dict[str, float] = field(default_factory=dict)
    
    # Parameter uncertainty
    parameter_uncertainty: bool = True
    parameter_std: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
    ])
    
    # Parallel processing
    use_parallel: bool = True
    max_workers: int = 4
    
    # Output settings
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    percentiles: List[float] = field(default_factory=lambda: [5, 25, 50, 75, 95])


@dataclass
class SimulationResult:
    """Result of a single simulation run"""
    
    simulation_id: int
    scenario_type: str
    parameters: Dict[str, Any]
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Path data
    equity_curve: pd.Series
    returns: pd.Series
    
    # Risk metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Metadata
    final_portfolio_value: float
    total_trades: int
    execution_time: float = 0.0


@dataclass
class MonteCarloResults:
    """Results of Monte Carlo simulation"""
    
    config: MonteCarloConfig
    strategy_name: str
    base_result: BacktestResult
    
    # Simulation results
    simulations: List[SimulationResult]
    
    # Statistical distributions
    metric_distributions: Dict[str, np.ndarray]
    
    # Summary statistics
    summary_stats: Dict[str, Dict[str, float]]
    
    # Risk analysis
    risk_metrics: Dict[str, Any]
    
    # Scenario analysis
    scenario_results: Dict[str, List[SimulationResult]]
    
    # Confidence intervals
    confidence_intervals: Dict[str, Dict[float, Tuple[float, float]]]
    
    # Probability analysis
    probability_analysis: Dict[str, Dict[str, float]]
    
    # Sensitivity analysis
    sensitivity_analysis: Dict[str, Dict[str, float]]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    total_time: float = 0.0
    successful_simulations: int = 0
    failed_simulations: int = 0


class MarketScenarioGenerator:
    """
    Generate different market scenarios for Monte Carlo simulation
    
    Educational Note:
    Market scenarios help test strategy performance under different
    market conditions like bull markets, bear markets, high volatility,
    and low volatility periods. This provides a more complete picture
    of strategy risk and robustness.
    """
    
    def __init__(self):
        self.scenarios = {
            'normal': {'mean_return': 0.0005, 'volatility': 0.01, 'skewness': 0, 'kurtosis': 3},
            'bull': {'mean_return': 0.001, 'volatility': 0.008, 'skewness': 0.2, 'kurtosis': 2.5},
            'bear': {'mean_return': -0.0005, 'volatility': 0.015, 'skewness': -0.3, 'kurtosis': 4},
            'volatile': {'mean_return': 0.0002, 'volatility': 0.025, 'skewness': 0, 'kurtosis': 5},
            'low_vol': {'mean_return': 0.0003, 'volatility': 0.005, 'skewness': 0, 'kurtosis': 2.5}
        }
    
    def generate_scenario_returns(
        self,
        scenario_type: str,
        num_days: int,
        base_returns: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate returns for a specific market scenario"""
        
        if scenario_type not in self.scenarios:
            scenario_type = 'normal'
        
        scenario_params = self.scenarios[scenario_type]
        
        if base_returns is not None and len(base_returns) > 0:
            # Use bootstrap with scenario adjustment
            return self._bootstrap_with_scenario(base_returns, scenario_params, num_days)
        else:
            # Generate synthetic returns
            return self._generate_synthetic_returns(scenario_params, num_days)
    
    def _bootstrap_with_scenario(
        self,
        base_returns: pd.Series,
        scenario_params: Dict[str, float],
        num_days: int
    ) -> pd.Series:
        """Bootstrap returns with scenario adjustment"""
        
        # Sample from base returns
        sampled_returns = np.random.choice(base_returns, size=num_days, replace=True)
        
        # Adjust for scenario
        adjustment_factor = (scenario_params['mean_return'] - base_returns.mean()) / base_returns.std()
        volatility_adjustment = scenario_params['volatility'] / base_returns.std()
        
        adjusted_returns = sampled_returns * volatility_adjustment + adjustment_factor * base_returns.std()
        
        return pd.Series(adjusted_returns)
    
    def _generate_synthetic_returns(self, scenario_params: Dict[str, float], num_days: int) -> pd.Series:
        """Generate synthetic returns for scenario"""
        
        # Use skewed normal distribution or similar
        returns = np.random.normal(
            scenario_params['mean_return'],
            scenario_params['volatility'],
            num_days
        )
        
        # Add skewness and kurtosis adjustments (simplified)
        if scenario_params['skewness'] != 0:
            # Add skewness (simplified approach)
            skew_adjustment = scenario_params['skewness'] * 0.1
            returns = returns + skew_adjustment * np.random.normal(0, 1, num_days)
        
        return pd.Series(returns)


class ParameterSampler:
    """
    Sample strategy parameters for uncertainty analysis
    
    Educational Note:
    Strategy parameters are never known with certainty. This sampler
    generates parameter variations to test how sensitive the strategy
    is to parameter changes. This helps identify robust parameters
    and assess parameter risk.
    """
    
    def __init__(self, parameter_std: Dict[str, float]):
        self.parameter_std = parameter_std
    
    def sample_parameters(
        self,
        base_parameters: Dict[str, Any],
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Sample parameters around base values"""
        
        sampled_params = []
        
        for _ in range(num_samples):
            params = base_parameters.copy()
            
            for param, std in self.parameter_std.items():
                if param in params:
                    # Sample from normal distribution
                    value = params[param]
                    if isinstance(value, (int, float)):
                        sampled_value = np.random.normal(value, std * abs(value))
                        
                        # Ensure reasonable bounds
                        if isinstance(value, int):
                            sampled_value = max(1, int(sampled_value))  # Keep positive integers
                        else:
                            sampled_value = max(0.001, sampled_value)  # Keep positive floats
                        
                        params[param] = sampled_value
            
            sampled_params.append(params)
        
        return sampled_params


class MonteCarloSimulator:
    """
    Monte Carlo Simulation Engine
    
    Educational Note:
    This engine runs thousands of simulations to understand the range
    of possible outcomes for a trading strategy. Each simulation
    represents a possible future scenario, allowing us to calculate
    probabilities and confidence intervals for different performance
    metrics.
    """
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.scenario_generator = MarketScenarioGenerator()
        self.parameter_sampler = ParameterSampler(config.parameter_std)
        
        # Set random seed if provided
        if config.random_seed:
            np.random.seed(config.random_seed)
    
    async def run_simulation(
        self,
        strategy_class: type,
        base_parameters: Dict[str, Any],
        base_result: BacktestResult,
        historical_data: Dict[str, pd.DataFrame]
    ) -> MonteCarloResults:
        """Run complete Monte Carlo simulation"""
        
        print(f"Starting Monte Carlo simulation with {self.config.num_simulations} runs")
        print(f"Time horizon: {self.config.time_horizon} days")
        
        start_time = datetime.now()
        
        # Prepare simulation parameters
        simulation_params = self._prepare_simulation_parameters(
            base_parameters, base_result, historical_data
        )
        
        # Run simulations
        simulations = []
        successful_sims = 0
        failed_sims = 0
        
        for i in range(self.config.num_simulations):
            try:
                # Determine scenario type
                scenario_type = self._select_scenario_type(i)
                
                # Sample parameters if uncertainty is enabled
                if self.config.parameter_uncertainty:
                    params = self.parameter_sampler.sample_parameters(
                        base_parameters, 1
                    )[0]
                else:
                    params = base_parameters.copy()
                
                # Run single simulation
                result = await self._run_single_simulation(
                    i, scenario_type, params, historical_data, base_result
                )
                
                if result:
                    simulations.append(result)
                    successful_sims += 1
                    
                    # Progress indicator
                    if (i + 1) % 100 == 0:
                        progress = ((i + 1) / self.config.num_simulations) * 100
                        print(f"Progress: {progress:.1f}% - Successful: {successful_sims}")
                
                else:
                    failed_sims += 1
                
            except Exception as e:
                print(f"Simulation {i} failed: {e}")
                failed_sims += 1
                continue
        
        # Calculate results
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        results = await self._calculate_results(
            simulations, base_result, total_time, successful_sims, failed_sims
        )
        
        print(f"Monte Carlo simulation completed in {total_time:.1f} seconds")
        print(f"Successful simulations: {successful_sims}/{self.config.num_simulations}")
        
        return results
    
    def _prepare_simulation_parameters(
        self,
        base_parameters: Dict[str, Any],
        base_result: BacktestResult,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare parameters for simulation"""
        
        # Extract base returns for bootstrap
        base_returns = base_result.returns
        
        return {
            'base_returns': base_returns,
            'historical_data': historical_data,
            'time_horizon': self.config.time_horizon
        }
    
    def _select_scenario_type(self, simulation_id: int) -> str:
        """Select scenario type for simulation"""
        
        if not self.config.include_scenarios:
            return 'normal'
        
        # Use scenario weights if provided
        if self.config.scenario_weights:
            scenarios = list(self.config.scenario_weights.keys())
            weights = list(self.config.scenario_weights.values())
            return np.random.choice(scenarios, p=weights)
        
        # Default scenario distribution
        scenarios = ['normal', 'normal', 'normal', 'bull', 'bear', 'volatile', 'low_vol']
        return np.random.choice(scenarios)
    
    async def _run_single_simulation(
        self,
        simulation_id: int,
        scenario_type: str,
        parameters: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        base_result: BacktestResult
    ) -> Optional[SimulationResult]:
        """Run a single Monte Carlo simulation"""
        
        start_time = datetime.now()
        
        try:
            # Generate scenario returns
            scenario_returns = self.scenario_generator.generate_scenario_returns(
                scenario_type, self.config.time_horizon, base_result.returns
            )
            
            # Create synthetic market data
            synthetic_data = self._create_synthetic_market_data(
                historical_data, scenario_returns
            )
            
            # Create strategy with sampled parameters
            strategy = strategy_class(f"mc_strategy_{simulation_id}", parameters)
            
            # Create backtest config
            bt_config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=self.config.time_horizon),
                end_date=datetime.now(),
                initial_cash=100000,
                commission_rate=0.001,
                slippage_rate=0.0001,
                mode='vectorized'  # Use vectorized for speed
            )
            
            # Run backtest
            engine = BacktestEngine(bt_config)
            result = await engine.run_backtest(strategy, synthetic_data)
            
            # Calculate additional metrics
            var_95 = result.returns.quantile(0.05) if len(result.returns) > 0 else 0
            cvar_95 = result.returns[result.returns <= var_95].mean() if len(result.returns) > 0 else 0
            skewness = result.returns.skew() if len(result.returns) > 0 else 0
            kurtosis = result.returns.kurtosis() if len(result.returns) > 0 else 0
            
            # Create simulation result
            sim_result = SimulationResult(
                simulation_id=simulation_id,
                scenario_type=scenario_type,
                parameters=parameters,
                total_return=result.total_return,
                annualized_return=result.annualized_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                max_drawdown=result.max_drawdown,
                calmar_ratio=result.calmar_ratio,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                equity_curve=result.equity_curve,
                returns=result.returns,
                var_95=var_95,
                cvar_95=cvar_95,
                skewness=skewness,
                kurtosis=kurtosis,
                final_portfolio_value=result.final_portfolio_value,
                total_trades=result.total_trades,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            return sim_result
            
        except Exception as e:
            print(f"Single simulation {simulation_id} failed: {e}")
            return None
    
    def _create_synthetic_market_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        scenario_returns: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """Create synthetic market data based on scenario returns"""
        
        synthetic_data = {}
        
        for symbol, hist_df in historical_data.items():
            # Start from last historical price
            last_price = hist_df['close'].iloc[-1]
            
            # Generate synthetic prices
            synthetic_prices = [last_price]
            
            for ret in scenario_returns:
                new_price = synthetic_prices[-1] * (1 + ret)
                synthetic_prices.append(new_price)
            
            # Create synthetic OHLCV data
            synthetic_df = pd.DataFrame({
                'open': synthetic_prices[:-1],
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in synthetic_prices[:-1]],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in synthetic_prices[:-1]],
                'close': synthetic_prices[1:],
                'volume': [int(np.random.normal(1000000, 200000)) for _ in range(len(synthetic_prices)-1)]
            })
            
            # Create date index
            start_date = datetime.now() - timedelta(days=len(scenario_returns))
            dates = pd.date_range(start=start_date, periods=len(synthetic_prices)-1, freq='D')
            synthetic_df.index = dates
            
            synthetic_data[symbol] = synthetic_df
        
        return synthetic_data
    
    async def _calculate_results(
        self,
        simulations: List[SimulationResult],
        base_result: BacktestResult,
        total_time: float,
        successful_sims: int,
        failed_sims: int
    ) -> MonteCarloResults:
        """Calculate comprehensive Monte Carlo results"""
        
        if not simulations:
            raise ValueError("No successful simulations to analyze")
        
        # Extract metric distributions
        metric_distributions = {}
        for metric in self.config.metrics:
            values = [getattr(sim, metric) for sim in simulations]
            metric_distributions[metric] = np.array(values)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(metric_distributions)
        
        # Risk analysis
        risk_metrics = self._analyze_risk(simulations)
        
        # Scenario analysis
        scenario_results = self._analyze_scenarios(simulations)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(metric_distributions)
        
        # Probability analysis
        probability_analysis = self._calculate_probabilities(simulations, metric_distributions)
        
        # Sensitivity analysis
        sensitivity_analysis = self._analyze_sensitivity(simulations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary_stats, risk_metrics)
        
        return MonteCarloResults(
            config=self.config,
            strategy_name=base_result.strategy_name,
            base_result=base_result,
            simulations=simulations,
            metric_distributions=metric_distributions,
            summary_stats=summary_stats,
            risk_metrics=risk_metrics,
            scenario_results=scenario_results,
            confidence_intervals=confidence_intervals,
            probability_analysis=probability_analysis,
            sensitivity_analysis=sensitivity_analysis,
            recommendations=recommendations,
            total_time=total_time,
            successful_simulations=successful_sims,
            failed_simulations=failed_sims
        )
    
    def _calculate_summary_statistics(self, metric_distributions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each metric"""
        
        summary_stats = {}
        
        for metric, values in metric_distributions.items():
            stats = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
            
            # Add percentiles from config
            for percentile in self.config.percentiles:
                stats[f'p{percentile}'] = np.percentile(values, percentile)
            
            summary_stats[metric] = stats
        
        return summary_stats
    
    def _analyze_risk(self, simulations: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        
        # Extract risk metrics
        max_drawdowns = [sim.max_drawdown for sim in simulations]
        var_95s = [sim.var_95 for sim in simulations]
        cvar_95s = [sim.cvar_95 for sim in simulations]
        volatilities = [sim.volatility for sim in simulations]
        
        # Calculate probability of loss
        loss_probability = len([sim for sim in simulations if sim.total_return < 0]) / len(simulations)
        
        # Calculate probability of large loss
        large_loss_probability = len([sim for sim in simulations if sim.total_return < -0.20]) / len(simulations)
        
        # Calculate probability of extreme drawdown
        extreme_dd_probability = len([sim for sim in simulations if sim.max_drawdown < -0.30]) / len(simulations)
        
        return {
            'loss_probability': loss_probability,
            'large_loss_probability': large_loss_probability,
            'extreme_drawdown_probability': extreme_dd_probability,
            'max_drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'worst': np.min(max_drawdowns),
                'best': np.max(max_drawdowns)
            },
            'var_95_stats': {
                'mean': np.mean(var_95s),
                'std': np.std(var_95s),
                'worst': np.min(var_95s)
            },
            'volatility_stats': {
                'mean': np.mean(volatilities),
                'std': np.std(volatilities),
                'lowest': np.min(volatilities),
                'highest': np.max(volatilities)
            }
        }
    
    def _analyze_scenarios(self, simulations: List[SimulationResult]) -> Dict[str, List[SimulationResult]]:
        """Analyze performance by scenario"""
        
        scenario_results = {}
        
        for sim in simulations:
            scenario = sim.scenario_type
            if scenario not in scenario_results:
                scenario_results[scenario] = []
            scenario_results[scenario].append(sim)
        
        return scenario_results
    
    def _calculate_confidence_intervals(
        self,
        metric_distributions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[float, Tuple[float, float]]]:
        """Calculate confidence intervals for metrics"""
        
        confidence_intervals = {}
        
        for metric, values in metric_distributions.items():
            intervals = {}
            
            for confidence_level in self.config.confidence_levels:
                alpha = 1 - confidence_level
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                intervals[confidence_level] = (lower, upper)
            
            confidence_intervals[metric] = intervals
        
        return confidence_intervals
    
    def _calculate_probabilities(
        self,
        simulations: List[SimulationResult],
        metric_distributions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate probability of various outcomes"""
        
        probabilities = {}
        
        # Return probabilities
        returns = metric_distributions['total_return']
        probabilities['return'] = {
            'positive': np.mean(returns > 0),
            'greater_than_10pct': np.mean(returns > 0.10),
            'greater_than_20pct': np.mean(returns > 0.20),
            'loss_greater_than_10pct': np.mean(returns < -0.10),
            'loss_greater_than_20pct': np.mean(returns < -0.20)
        }
        
        # Sharpe ratio probabilities
        sharpe_ratios = metric_distributions['sharpe_ratio']
        probabilities['sharpe_ratio'] = {
            'positive': np.mean(sharpe_ratios > 0),
            'greater_than_1': np.mean(sharpe_ratios > 1),
            'greater_than_2': np.mean(sharpe_ratios > 2)
        }
        
        # Drawdown probabilities
        max_drawdowns = metric_distributions['max_drawdown']
        probabilities['max_drawdown'] = {
            'less_than_10pct': np.mean(max_drawdowns > -0.10),
            'less_than_20pct': np.mean(max_drawdowns > -0.20),
            'greater_than_20pct': np.mean(max_drawdowns < -0.20),
            'greater_than_30pct': np.mean(max_drawdowns < -0.30)
        }
        
        return probabilities
    
    def _analyze_sensitivity(self, simulations: List[SimulationResult]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity"""
        
        sensitivity = {}
        
        if not simulations or not self.config.parameter_uncertainty:
            return sensitivity
        
        # Get all parameter names
        all_params = set()
        for sim in simulations:
            all_params.update(sim.parameters.keys())
        
        # Analyze sensitivity for each parameter
        for param in all_params:
            param_values = []
            sharpe_ratios = []
            
            for sim in simulations:
                if param in sim.parameters:
                    param_values.append(sim.parameters[param])
                    sharpe_ratios.append(sim.sharpe_ratio)
            
            if len(param_values) > 1:
                # Calculate correlation
                correlation = np.corrcoef(param_values, sharpe_ratios)[0, 1]
                sensitivity[param] = {
                    'correlation_with_sharpe': correlation,
                    'sensitivity_score': abs(correlation)
                }
        
        return sensitivity
    
    def _generate_recommendations(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        risk_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on Monte Carlo results"""
        
        recommendations = []
        
        # Return analysis
        mean_return = summary_stats['total_return']['mean']
        return_std = summary_stats['total_return']['std']
        
        if mean_return < 0:
            recommendations.append("Negative expected return - strategy may not be profitable")
        elif mean_return > 0 and return_std > mean_return:
            recommendations.append("High return volatility - consider position sizing adjustments")
        
        # Risk analysis
        loss_prob = risk_metrics['loss_probability']
        if loss_prob > 0.4:
            recommendations.append("High probability of loss (> 40%) - review strategy logic")
        elif loss_prob < 0.1:
            recommendations.append("Low probability of loss (< 10%) - strong strategy performance")
        
        # Drawdown analysis
        extreme_dd_prob = risk_metrics['extreme_drawdown_probability']
        if extreme_dd_prob > 0.1:
            recommendations.append("High probability of extreme drawdown - implement stronger risk controls")
        
        # Sharpe ratio analysis
        mean_sharpe = summary_stats['sharpe_ratio']['mean']
        if mean_sharpe < 0.5:
            recommendations.append("Low risk-adjusted returns - consider strategy improvements")
        elif mean_sharpe > 1.5:
            recommendations.append("Excellent risk-adjusted performance - strategy appears robust")
        
        # Consistency analysis
        return_consistency = 1 - (return_std / abs(mean_return)) if mean_return != 0 else 0
        if return_consistency < 0.5:
            recommendations.append("High performance variability - strategy may be unstable")
        
        if not recommendations:
            recommendations.append("Monte Carlo analysis indicates solid strategy performance")
        
        return recommendations


def explain_monte_carlo():
    """
    Educational explanation of Monte Carlo simulation
    """
    
    print("=== Monte Carlo Simulation Educational Guide ===\n")
    
    concepts = {
        'Monte Carlo Simulation': "Using random sampling to understand possible outcomes and their probabilities",
        
        'Confidence Interval': "Range of values within which the true parameter likely falls",
        
        'Probability Distribution': "Mathematical function describing likelihood of different outcomes",
        
        'Scenario Analysis': "Testing strategy under different market conditions",
        
        'Parameter Uncertainty': "Accounting for uncertainty in strategy parameters",
        
        'Bootstrap Method': "Resampling historical data to create new scenarios",
        
        'Value at Risk (VaR)': "Maximum expected loss at a given confidence level",
        
        'Conditional VaR': "Expected loss given that VaR is exceeded",
        
        'Stress Testing': "Testing strategy under extreme market conditions",
        
        'Sensitivity Analysis': "Understanding how changes in parameters affect performance"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Monte Carlo vs Single Backtest ===")
    differences = {
        "Result Type": "Single: Point estimate | Monte Carlo: Distribution of outcomes",
        "Risk Assessment": "Single: Limited risk view | Monte Carlo: Comprehensive risk analysis",
        "Confidence": "Single: No confidence measure | Monte Carlo: Statistical confidence intervals",
        "Scenarios": "Single: One historical path | Monte Carlo: Multiple possible paths",
        "Parameter Risk": "Single: Fixed parameters | Monte Carlo: Parameter uncertainty",
        "Decision Making": "Single: Binary decision | Monte Carlo: Probabilistic decision making"
    }
    
    for aspect, comparison in differences.items():
        print(f"{aspect}:")
        print(f"  {comparison}\n")
    
    print("=== Monte Carlo Best Practices ===")
    practices = [
        "1. Use sufficient number of simulations (1000+ for reliable results)",
        "2. Include different market scenarios (bull, bear, volatile)",
        "3. Account for parameter uncertainty",
        "4. Use appropriate time horizons",
        "5. Analyze multiple performance metrics",
        "6. Consider transaction costs and slippage",
        "7. Validate with out-of-sample data",
        "8. Use proper statistical methods",
        "9. Focus on risk metrics as well as returns",
        "10. Update simulations regularly with new data"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Interpreting Monte Carlo Results ===")
    interpretations = {
        "Wide Confidence Intervals": "High uncertainty in performance estimates",
        "High Loss Probability": "Strategy carries significant risk of losses",
        "Parameter Sensitivity": "Strategy performance depends heavily on specific parameters",
        "Scenario Dependence": "Strategy performs very differently across market conditions",
        "Consistent Positive Returns": "Strategy appears robust across scenarios",
        "Low Volatility of Results": "Strategy performance is predictable and stable"
    }
    
    for metric, interpretation in interpretations.items():
        print(f"{metric}:")
        print(f"  {interpretation}\n")


if __name__ == "__main__":
    # Example usage
    explain_monte_carlo()
    
    print("\n=== Monte Carlo Simulation Example ===")
    print("To use Monte Carlo simulation:")
    print("1. Create MonteCarloConfig with simulation parameters")
    print("2. Define parameter uncertainty ranges")
    print("3. Set up market scenarios")
    print("4. Run the simulation")
    print("5. Analyze probability distributions and confidence intervals")
    
    # Example configuration
    config = MonteCarloConfig(
        num_simulations=1000,
        time_horizon=252,  # 1 year
        include_scenarios=True,
        parameter_uncertainty=True,
        parameter_std={
            'lookback_period': 0.1,  # 10% standard deviation
            'entry_threshold': 0.05,  # 5% standard deviation
            'exit_threshold': 0.05
        },
        confidence_levels=[0.90, 0.95, 0.99],
        scenario_weights={
            'normal': 0.4,
            'bull': 0.2,
            'bear': 0.2,
            'volatile': 0.1,
            'low_vol': 0.1
        }
    )
    
    print(f"\nSample configuration:")
    print(f"  Simulations: {config.num_simulations}")
    print(f"  Time horizon: {config.time_horizon} days")
    print(f"  Parameter uncertainty: {config.parameter_uncertainty}")
    print(f"  Market scenarios: {config.include_scenarios}")
    print(f"  Confidence levels: {config.confidence_levels}")
    
    print("\nMonte Carlo simulation provides:")
    print("• Probability distributions for performance metrics")
    print("• Confidence intervals for key statistics")
    print("• Risk assessment across market scenarios")
    print("• Parameter sensitivity analysis")
    print("• Stress testing capabilities")
    print("• Robust decision-making framework")