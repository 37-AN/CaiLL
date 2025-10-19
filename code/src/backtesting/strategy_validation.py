"""
Strategy Validation Framework - AI Trading System

This module implements statistical validation tools for trading strategies,
including hypothesis testing, significance testing, and robust validation
methodologies to ensure strategies are genuinely profitable.

Educational Note:
Statistical validation is crucial to distinguish between luck and skill.
A strategy might appear profitable due to random chance or overfitting.
This framework provides rigorous statistical tests to validate
strategy effectiveness and avoid false positives.
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
from .backtest_engine import BacktestResult
from .performance_calculator import PerformanceReport
from .walk_forward import WalkForwardResult
from .monte_carlo import MonteCarloResults


@dataclass
class ValidationConfig:
    """Configuration for statistical validation"""
    
    # Significance levels
    significance_level: float = 0.05  # 5% significance level
    confidence_level: float = 0.95    # 95% confidence
    
    # Bootstrap parameters
    bootstrap_samples: int = 10000
    block_bootstrap: bool = True
    block_size: int = 20
    
    # Hypothesis tests
    test_sharpe_ratio: bool = True
    test_alpha: bool = True
    test_beta: bool = True
    test_drawdown: bool = True
    
    # Multiple testing correction
    multiple_testing_correction: bool = True
    correction_method: str = "bonferroni"  # bonferroni, holm, benjamini_hochberg
    
    # Benchmark comparison
    benchmark_tests: bool = True
    benchmark_returns: Optional[pd.Series] = None
    
    # Robustness checks
    out_of_sample_test: bool = True
    parameter_stability_test: bool = True
    regime_analysis: bool = True
    
    # Power analysis
    power_analysis: bool = True
    min_effect_size: float = 0.1  # Minimum effect size to detect


@dataclass
class HypothesisTest:
    """Results of a hypothesis test"""
    
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float]
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: float
    power: Optional[float]
    interpretation: str


@dataclass
class ValidationResults:
    """Comprehensive validation results"""
    
    strategy_name: str
    validation_date: datetime
    
    # Hypothesis tests
    hypothesis_tests: List[HypothesisTest]
    
    # Overall assessment
    is_validated: bool
    confidence_level: float
    
    # Statistical metrics
    statistical_metrics: Dict[str, float]
    
    # Robustness checks
    robustness_results: Dict[str, Any]
    
    # Risk assessment
    risk_assessment: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    sample_size: int
    data_period: Tuple[datetime, datetime]
    validation_config: ValidationConfig


class StatisticalValidator:
    """
    Statistical Strategy Validator
    
    Educational Note:
    This validator implements rigorous statistical tests to determine
    if a strategy's performance is statistically significant or could
    be due to random chance. It uses established statistical methods
    from academic finance literature.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
    
    def validate_strategy(
        self,
        result: BacktestResult,
        walk_forward_result: Optional[WalkForwardResult] = None,
        monte_carlo_results: Optional[MonteCarloResults] = None
    ) -> ValidationResults:
        """Perform comprehensive statistical validation"""
        
        print(f"Performing statistical validation for {result.strategy_name}")
        
        hypothesis_tests = []
        
        # Test Sharpe ratio significance
        if self.config.test_sharpe_ratio:
            test = self._test_sharpe_significance(result)
            hypothesis_tests.append(test)
        
        # Test alpha significance
        if self.config.test_alpha and self.config.benchmark_returns is not None:
            test = self._test_alpha_significance(result)
            hypothesis_tests.append(test)
        
        # Test beta significance
        if self.config.test_beta and self.config.benchmark_returns is not None:
            test = self._test_beta_significance(result)
            hypothesis_tests.append(test)
        
        # Test drawdown significance
        if self.config.test_drawdown:
            test = self._test_drawdown_significance(result)
            hypothesis_tests.append(test)
        
        # Apply multiple testing correction
        if self.config.multiple_testing_correction:
            hypothesis_tests = self._apply_multiple_testing_correction(hypothesis_tests)
        
        # Perform robustness checks
        robustness_results = self._perform_robustness_checks(
            result, walk_forward_result, monte_carlo_results
        )
        
        # Calculate statistical metrics
        statistical_metrics = self._calculate_statistical_metrics(result)
        
        # Risk assessment
        risk_assessment = self._assess_risk(result, hypothesis_tests)
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            hypothesis_tests, robustness_results, risk_assessment
        )
        
        # Overall validation assessment
        is_validated = self._assess_overall_validation(hypothesis_tests, robustness_results)
        
        return ValidationResults(
            strategy_name=result.strategy_name,
            validation_date=datetime.now(),
            hypothesis_tests=hypothesis_tests,
            is_validated=is_validated,
            confidence_level=self.config.confidence_level,
            statistical_metrics=statistical_metrics,
            robustness_results=robustness_results,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            sample_size=len(result.returns),
            data_period=(result.start_date, result.end_date),
            validation_config=self.config
        )
    
    def _test_sharpe_significance(self, result: BacktestResult) -> HypothesisTest:
        """Test if Sharpe ratio is significantly different from zero"""
        
        returns = result.returns
        n = len(returns)
        
        if n < 2:
            return HypothesisTest(
                test_name="Sharpe Ratio Significance",
                null_hypothesis="Sharpe ratio = 0 (no skill)",
                alternative_hypothesis="Sharpe ratio ≠ 0 (skill present)",
                test_statistic=0,
                p_value=1.0,
                critical_value=None,
                is_significant=False,
                confidence_interval=None,
                effect_size=0,
                power=None,
                interpretation="Insufficient data for testing"
            )
        
        # Calculate test statistic
        sharpe_observed = result.sharpe_ratio
        
        # Standard error of Sharpe ratio (Jobson & Korkie approximation)
        sr_std = np.sqrt((1 + 0.5 * sharpe_observed**2) / n)
        
        # Test statistic
        test_statistic = sharpe_observed / sr_std
        
        # Two-sided test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Critical value
        critical_value = stats.norm.ppf(1 - self.config.significance_level / 2)
        
        # Confidence interval
        margin = critical_value * sr_std
        confidence_interval = (sharpe_observed - margin, sharpe_observed + margin)
        
        # Effect size (Cohen's d for Sharpe ratio)
        effect_size = abs(test_statistic)
        
        # Power calculation
        power = self._calculate_test_power(effect_size, n, alpha=self.config.significance_level)
        
        # Interpretation
        if p_value < self.config.significance_level:
            interpretation = f"Sharpe ratio is statistically significant (p={p_value:.4f})"
        else:
            interpretation = f"Sharpe ratio is not statistically significant (p={p_value:.4f})"
        
        return HypothesisTest(
            test_name="Sharpe Ratio Significance",
            null_hypothesis="Sharpe ratio = 0 (no skill)",
            alternative_hypothesis="Sharpe ratio ≠ 0 (skill present)",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=p_value < self.config.significance_level,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation
        )
    
    def _test_alpha_significance(self, result: BacktestResult) -> HypothesisTest:
        """Test if alpha is significantly different from zero"""
        
        if self.config.benchmark_returns is None:
            return HypothesisTest(
                test_name="Alpha Significance",
                null_hypothesis="Alpha = 0",
                alternative_hypothesis="Alpha ≠ 0",
                test_statistic=0,
                p_value=1.0,
                critical_value=None,
                is_significant=False,
                confidence_interval=None,
                effect_size=0,
                power=None,
                interpretation="No benchmark data available"
            )
        
        returns = result.returns
        benchmark_returns = self.config.benchmark_returns
        
        # Ensure same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[:min_length]
        benchmark_returns = benchmark_returns.iloc[:min_length]
        
        # Calculate alpha and beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        strategy_excess = returns.mean() - risk_free_rate
        benchmark_excess = benchmark_returns.mean() - risk_free_rate
        alpha = strategy_excess - beta * benchmark_excess
        
        # Standard error of alpha
        residuals = returns - beta * benchmark_returns
        alpha_std = np.std(residuals) / np.sqrt(len(returns))
        
        # Test statistic
        test_statistic = alpha / alpha_std if alpha_std > 0 else 0
        
        # Two-sided test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Critical value
        critical_value = stats.norm.ppf(1 - self.config.significance_level / 2)
        
        # Confidence interval
        margin = critical_value * alpha_std
        confidence_interval = (alpha - margin, alpha + margin)
        
        # Effect size
        effect_size = abs(test_statistic)
        
        # Power calculation
        power = self._calculate_test_power(effect_size, len(returns), alpha=self.config.significance_level)
        
        # Interpretation
        if p_value < self.config.significance_level:
            interpretation = f"Alpha is statistically significant (p={p_value:.4f})"
        else:
            interpretation = f"Alpha is not statistically significant (p={p_value:.4f})"
        
        return HypothesisTest(
            test_name="Alpha Significance",
            null_hypothesis="Alpha = 0",
            alternative_hypothesis="Alpha ≠ 0",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=p_value < self.config.significance_level,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation
        )
    
    def _test_beta_significance(self, result: BacktestResult) -> HypothesisTest:
        """Test if beta is significantly different from zero"""
        
        if self.config.benchmark_returns is None:
            return HypothesisTest(
                test_name="Beta Significance",
                null_hypothesis="Beta = 0",
                alternative_hypothesis="Beta ≠ 0",
                test_statistic=0,
                p_value=1.0,
                critical_value=None,
                is_significant=False,
                confidence_interval=None,
                effect_size=0,
                power=None,
                interpretation="No benchmark data available"
            )
        
        returns = result.returns
        benchmark_returns = self.config.benchmark_returns
        
        # Ensure same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[:min_length]
        benchmark_returns = benchmark_returns.iloc[:min_length]
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Standard error of beta
        residuals = returns - beta * benchmark_returns
        beta_std = np.std(residuals) / (np.std(benchmark_returns) * np.sqrt(len(returns)))
        
        # Test statistic
        test_statistic = beta / beta_std if beta_std > 0 else 0
        
        # Two-sided test
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Critical value
        critical_value = stats.norm.ppf(1 - self.config.significance_level / 2)
        
        # Confidence interval
        margin = critical_value * beta_std
        confidence_interval = (beta - margin, beta + margin)
        
        # Effect size
        effect_size = abs(test_statistic)
        
        # Power calculation
        power = self._calculate_test_power(effect_size, len(returns), alpha=self.config.significance_level)
        
        # Interpretation
        if p_value < self.config.significance_level:
            interpretation = f"Beta is statistically significant (p={p_value:.4f})"
        else:
            interpretation = f"Beta is not statistically significant (p={p_value:.4f})"
        
        return HypothesisTest(
            test_name="Beta Significance",
            null_hypothesis="Beta = 0",
            alternative_hypothesis="Beta ≠ 0",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=p_value < self.config.significance_level,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation
        )
    
    def _test_drawdown_significance(self, result: BacktestResult) -> HypothesisTest:
        """Test if maximum drawdown is statistically significant"""
        
        returns = result.returns
        n = len(returns)
        
        if n < 10:
            return HypothesisTest(
                test_name="Drawdown Significance",
                null_hypothesis="Max drawdown is from random variation",
                alternative_hypothesis="Max drawdown is extreme",
                test_statistic=0,
                p_value=1.0,
                critical_value=None,
                is_significant=False,
                confidence_interval=None,
                effect_size=0,
                power=None,
                interpretation="Insufficient data for testing"
            )
        
        # Bootstrap test for drawdown significance
        max_dd_observed = result.max_drawdown
        
        # Generate bootstrap samples
        bootstrap_max_dds = []
        
        for _ in range(self.config.bootstrap_samples):
            if self.config.block_bootstrap:
                # Block bootstrap to preserve autocorrelation
                bootstrap_returns = self._block_bootstrap(returns, self.config.block_size)
            else:
                # Simple bootstrap
                bootstrap_returns = np.random.choice(returns, size=n, replace=True)
            
            # Calculate equity curve and max drawdown
            equity_curve = np.cumprod(1 + bootstrap_returns)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            bootstrap_max_dds.append(drawdown.min())
        
        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_max_dds) <= max_dd_observed)
        
        # Test statistic (normalized drawdown)
        test_statistic = max_dd_observed / np.std(bootstrap_max_dds) if np.std(bootstrap_max_dds) > 0 else 0
        
        # Critical value (from bootstrap distribution)
        critical_value = np.percentile(bootstrap_max_dds, self.config.significance_level * 100)
        
        # Confidence interval
        confidence_interval = (
            np.percentile(bootstrap_max_dds, (1 - self.config.confidence_level) * 100),
            np.percentile(bootstrap_max_dds, self.config.confidence_level * 100)
        )
        
        # Effect size
        effect_size = abs(test_statistic)
        
        # Power calculation (simplified)
        power = self._calculate_test_power(effect_size, n, alpha=self.config.significance_level)
        
        # Interpretation
        if p_value < self.config.significance_level:
            interpretation = f"Max drawdown is statistically significant (p={p_value:.4f})"
        else:
            interpretation = f"Max drawdown is not statistically significant (p={p_value:.4f})"
        
        return HypothesisTest(
            test_name="Drawdown Significance",
            null_hypothesis="Max drawdown is from random variation",
            alternative_hypothesis="Max drawdown is extreme",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=p_value < self.config.significance_level,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation
        )
    
    def _block_bootstrap(self, returns: pd.Series, block_size: int) -> np.ndarray:
        """Perform block bootstrap to preserve autocorrelation"""
        
        n = len(returns)
        num_blocks = n // block_size
        
        # Create blocks
        blocks = []
        for i in range(n - block_size + 1):
            blocks.append(returns.iloc[i:i+block_size].values)
        
        # Randomly sample blocks
        selected_blocks = np.random.choice(len(blocks), size=num_blocks, replace=True)
        bootstrap_returns = np.concatenate([blocks[i] for i in selected_blocks])
        
        # Ensure correct length
        return bootstrap_returns[:n]
    
    def _apply_multiple_testing_correction(
        self,
        hypothesis_tests: List[HypothesisTest]
    ) -> List[HypothesisTest]:
        """Apply multiple testing correction to p-values"""
        
        if not hypothesis_tests:
            return hypothesis_tests
        
        # Extract p-values
        p_values = [test.p_value for test in hypothesis_tests]
        
        # Apply correction
        if self.config.correction_method == "bonferroni":
            corrected_p_values = [min(p * len(p_values), 1.0) for p in p_values]
        elif self.config.correction_method == "holm":
            corrected_p_values = self._holm_correction(p_values)
        elif self.config.correction_method == "benjamini_hochberg":
            corrected_p_values = self._benjamini_hochberg_correction(p_values)
        else:
            corrected_p_values = p_values
        
        # Update tests with corrected p-values
        for i, test in enumerate(hypothesis_tests):
            test.p_value = corrected_p_values[i]
            test.is_significant = corrected_p_values[i] < self.config.significance_level
            
            # Update interpretation
            if test.is_significant:
                test.interpretation = f"{test.test_name} is significant after correction (p={corrected_p_values[i]:.4f})"
            else:
                test.interpretation = f"{test.test_name} is not significant after correction (p={corrected_p_values[i]:.4f})"
        
        return hypothesis_tests
    
    def _holm_correction(self, p_values: List[float]) -> List[float]:
        """Holm-Bonferroni correction"""
        
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        corrected_p = p_values.copy()
        
        for rank, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_values[idx] * (len(p_values) - rank), 1.0)
        
        return corrected_p
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Benjamini-Hochberg false discovery rate correction"""
        
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        corrected_p = p_values.copy()
        
        for rank, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_values[idx] * len(p_values) / (rank + 1), 1.0)
        
        return corrected_p
    
    def _calculate_test_power(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> float:
        """Calculate statistical power of test"""
        
        # Simplified power calculation for two-sided test
        # Using normal approximation
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * np.sqrt(sample_size) - z_alpha
        
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def _perform_robustness_checks(
        self,
        result: BacktestResult,
        walk_forward_result: Optional[WalkForwardResult],
        monte_carlo_results: Optional[MonteCarloResults]
    ) -> Dict[str, Any]:
        """Perform robustness checks"""
        
        robustness_results = {}
        
        # Out-of-sample test
        if self.config.out_of_sample_test:
            robustness_results['out_of_sample'] = self._test_out_of_sample_robustness(result)
        
        # Parameter stability test
        if self.config.parameter_stability_test and walk_forward_result:
            robustness_results['parameter_stability'] = self._test_parameter_stability(walk_forward_result)
        
        # Regime analysis
        if self.config.regime_analysis:
            robustness_results['regime_analysis'] = self._analyze_regime_robustness(result)
        
        # Monte Carlo robustness
        if monte_carlo_results:
            robustness_results['monte_carlo'] = self._test_monte_carlo_robustness(monte_carlo_results)
        
        return robustness_results
    
    def _test_out_of_sample_robustness(self, result: BacktestResult) -> Dict[str, Any]:
        """Test out-of-sample robustness"""
        
        returns = result.returns
        n = len(returns)
        
        if n < 100:  # Need sufficient data
            return {"sufficient_data": False}
        
        # Split data (70% in-sample, 30% out-of-sample)
        split_point = int(n * 0.7)
        in_sample_returns = returns.iloc[:split_point]
        out_sample_returns = returns.iloc[split_point:]
        
        # Calculate performance metrics
        in_sample_sharpe = (in_sample_returns.mean() * 252) / (in_sample_returns.std() * np.sqrt(252))
        out_sample_sharpe = (out_sample_returns.mean() * 252) / (out_sample_returns.std() * np.sqrt(252))
        
        # Performance degradation
        degradation = (out_sample_sharpe - in_sample_sharpe) / abs(in_sample_sharpe) if in_sample_sharpe != 0 else 0
        
        return {
            "sufficient_data": True,
            "in_sample_sharpe": in_sample_sharpe,
            "out_sample_sharpe": out_sample_sharpe,
            "performance_degradation": degradation,
            "is_robust": abs(degradation) < 0.3  # Less than 30% degradation
        }
    
    def _test_parameter_stability(self, walk_forward_result: WalkForwardResult) -> Dict[str, Any]:
        """Test parameter stability"""
        
        if not walk_forward_result.parameter_stability:
            return {"has_parameters": False}
        
        # Average stability across all parameters
        avg_stability = np.mean(list(walk_forward_result.parameter_stability.values()))
        
        # Count stable parameters (stability > 0.7)
        stable_params = sum(1 for s in walk_forward_result.parameter_stability.values() if s > 0.7)
        total_params = len(walk_forward_result.parameter_stability)
        
        return {
            "has_parameters": True,
            "average_stability": avg_stability,
            "stable_parameters": stable_params,
            "total_parameters": total_params,
            "stability_ratio": stable_params / total_params if total_params > 0 else 0,
            "is_stable": avg_stability > 0.7
        }
    
    def _analyze_regime_robustness(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        
        if not hasattr(result, 'regime_performance') or not result.regime_performance:
            return {"has_regime_data": False}
        
        regime_performance = result.regime_performance
        
        # Calculate performance consistency
        sharpe_ratios = [perf.get('sharpe_ratio', 0) for perf in regime_performance.values()]
        
        if not sharpe_ratios:
            return {"has_regime_data": False}
        
        avg_sharpe = np.mean(sharpe_ratios)
        sharpe_std = np.std(sharpe_ratios)
        consistency = 1 - (sharpe_std / abs(avg_sharpe)) if avg_sharpe != 0 else 0
        
        # Count positive performing regimes
        positive_regimes = sum(1 for perf in regime_performance.values() if perf.get('sharpe_ratio', 0) > 0)
        
        return {
            "has_regime_data": True,
            "average_sharpe": avg_sharpe,
            "sharpe_volatility": sharpe_std,
            "consistency": consistency,
            "positive_regimes": positive_regimes,
            "total_regimes": len(regime_performance),
            "is_consistent": consistency > 0.5
        }
    
    def _test_monte_carlo_robustness(self, monte_carlo_results: MonteCarloResults) -> Dict[str, Any]:
        """Test Monte Carlo robustness"""
        
        # Calculate probability of positive performance
        positive_returns = [sim for sim in monte_carlo_results.simulations if sim.total_return > 0]
        prob_positive = len(positive_returns) / len(monte_carlo_results.simulations)
        
        # Calculate probability of beating benchmark
        prob_beat_benchmark = 0.5  # Simplified
        
        # Calculate consistency (low coefficient of variation)
        returns = [sim.total_return for sim in monte_carlo_results.simulations]
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        consistency = 1 - (return_std / abs(mean_return)) if mean_return != 0 else 0
        
        return {
            "probability_positive": prob_positive,
            "probability_beat_benchmark": prob_beat_benchmark,
            "return_consistency": consistency,
            "is_robust": prob_positive > 0.6 and consistency > 0.5
        }
    
    def _calculate_statistical_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate additional statistical metrics"""
        
        returns = result.returns
        
        metrics = {
            "sample_size": len(returns),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "jarque_bera_stat": 0,  # Would calculate Jarque-Bera test
            "jarque_bera_pvalue": 1,
            "autocorrelation_lag1": returns.autocorr(lag=1),
            "ljung_box_stat": 0,  # Would calculate Ljung-Box test
            "ljung_box_pvalue": 1
        }
        
        # Jarque-Bera test for normality
        if len(returns) > 7:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            metrics["jarque_bera_stat"] = jb_stat
            metrics["jarque_bera_pvalue"] = jb_pvalue
        
        # Ljung-Box test for autocorrelation
        if len(returns) > 10:
            lb_stat, lb_pvalue = stats.acorr_ljungbox(returns, lags=10, return_df=False)
            metrics["ljung_box_stat"] = lb_stat[0]
            metrics["ljung_box_pvalue"] = lb_pvalue[0]
        
        return metrics
    
    def _assess_risk(
        self,
        result: BacktestResult,
        hypothesis_tests: List[HypothesisTest]
    ) -> Dict[str, Any]:
        """Assess risk characteristics"""
        
        risk_assessment = {
            "max_drawdown": result.max_drawdown,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "volatility": result.volatility,
            "risk_level": "Medium"  # Would calculate based on metrics
        }
        
        # Classify risk level
        if result.max_drawdown < -0.1 and result.volatility < 0.15:
            risk_assessment["risk_level"] = "Low"
        elif result.max_drawdown < -0.25 and result.volatility < 0.25:
            risk_assessment["risk_level"] = "Medium"
        else:
            risk_assessment["risk_level"] = "High"
        
        # Statistical risk (failed tests)
        failed_tests = [test for test in hypothesis_tests if not test.is_significant and test.test_name != "Drawdown Significance"]
        risk_assessment["statistical_risk"] = len(failed_tests) / len(hypothesis_tests) if hypothesis_tests else 0
        
        return risk_assessment
    
    def _generate_validation_recommendations(
        self,
        hypothesis_tests: List[HypothesisTest],
        robustness_results: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate validation recommendations"""
        
        recommendations = []
        
        # Check hypothesis tests
        significant_tests = [test for test in hypothesis_tests if test.is_significant]
        
        if len(significant_tests) == 0:
            recommendations.append("No statistically significant results - strategy may be due to random chance")
        elif len(significant_tests) < len(hypothesis_tests) / 2:
            recommendations.append("Mixed statistical significance - consider strategy refinement")
        else:
            recommendations.append("Strong statistical significance - strategy shows genuine skill")
        
        # Check robustness
        if robustness_results.get("out_of_sample", {}).get("is_robust", False):
            recommendations.append("Good out-of-sample performance - strategy appears robust")
        else:
            recommendations.append("Poor out-of-sample performance - strategy may be overfitted")
        
        if robustness_results.get("parameter_stability", {}).get("is_stable", False):
            recommendations.append("Parameters are stable across periods - good robustness")
        else:
            recommendations.append("Parameters are unstable - consider fixing or simplifying strategy")
        
        # Check risk
        if risk_assessment["risk_level"] == "High":
            recommendations.append("High risk level - implement stronger risk controls")
        elif risk_assessment["risk_level"] == "Low":
            recommendations.append("Low risk level - strategy appears conservative")
        
        return recommendations
    
    def _assess_overall_validation(
        self,
        hypothesis_tests: List[HypothesisTest],
        robustness_results: Dict[str, Any]
    ) -> bool:
        """Assess overall validation status"""
        
        # Check if key tests are significant
        key_tests = ["Sharpe Ratio Significance", "Alpha Significance"]
        significant_key_tests = [
            test for test in hypothesis_tests 
            if test.test_name in key_tests and test.is_significant
        ]
        
        if len(significant_key_tests) < len(key_tests) / 2:
            return False
        
        # Check robustness
        robustness_score = 0
        total_checks = 0
        
        if "out_of_sample" in robustness_results:
            total_checks += 1
            if robustness_results["out_of_sample"].get("is_robust", False):
                robustness_score += 1
        
        if "parameter_stability" in robustness_results:
            total_checks += 1
            if robustness_results["parameter_stability"].get("is_stable", False):
                robustness_score += 1
        
        if "monte_carlo" in robustness_results:
            total_checks += 1
            if robustness_results["monte_carlo"].get("is_robust", False):
                robustness_score += 1
        
        if total_checks > 0 and robustness_score / total_checks < 0.5:
            return False
        
        return True


def explain_statistical_validation():
    """
    Educational explanation of statistical validation
    """
    
    print("=== Statistical Validation Educational Guide ===\n")
    
    concepts = {
        'Hypothesis Testing': "Statistical method to determine if results are due to chance or genuine effect",
        
        'Null Hypothesis': "Default assumption that there is no effect or relationship",
        
        'P-value': "Probability of observing results as extreme as those observed, assuming null hypothesis is true",
        
        'Significance Level': "Threshold for rejecting null hypothesis (typically 5%)",
        
        'Confidence Interval': "Range of values likely to contain the true parameter",
        
        'Effect Size': "Magnitude of the difference or relationship",
        
        'Statistical Power': "Probability of detecting an effect when it exists",
        
        'Multiple Testing Correction': "Adjustment for testing multiple hypotheses simultaneously",
        
        'Bootstrap Method': "Resampling technique to estimate sampling distribution",
        
        'Out-of-Sample Testing': "Testing on data not used for model development"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Statistical Validation Best Practices ===")
    practices = [
        "1. Always use appropriate statistical tests for your data",
        "2. Apply multiple testing correction when testing multiple hypotheses",
        "3. Use out-of-sample data to validate strategy performance",
        "4. Consider effect size, not just statistical significance",
        "5. Ensure sufficient sample size for reliable results",
        "6. Test assumptions of statistical methods",
        "7. Use bootstrap methods for non-parametric inference",
        "8. Validate across different market regimes",
        "9. Consider both statistical and practical significance",
        "10. Document all validation procedures and results"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Common Statistical Pitfalls ===")
    pitfalls = [
        "• P-hacking: Testing many hypotheses until finding significance",
        "• Overfitting: Creating models that work too well on historical data",
        "• Data mining: Finding spurious patterns in large datasets",
        "• Ignoring multiple testing corrections",
        "• Confusing statistical significance with practical importance",
        "• Using insufficient sample sizes",
        "• Violating test assumptions",
        "• Not validating out-of-sample",
        "• Ignoring effect sizes",
        "• Selection bias in data selection"
    ]
    
    for pitfall in pitfalls:
        print(pitfall)


if __name__ == "__main__":
    # Example usage
    explain_statistical_validation()
    
    print("\n=== Statistical Validation Example ===")
    print("To use the statistical validator:")
    print("1. Create validation configuration with significance levels")
    print("2. Perform hypothesis tests for key performance metrics")
    print("3. Apply multiple testing corrections")
    print("4. Conduct robustness checks")
    print("5. Generate comprehensive validation report")
    
    # Example configuration
    config = ValidationConfig(
        significance_level=0.05,
        confidence_level=0.95,
        bootstrap_samples=10000,
        multiple_testing_correction=True,
        correction_method="bonferroni",
        out_of_sample_test=True,
        parameter_stability_test=True,
        regime_analysis=True
    )
    
    print(f"\nSample configuration:")
    print(f"  Significance level: {config.significance_level}")
    print(f"  Confidence level: {config.confidence_level}")
    print(f"  Bootstrap samples: {config.bootstrap_samples}")
    print(f"  Multiple testing correction: {config.multiple_testing_correction}")
    print(f"  Out-of-sample test: {config.out_of_sample_test}")
    
    print("\nThe validator provides:")
    print("• Rigorous hypothesis testing")
    print("• Multiple testing corrections")
    print("• Bootstrap validation methods")
    print("• Robustness checks")
    print("• Statistical power analysis")
    print("• Comprehensive validation reports")
    print("• Evidence-based recommendations")