"""
Visualization System - AI Trading System

This module implements comprehensive visualization tools for backtesting
results, performance analysis, and strategy evaluation. It provides
publication-ready charts and interactive dashboards.

Educational Note:
Good visualization is essential for understanding strategy performance.
Charts help identify patterns, spot issues, and communicate results
effectively. This system provides institutional-quality visualizations
that would be suitable for investment committees or client presentations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting components
from .backtest_engine import BacktestResult
from .performance_calculator import PerformanceReport
from .walk_forward import WalkForwardResult
from .monte_carlo import MonteCarloResults

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ChartConfig:
    """Configuration for chart styling"""
    
    # Figure settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    
    # Color settings
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    positive_color: str = "#2ECC71"
    negative_color: str = "#E74C3C"
    benchmark_color: str = "#95A5A6"
    
    # Font settings
    title_fontsize: int = 16
    label_fontsize: int = 12
    legend_fontsize: int = 10
    
    # Grid settings
    grid_alpha: float = 0.3
    grid_linestyle: str = "--"
    
    # Save settings
    save_format: str = "png"
    save_dpi: int = 300
    transparent: bool = False


class PerformanceVisualizer:
    """
    Performance Visualization Tools
    
    Educational Note:
    This visualizer creates professional charts for analyzing strategy
    performance. Good visualizations help identify strengths and
    weaknesses in strategies that might not be obvious from numbers
    alone.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.label_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.label_fontsize,
            'ytick.labelsize': self.config.label_fontsize,
            'legend.fontsize': self.config.legend_fontsize,
            'axes.grid': True,
            'grid.alpha': self.config.grid_alpha,
            'grid.linestyle': self.config.grid_linestyle
        })
    
    def plot_equity_curve(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot equity curve with optional benchmark"""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Plot strategy equity curve
        ax.plot(result.equity_curve.index, result.equity_curve.values,
                label=f"{result.strategy_name} Strategy", 
                color=self.config.primary_color, linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_equity = (1 + benchmark_returns).cumprod() * result.config.initial_cash
            benchmark_equity.index = result.equity_curve.index[:len(benchmark_equity)]
            ax.plot(benchmark_equity.index, benchmark_equity.values,
                    label="Benchmark", color=self.config.benchmark_color, 
                    linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_title(f"Equity Curve - {result.strategy_name}", fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel("Date", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Portfolio Value ($)", fontsize=self.config.label_fontsize)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_drawdown(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot underwater curve (drawdown)"""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Calculate drawdown
        peak = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - peak) / peak * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0,
                       color=self.config.negative_color, alpha=0.3)
        ax.plot(drawdown.index, drawdown.values,
                color=self.config.negative_color, linewidth=1)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_title(f"Drawdown Analysis - {result.strategy_name}", 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel("Date", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Drawdown (%)", fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add max drawdown annotation
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_value:.1f}%',
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_returns_distribution(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot returns distribution"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(result.returns * 100, bins=50, alpha=0.7, 
                color=self.config.primary_color, edgecolor='black')
        ax1.axvline(result.returns.mean() * 100, color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
        
        ax1.set_title("Returns Distribution", fontweight='bold')
        ax1.set_xlabel("Daily Return (%)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(result.returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)", fontweight='bold')
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_rolling_metrics(
        self,
        result: BacktestResult,
        window: int = 63,  # 3 months
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot rolling performance metrics"""
        
        if len(result.returns) < window:
            print(f"Not enough data for rolling metrics with window {window}")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rolling Sharpe ratio
        rolling_sharpe = result.returns.rolling(window=window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, 
                color=self.config.primary_color, linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title(f"Rolling Sharpe Ratio ({window} days)", fontweight='bold')
        ax1.set_ylabel("Sharpe Ratio")
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Rolling volatility
        rolling_vol = result.returns.rolling(window=window).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol.values, 
                color=self.config.secondary_color, linewidth=2)
        ax2.set_title(f"Rolling Volatility ({window} days)", fontweight='bold')
        ax2.set_ylabel("Volatility (%)")
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        # Rolling returns
        rolling_returns = result.returns.rolling(window=window).mean() * 252 * 100
        ax3.plot(rolling_returns.index, rolling_returns.values, 
                color=self.config.positive_color, linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title(f"Rolling Annualized Return ({window} days)", fontweight='bold')
        ax3.set_ylabel("Annualized Return (%)")
        ax3.grid(True, alpha=self.config.grid_alpha)
        
        # Rolling max drawdown
        rolling_equity = result.equity_curve
        rolling_peak = rolling_equity.rolling(window=window).max()
        rolling_dd = (rolling_equity - rolling_peak) / rolling_peak * 100
        ax4.fill_between(rolling_dd.index, rolling_dd.values, 0,
                        color=self.config.negative_color, alpha=0.3)
        ax4.plot(rolling_dd.index, rolling_dd.values, 
                color=self.config.negative_color, linewidth=1)
        ax4.set_title(f"Rolling Max Drawdown ({window} days)", fontweight='bold')
        ax4.set_ylabel("Drawdown (%)")
        ax4.grid(True, alpha=self.config.grid_alpha)
        
        # Format x-axis dates for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_monthly_returns(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot monthly returns heatmap"""
        
        # Calculate monthly returns
        monthly_returns = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create custom colormap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap=cmap, 
                   center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title(f"Monthly Returns Heatmap - {result.strategy_name}", 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig


class MonteCarloVisualizer:
    """
    Monte Carlo Simulation Visualization
    
    Educational Note:
    Monte Carlo visualizations help understand the range of possible
    outcomes and probabilities. They show uncertainty in strategy
    performance and help with risk management decisions.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
    
    def plot_simulation_paths(
        self,
        results: MonteCarloResults,
        num_paths: int = 100,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot Monte Carlo simulation paths"""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Sample paths to plot
        if len(results.simulations) > num_paths:
            sample_simulations = np.random.choice(results.simulations, num_paths, replace=False)
        else:
            sample_simulations = results.simulations
        
        # Plot individual paths
        for sim in sample_simulations:
            ax.plot(sim.equity_curve.index, sim.equity_curve.values,
                   alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot percentiles
        all_equity_curves = pd.DataFrame([sim.equity_curve.values for sim in results.simulations],
                                       index=[sim.equity_curve.index for sim in results.simulations]).T
        
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            percentile_values = all_equity_curves.quantile(p/100, axis=1)
            if p == 50:
                ax.plot(percentile_values.index, percentile_values.values,
                       label=f'{p}th percentile (median)', 
                       color='red', linewidth=2)
            else:
                ax.plot(percentile_values.index, percentile_values.values,
                       label=f'{p}th percentile', 
                       color='gray', linewidth=1, alpha=0.7)
        
        # Plot base result
        if results.base_result:
            ax.plot(results.base_result.equity_curve.index, 
                   results.base_result.equity_curve.values,
                   label='Base Strategy', color='green', linewidth=2)
        
        ax.set_title(f"Monte Carlo Simulation Paths - {results.strategy_name}", 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_distribution_comparison(
        self,
        results: MonteCarloResults,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution comparison of key metrics"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total return distribution
        total_returns = [sim.total_return for sim in results.simulations]
        ax1.hist(total_returns, bins=50, alpha=0.7, color=self.config.primary_color, edgecolor='black')
        ax1.axvline(np.mean(total_returns), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(results.base_result.total_return, color='green', linestyle='--', linewidth=2, label='Base')
        ax1.set_title("Total Return Distribution", fontweight='bold')
        ax1.set_xlabel("Total Return")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Sharpe ratio distribution
        sharpe_ratios = [sim.sharpe_ratio for sim in results.simulations]
        ax2.hist(sharpe_ratios, bins=50, alpha=0.7, color=self.config.secondary_color, edgecolor='black')
        ax2.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(results.base_result.sharpe_ratio, color='green', linestyle='--', linewidth=2, label='Base')
        ax2.set_title("Sharpe Ratio Distribution", fontweight='bold')
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        # Max drawdown distribution
        max_drawdowns = [sim.max_drawdown for sim in results.simulations]
        ax3.hist(max_drawdowns, bins=50, alpha=0.7, color=self.config.negative_color, edgecolor='black')
        ax3.axvline(np.mean(max_drawdowns), color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.axvline(results.base_result.max_drawdown, color='green', linestyle='--', linewidth=2, label='Base')
        ax3.set_title("Max Drawdown Distribution", fontweight='bold')
        ax3.set_xlabel("Max Drawdown")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=self.config.grid_alpha)
        
        # Win rate distribution
        win_rates = [sim.win_rate for sim in results.simulations]
        ax4.hist(win_rates, bins=50, alpha=0.7, color=self.config.positive_color, edgecolor='black')
        ax4.axvline(np.mean(win_rates), color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.axvline(results.base_result.win_rate, color='green', linestyle='--', linewidth=2, label='Base')
        ax4.set_title("Win Rate Distribution", fontweight='bold')
        ax4.set_xlabel("Win Rate")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_scenario_analysis(
        self,
        results: MonteCarloResults,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot performance by market scenario"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scenario performance boxplot
        scenario_data = []
        scenario_labels = []
        
        for scenario, sims in results.scenario_results.items():
            returns = [sim.total_return for sim in sims]
            scenario_data.extend(returns)
            scenario_labels.extend([scenario] * len(returns))
        
        scenario_df = pd.DataFrame({'return': scenario_data, 'scenario': scenario_labels})
        
        sns.boxplot(data=scenario_df, x='scenario', y='return', ax=ax1)
        ax1.set_title("Performance by Market Scenario", fontweight='bold')
        ax1.set_xlabel("Scenario")
        ax1.set_ylabel("Total Return")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Scenario count pie chart
        scenario_counts = {scenario: len(sims) for scenario, sims in results.scenario_results.items()}
        
        ax2.pie(scenario_counts.values(), labels=scenario_counts.keys(), autopct='%1.1f%%')
        ax2.set_title("Simulation Distribution by Scenario", fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig


class WalkForwardVisualizer:
    """
    Walk-Forward Analysis Visualization
    
    Educational Note:
    Walk-forward visualizations show how strategy performance
    varies across different time periods and how stable the
    parameters are. This is crucial for assessing strategy
    robustness.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
    
    def plot_performance_comparison(
        self,
        results: WalkForwardResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training vs testing performance comparison"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        periods = list(range(1, len(results.periods) + 1))
        
        # Sharpe ratio comparison
        training_sharpe = [p.training_performance.get('sharpe_ratio', 0) for p in results.periods]
        testing_sharpe = [p.testing_performance.get('sharpe_ratio', 0) for p in results.periods]
        
        ax1.plot(periods, training_sharpe, 'o-', label='Training', color=self.config.primary_color, linewidth=2)
        ax1.plot(periods, testing_sharpe, 'o-', label='Testing', color=self.config.secondary_color, linewidth=2)
        ax1.set_title("Sharpe Ratio: Training vs Testing", fontweight='bold')
        ax1.set_xlabel("Period")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Return comparison
        training_returns = [p.training_performance.get('total_return', 0) for p in results.periods]
        testing_returns = [p.testing_performance.get('total_return', 0) for p in results.periods]
        
        ax2.plot(periods, training_returns, 'o-', label='Training', color=self.config.primary_color, linewidth=2)
        ax2.plot(periods, testing_returns, 'o-', label='Testing', color=self.config.secondary_color, linewidth=2)
        ax2.set_title("Total Return: Training vs Testing", fontweight='bold')
        ax2.set_xlabel("Period")
        ax2.set_ylabel("Total Return")
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        # Max drawdown comparison
        training_dd = [p.training_performance.get('max_drawdown', 0) for p in results.periods]
        testing_dd = [p.testing_performance.get('max_drawdown', 0) for p in results.periods]
        
        ax3.plot(periods, training_dd, 'o-', label='Training', color=self.config.primary_color, linewidth=2)
        ax3.plot(periods, testing_dd, 'o-', label='Testing', color=self.config.secondary_color, linewidth=2)
        ax3.set_title("Max Drawdown: Training vs Testing", fontweight='bold')
        ax3.set_xlabel("Period")
        ax3.set_ylabel("Max Drawdown")
        ax3.legend()
        ax3.grid(True, alpha=self.config.grid_alpha)
        
        # Performance degradation
        performance_degradation = [(t - tr) / tr * 100 if tr != 0 else 0 
                                for tr, t in zip(training_sharpe, testing_sharpe)]
        
        colors = [self.config.positive_color if x >= 0 else self.config.negative_color for x in performance_degradation]
        ax4.bar(periods, performance_degradation, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title("Performance Degradation (Testing - Training)", fontweight='bold')
        ax4.set_xlabel("Period")
        ax4.set_ylabel("Degradation (%)")
        ax4.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig
    
    def plot_parameter_stability(
        self,
        results: WalkForwardResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot parameter stability over time"""
        
        if not results.parameter_evolution:
            print("No parameter evolution data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        parameters = list(results.parameter_evolution.keys())[:4]  # Plot first 4 parameters
        
        for i, param in enumerate(parameters):
            if i >= len(axes):
                break
                
            values = results.parameter_evolution[param]
            periods = list(range(1, len(values) + 1))
            
            axes[i].plot(periods, values, 'o-', color=self.config.primary_color, linewidth=2)
            axes[i].set_title(f"Parameter Evolution: {param}", fontweight='bold')
            axes[i].set_xlabel("Period")
            axes[i].set_ylabel("Parameter Value")
            axes[i].grid(True, alpha=self.config.grid_alpha)
            
            # Add stability score
            if param in results.parameter_stability:
                stability = results.parameter_stability[param]
                axes[i].text(0.02, 0.98, f"Stability: {stability:.2f}",
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for i in range(len(parameters), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, 
                       bbox_inches='tight', transparent=self.config.transparent)
        
        return fig


def explain_visualization():
    """
    Educational explanation of performance visualization
    """
    
    print("=== Performance Visualization Educational Guide ===\n")
    
    concepts = {
        'Equity Curve': "Shows portfolio value over time, the most fundamental performance chart",
        
        'Drawdown Chart': "Displays periods of portfolio decline from peak values",
        
        'Returns Distribution': "Histogram showing the frequency of different return levels",
        
        'Rolling Metrics': "Performance metrics calculated over rolling time windows",
        
        'Heatmap': "Color-coded matrix showing monthly returns patterns",
        
        'Monte Carlo Paths': "Multiple possible future outcomes from simulation",
        
        'Scenario Analysis': "Performance under different market conditions",
        
        'Parameter Stability': "How optimal parameters change over time",
        
        'Performance Attribution': "Sources of returns by different factors",
        
        'Risk-Return Scatter': "Visualizes the relationship between risk and return"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Visualization Best Practices ===")
    practices = [
        "1. Use consistent color schemes across charts",
        "2. Include appropriate labels and legends",
        "3. Choose the right chart type for the data",
        "4. Avoid chartjunk and unnecessary elements",
        "5. Use clear and descriptive titles",
        "6. Format axes appropriately (currency, percentages, dates)",
        "7. Include benchmark comparisons when relevant",
        "8. Add annotations for key insights",
        "9. Ensure charts are readable at different sizes",
        "10. Use accessibility-friendly color choices"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Interpreting Common Charts ===")
    interpretations = {
        "Equity Curve": "Look for consistent upward trend, volatility, and recovery patterns",
        "Drawdown Chart": "Identify maximum loss, recovery time, and drawdown frequency",
        "Returns Distribution": "Check for normality, fat tails, and skewness",
        "Rolling Sharpe": "Assess consistency of risk-adjusted performance over time",
        "Monthly Heatmap": "Identify seasonal patterns and good/bad months",
        "Monte Carlo Distribution": "Understand range of possible outcomes and probabilities"
    }
    
    for chart, interpretation in interpretations.items():
        print(f"{chart}:")
        print(f"  {interpretation}\n")


if __name__ == "__main__":
    # Example usage
    explain_visualization()
    
    print("\n=== Visualization System Example ===")
    print("To use the visualization system:")
    print("1. Create visualizer with custom styling if needed")
    print("2. Generate different types of charts for analysis")
    print("3. Combine charts into comprehensive dashboards")
    print("4. Save charts for reports and presentations")
    
    print("\nAvailable visualizers:")
    print("• PerformanceVisualizer - Basic performance charts")
    print("• MonteCarloVisualizer - Monte Carlo simulation charts")
    print("• WalkForwardVisualizer - Walk-forward analysis charts")
    
    print("\nKey chart types:")
    print("• Equity curves with benchmarks")
    print("• Drawdown analysis")
    print("• Returns distribution")
    print("• Rolling performance metrics")
    print("• Monthly returns heatmap")
    print("• Monte Carlo simulation paths")
    print("• Parameter stability analysis")
    print("• Scenario comparison charts")
    
    print("\nThe visualization system provides:")
    print("• Publication-ready chart quality")
    print("• Customizable styling and colors")
    print("• Multiple chart formats (PNG, PDF, SVG)")
    print("• Professional annotations and formatting")
    print("• Comprehensive chart library")