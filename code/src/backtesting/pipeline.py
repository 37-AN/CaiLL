"""
Complete Backtesting Pipeline - AI Trading System

This module integrates all backtesting components into a comprehensive
pipeline that provides end-to-end strategy validation, from basic
backtesting to advanced statistical analysis and visualization.

Educational Note:
A complete backtesting pipeline is essential for professional
strategy development. It ensures consistent methodology, proper
validation, and comprehensive analysis. This pipeline provides
institutional-grade strategy evaluation capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all backtesting components
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, TradingStrategy
from .walk_forward import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResults
from .performance_calculator import PerformanceCalculator, PerformanceReport
from .visualization import PerformanceVisualizer, MonteCarloVisualizer, WalkForwardVisualizer, ChartConfig
from .strategy_validation import StatisticalValidator, ValidationConfig, ValidationResults


@dataclass
class PipelineConfig:
    """Configuration for the complete backtesting pipeline"""
    
    # Basic backtesting
    backtest_config: BacktestConfig
    
    # Advanced analysis
    enable_walk_forward: bool = True
    walk_forward_config: Optional[WalkForwardConfig] = None
    
    enable_monte_carlo: bool = True
    monte_carlo_config: Optional[MonteCarloConfig] = None
    
    enable_validation: bool = True
    validation_config: Optional[ValidationConfig] = None
    
    # Visualization
    enable_visualization: bool = True
    chart_config: Optional[ChartConfig] = None
    save_charts: bool = True
    chart_output_dir: str = "./charts"
    
    # Reporting
    generate_report: bool = True
    report_format: str = "html"  # html, pdf, json
    report_output_dir: str = "./reports"
    
    # Performance
    use_parallel: bool = True
    max_workers: int = 4
    
    # Data
    benchmark_symbol: Optional[str] = None
    benchmark_data: Optional[pd.Series] = None
    
    # Output
    save_results: bool = True
    results_output_dir: str = "./results"
    verbose: bool = True


@dataclass
class PipelineResults:
    """Complete results from the backtesting pipeline"""
    
    strategy_name: str
    pipeline_config: PipelineConfig
    execution_time: float
    
    # Core results
    backtest_result: BacktestResult
    performance_report: PerformanceReport
    
    # Advanced analysis results
    walk_forward_result: Optional[WalkForwardResult] = None
    monte_carlo_results: Optional[MonteCarloResults] = None
    validation_results: Optional[ValidationResults] = None
    
    # Visualizations
    chart_paths: Dict[str, str] = field(default_factory=dict)
    
    # Summary
    overall_assessment: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    execution_date: datetime = field(default_factory=datetime.now)
    success: bool = True
    errors: List[str] = field(default_factory=list)


class BacktestingPipeline:
    """
    Complete Backtesting Pipeline
    
    Educational Note:
    This pipeline orchestrates all backtesting components to provide
    comprehensive strategy analysis. It ensures consistent methodology,
    proper validation, and professional-grade reporting. This is the
    tool you would use in a professional trading firm.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize components
        self.backtest_engine = BacktestEngine(config.backtest_config)
        self.performance_calculator = PerformanceCalculator()
        
        if config.enable_walk_forward:
            wf_config = config.walk_forward_config or self._create_default_walk_forward_config()
            self.walk_forward_analyzer = WalkForwardAnalyzer(wf_config)
        
        if config.enable_monte_carlo:
            mc_config = config.monte_carlo_config or self._create_default_monte_carlo_config()
            self.monte_carlo_simulator = MonteCarloSimulator(mc_config)
        
        if config.enable_validation:
            val_config = config.validation_config or self._create_default_validation_config()
            self.statistical_validator = StatisticalValidator(val_config)
        
        if config.enable_visualization:
            chart_config = config.chart_config or ChartConfig()
            self.performance_visualizer = PerformanceVisualizer(chart_config)
            self.monte_carlo_visualizer = MonteCarloVisualizer(chart_config)
            self.walk_forward_visualizer = WalkForwardVisualizer(chart_config)
        
        # Create output directories
        self._create_output_directories()
    
    async def run_complete_analysis(
        self,
        strategy_class: type,
        strategy_parameters: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> PipelineResults:
        """Run complete backtesting pipeline"""
        
        start_time = datetime.now()
        
        if self.config.verbose:
            print(f"🚀 Starting complete backtesting pipeline for {strategy_class.__name__}")
            print(f"⏰ Started at: {start_time}")
        
        results = PipelineResults(
            strategy_name=strategy_class.__name__,
            pipeline_config=self.config,
            execution_time=0
        )
        
        try:
            # Step 1: Basic backtesting
            if self.config.verbose:
                print("\n📊 Step 1: Basic Backtesting")
            
            strategy = strategy_class(f"{strategy_class.__name__}_base", strategy_parameters)
            results.backtest_result = await self.backtest_engine.run_backtest(strategy, data)
            
            if self.config.verbose:
                print(f"✅ Basic backtest completed")
                print(f"   Total Return: {results.backtest_result.total_return:.2%}")
                print(f"   Sharpe Ratio: {results.backtest_result.sharpe_ratio:.2f}")
                print(f"   Max Drawdown: {results.backtest_result.max_drawdown:.2%}")
            
            # Step 2: Performance analysis
            if self.config.verbose:
                print("\n📈 Step 2: Performance Analysis")
            
            results.performance_report = self.performance_calculator.calculate_performance(
                results.backtest_result,
                self.config.benchmark_data
            )
            
            if self.config.verbose:
                print(f"✅ Performance analysis completed")
                print(f"   Risk-adjusted metrics calculated")
                print(f"   Attribution analysis completed")
            
            # Step 3: Walk-forward analysis
            if self.config.enable_walk_forward:
                if self.config.verbose:
                    print("\n🔄 Step 3: Walk-Forward Analysis")
                
                results.walk_forward_result = await self.walk_forward_analyzer.run_analysis(
                    strategy_class, data
                )
                
                if self.config.verbose:
                    print(f"✅ Walk-forward analysis completed")
                    print(f"   Hit rate: {results.walk_forward_result.hit_rate:.1%}")
                    print(f"   Parameter stability analyzed")
            
            # Step 4: Monte Carlo simulation
            if self.config.enable_monte_carlo:
                if self.config.verbose:
                    print("\n🎲 Step 4: Monte Carlo Simulation")
                
                results.monte_carlo_results = await self.monte_carlo_simulator.run_simulation(
                    strategy_class, strategy_parameters, results.backtest_result, data
                )
                
                if self.config.verbose:
                    print(f"✅ Monte Carlo simulation completed")
                    print(f"   Simulations: {len(results.monte_carlo_results.simulations)}")
                    print(f"   Probability analysis completed")
            
            # Step 5: Statistical validation
            if self.config.enable_validation:
                if self.config.verbose:
                    print("\n🔬 Step 5: Statistical Validation")
                
                results.validation_results = self.statistical_validator.validate_strategy(
                    results.backtest_result,
                    results.walk_forward_result,
                    results.monte_carlo_results
                )
                
                if self.config.verbose:
                    print(f"✅ Statistical validation completed")
                    print(f"   Strategy validated: {results.validation_results.is_validated}")
                    print(f"   Significant tests: {sum(1 for t in results.validation_results.hypothesis_tests if t.is_significant)}")
            
            # Step 6: Visualization
            if self.config.enable_visualization:
                if self.config.verbose:
                    print("\n📊 Step 6: Generating Visualizations")
                
                await self._generate_visualizations(results)
                
                if self.config.verbose:
                    print(f"✅ Visualizations generated")
                    print(f"   Charts saved: {len(results.chart_paths)}")
            
            # Step 7: Overall assessment
            if self.config.verbose:
                print("\n🎯 Step 7: Overall Assessment")
            
            results.overall_assessment = self._generate_overall_assessment(results)
            results.recommendations = self._generate_pipeline_recommendations(results)
            
            if self.config.verbose:
                print(f"✅ Overall assessment completed")
                print(f"   Overall score: {results.overall_assessment.get('score', 0):.1f}/10")
                print(f"   Recommendations: {len(results.recommendations)}")
            
            # Step 8: Save results
            if self.config.save_results:
                if self.config.verbose:
                    print("\n💾 Step 8: Saving Results")
                
                await self._save_results(results)
                
                if self.config.verbose:
                    print(f"✅ Results saved")
            
            # Calculate execution time
            end_time = datetime.now()
            results.execution_time = (end_time - start_time).total_seconds()
            
            if self.config.verbose:
                print(f"\n🎉 Pipeline completed successfully!")
                print(f"⏰ Total execution time: {results.execution_time:.1f} seconds")
                print(f"📅 Completed at: {end_time}")
        
        except Exception as e:
            results.success = False
            results.errors.append(str(e))
            
            if self.config.verbose:
                print(f"❌ Pipeline failed: {e}")
        
        return results
    
    async def _generate_visualizations(self, results: PipelineResults):
        """Generate all visualizations"""
        
        # Performance visualizations
        equity_chart = self.performance_visualizer.plot_equity_curve(
            results.backtest_result,
            self.config.benchmark_data
        )
        if self.config.save_charts:
            equity_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_equity_curve.png"
            equity_chart.savefig(equity_path, dpi=300, bbox_inches='tight')
            results.chart_paths['equity_curve'] = str(equity_path)
        
        drawdown_chart = self.performance_visualizer.plot_drawdown(results.backtest_result)
        if self.config.save_charts:
            dd_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_drawdown.png"
            drawdown_chart.savefig(dd_path, dpi=300, bbox_inches='tight')
            results.chart_paths['drawdown'] = str(dd_path)
        
        returns_dist_chart = self.performance_visualizer.plot_returns_distribution(results.backtest_result)
        if self.config.save_charts:
            dist_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_returns_distribution.png"
            returns_dist_chart.savefig(dist_path, dpi=300, bbox_inches='tight')
            results.chart_paths['returns_distribution'] = str(dist_path)
        
        rolling_metrics_chart = self.performance_visualizer.plot_rolling_metrics(results.backtest_result)
        if self.config.save_charts:
            rolling_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_rolling_metrics.png"
            rolling_metrics_chart.savefig(rolling_path, dpi=300, bbox_inches='tight')
            results.chart_paths['rolling_metrics'] = str(rolling_path)
        
        monthly_heatmap = self.performance_visualizer.plot_monthly_returns(results.backtest_result)
        if self.config.save_charts:
            heatmap_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_monthly_heatmap.png"
            monthly_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            results.chart_paths['monthly_heatmap'] = str(heatmap_path)
        
        # Monte Carlo visualizations
        if results.monte_carlo_results:
            mc_paths_chart = self.monte_carlo_visualizer.plot_simulation_paths(results.monte_carlo_results)
            if self.config.save_charts:
                mc_paths_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_monte_carlo_paths.png"
                mc_paths_chart.savefig(mc_paths_path, dpi=300, bbox_inches='tight')
                results.chart_paths['monte_carlo_paths'] = str(mc_paths_path)
            
            mc_dist_chart = self.monte_carlo_visualizer.plot_distribution_comparison(results.monte_carlo_results)
            if self.config.save_charts:
                mc_dist_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_monte_carlo_distribution.png"
                mc_dist_chart.savefig(mc_dist_path, dpi=300, bbox_inches='tight')
                results.chart_paths['monte_carlo_distribution'] = str(mc_dist_path)
        
        # Walk-forward visualizations
        if results.walk_forward_result:
            wf_comparison_chart = self.walk_forward_visualizer.plot_performance_comparison(results.walk_forward_result)
            if self.config.save_charts:
                wf_comp_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_walk_forward_comparison.png"
                wf_comparison_chart.savefig(wf_comp_path, dpi=300, bbox_inches='tight')
                results.chart_paths['walk_forward_comparison'] = str(wf_comp_path)
            
            wf_stability_chart = self.walk_forward_visualizer.plot_parameter_stability(results.walk_forward_result)
            if self.config.save_charts:
                wf_stab_path = Path(self.config.chart_output_dir) / f"{results.strategy_name}_parameter_stability.png"
                wf_stability_chart.savefig(wf_stab_path, dpi=300, bbox_inches='tight')
                results.chart_paths['parameter_stability'] = str(wf_stab_path)
    
    def _generate_overall_assessment(self, results: PipelineResults) -> Dict[str, Any]:
        """Generate overall assessment of strategy"""
        
        assessment = {
            'score': 0,
            'grade': 'C',
            'strengths': [],
            'weaknesses': [],
            'risk_level': 'Medium',
            'recommendation': 'Hold'
        }
        
        # Score components (0-10 each)
        scores = {}
        
        # Performance score
        perf = results.backtest_result
        perf_score = 0
        
        if perf.sharpe_ratio > 2:
            perf_score += 4
        elif perf.sharpe_ratio > 1:
            perf_score += 3
        elif perf.sharpe_ratio > 0.5:
            perf_score += 2
        elif perf.sharpe_ratio > 0:
            perf_score += 1
        
        if perf.max_drawdown > -0.1:
            perf_score += 3
        elif perf.max_drawdown > -0.2:
            perf_score += 2
        elif perf.max_drawdown > -0.3:
            perf_score += 1
        
        if perf.total_return > 0.2:
            perf_score += 3
        elif perf.total_return > 0.1:
            perf_score += 2
        elif perf.total_return > 0:
            perf_score += 1
        
        scores['performance'] = min(10, perf_score)
        
        # Validation score
        if results.validation_results:
            val_score = 0
            significant_tests = sum(1 for t in results.validation_results.hypothesis_tests if t.is_significant)
            val_score += min(5, significant_tests)
            
            if results.validation_results.is_validated:
                val_score += 3
            
            if results.validation_results.robustness_results.get('out_of_sample', {}).get('is_robust', False):
                val_score += 2
            
            scores['validation'] = min(10, val_score)
        else:
            scores['validation'] = 5  # Neutral
        
        # Robustness score
        robust_score = 0
        
        if results.walk_forward_result:
            if results.walk_forward_result.hit_rate > 0.6:
                robust_score += 4
            elif results.walk_forward_result.hit_rate > 0.4:
                robust_score += 2
            
            avg_stability = np.mean(list(results.walk_forward_result.parameter_stability.values())) if results.walk_forward_result.parameter_stability else 0
            if avg_stability > 0.7:
                robust_score += 3
            elif avg_stability > 0.5:
                robust_score += 1
        
        if results.monte_carlo_results:
            positive_prob = len([s for s in results.monte_carlo_results.simulations if s.total_return > 0]) / len(results.monte_carlo_results.simulations)
            if positive_prob > 0.7:
                robust_score += 3
            elif positive_prob > 0.5:
                robust_score += 1
        
        scores['robustness'] = min(10, robust_score)
        
        # Calculate overall score
        overall_score = np.mean(list(scores.values()))
        assessment['score'] = round(overall_score, 1)
        
        # Determine grade
        if overall_score >= 8.5:
            assessment['grade'] = 'A+'
        elif overall_score >= 8:
            assessment['grade'] = 'A'
        elif overall_score >= 7.5:
            assessment['grade'] = 'B+'
        elif overall_score >= 7:
            assessment['grade'] = 'B'
        elif overall_score >= 6.5:
            assessment['grade'] = 'C+'
        elif overall_score >= 6:
            assessment['grade'] = 'C'
        elif overall_score >= 5.5:
            assessment['grade'] = 'D+'
        elif overall_score >= 5:
            assessment['grade'] = 'D'
        else:
            assessment['grade'] = 'F'
        
        # Identify strengths and weaknesses
        if scores['performance'] >= 7:
            assessment['strengths'].append('Strong performance metrics')
        elif scores['performance'] < 5:
            assessment['weaknesses'].append('Weak performance metrics')
        
        if scores['validation'] >= 7:
            assessment['strengths'].append('Statistically significant results')
        elif scores['validation'] < 5:
            assessment['weaknesses'].append('Lack of statistical significance')
        
        if scores['robustness'] >= 7:
            assessment['strengths'].append('High robustness across tests')
        elif scores['robustness'] < 5:
            assessment['weaknesses'].append('Poor robustness or overfitting')
        
        # Risk level
        if perf.max_drawdown < -0.3 or perf.volatility > 0.3:
            assessment['risk_level'] = 'High'
        elif perf.max_drawdown < -0.15 and perf.volatility < 0.2:
            assessment['risk_level'] = 'Low'
        
        # Recommendation
        if overall_score >= 7.5:
            assessment['recommendation'] = 'Deploy'
        elif overall_score >= 6:
            assessment['recommendation'] = 'Consider with caution'
        elif overall_score >= 5:
            assessment['recommendation'] = 'Major improvements needed'
        else:
            assessment['recommendation'] = 'Do not deploy'
        
        assessment['component_scores'] = scores
        
        return assessment
    
    def _generate_pipeline_recommendations(self, results: PipelineResults) -> List[str]:
        """Generate comprehensive recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        perf = results.backtest_result
        if perf.sharpe_ratio < 0.5:
            recommendations.append("Improve risk-adjusted returns - consider strategy refinement")
        
        if perf.max_drawdown < -0.25:
            recommendations.append("Implement stronger risk controls - current drawdown is excessive")
        
        if perf.win_rate < 0.4:
            recommendations.append("Low win rate - review entry/exit criteria")
        
        # Validation-based recommendations
        if results.validation_results:
            if not results.validation_results.is_validated:
                recommendations.append("Strategy not statistically validated - address significance issues")
            
            recommendations.extend(results.validation_results.recommendations)
        
        # Robustness-based recommendations
        if results.walk_forward_result:
            if results.walk_forward_result.hit_rate < 0.5:
                recommendations.append("Poor out-of-sample performance - strategy may be overfitted")
        
        # Monte Carlo-based recommendations
        if results.monte_carlo_results:
            positive_prob = len([s for s in results.monte_carlo_results.simulations if s.total_return > 0]) / len(results.monte_carlo_results.simulations)
            if positive_prob < 0.6:
                recommendations.append("High probability of losses - reconsider strategy logic")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Strategy shows strong performance - consider deployment with proper risk management")
        
        return recommendations
    
    async def _save_results(self, results: PipelineResults):
        """Save all results to files"""
        
        # Create results directory
        results_dir = Path(self.config.results_output_dir)
        results_dir.mkdir(exist_ok=True)
        
        # Save basic results
        basic_results = {
            'strategy_name': results.strategy_name,
            'execution_time': results.execution_time,
            'success': results.success,
            'backtest_result': {
                'total_return': results.backtest_result.total_return,
                'sharpe_ratio': results.backtest_result.sharpe_ratio,
                'max_drawdown': results.backtest_result.max_drawdown,
                'win_rate': results.backtest_result.win_rate,
                'total_trades': results.backtest_result.total_trades
            },
            'overall_assessment': results.overall_assessment,
            'recommendations': results.recommendations,
            'chart_paths': results.chart_paths
        }
        
        # Save as JSON
        with open(results_dir / f"{results.strategy_name}_results.json", 'w') as f:
            json.dump(basic_results, f, indent=2, default=str)
        
        # Save detailed performance report
        if self.config.generate_report:
            await self._generate_html_report(results, results_dir)
    
    async def _generate_html_report(self, results: PipelineResults, output_dir: Path):
        """Generate comprehensive HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Analysis Report - {strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }}
                .grade-{{grade}} {{ font-size: 48px; font-weight: bold; }}
                .grade-A {{ color: #2ECC71; }}
                .grade-B {{ color: #F39C12; }}
                .grade-C {{ color: #E67E22; }}
                .grade-D {{ color: #E74C3C; }}
                .grade-F {{ color: #C0392B; }}
                .recommendation {{ background: #E8F5E8; padding: 15px; margin: 10px 0; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Strategy Analysis Report</h1>
                <h2>{strategy_name}</h2>
                <div class="grade-{grade}">{grade}</div>
                <p>Overall Score: {score}/10</p>
                <p>Generated: {date}</p>
            </div>
            
            <div class="section">
                <h3>Performance Summary</h3>
                <div class="metric">Total Return: {total_return:.2%}</div>
                <div class="metric">Sharpe Ratio: {sharpe_ratio:.2f}</div>
                <div class="metric">Max Drawdown: {max_drawdown:.2%}</div>
                <div class="metric">Win Rate: {win_rate:.1%}</div>
                <div class="metric">Total Trades: {total_trades}</div>
            </div>
            
            <div class="section">
                <h3>Overall Assessment</h3>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p><strong>Recommendation:</strong> {recommendation}</p>
                
                <h4>Strengths:</h4>
                <ul>
                {strengths}
                </ul>
                
                <h4>Areas for Improvement:</h4>
                <ul>
                {weaknesses}
                </ul>
            </div>
            
            <div class="section">
                <h3>Recommendations</h3>
                {recommendations_html}
            </div>
            
            <div class="section">
                <h3>Charts</h3>
                {charts_html}
            </div>
        </body>
        </html>
        """
        
        # Prepare template variables
        strengths_html = "\n".join([f"<li>{s}</li>" for s in results.overall_assessment.get('strengths', [])])
        weaknesses_html = "\n".join([f"<li>{w}</li>" for w in results.overall_assessment.get('weaknesses', [])])
        recommendations_html = "\n".join([f'<div class="recommendation">{r}</div>' for r in results.recommendations])
        
        charts_html = ""
        for chart_name, chart_path in results.chart_paths.items():
            chart_filename = Path(chart_path).name
            charts_html += f'<div class="chart"><h4>{chart_name.replace("_", " ").title()}</h4><img src="{chart_filename}" alt="{chart_name}" style="max-width: 100%;"></div>'
        
        html_content = html_template.format(
            strategy_name=results.strategy_name,
            grade=results.overall_assessment.get('grade', 'C'),
            score=results.overall_assessment.get('score', 0),
            date=results.execution_date.strftime('%Y-%m-%d %H:%M:%S'),
            total_return=results.backtest_result.total_return,
            sharpe_ratio=results.backtest_result.sharpe_ratio,
            max_drawdown=results.backtest_result.max_drawdown,
            win_rate=results.backtest_result.win_rate,
            total_trades=results.backtest_result.total_trades,
            risk_level=results.overall_assessment.get('risk_level', 'Medium'),
            recommendation=results.overall_assessment.get('recommendation', 'Hold'),
            strengths=strengths_html,
            weaknesses=weaknesses_html,
            recommendations_html=recommendations_html,
            charts_html=charts_html
        )
        
        # Save HTML report
        with open(output_dir / f"{results.strategy_name}_report.html", 'w') as f:
            f.write(html_content)
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        
        Path(self.config.chart_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.report_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_default_walk_forward_config(self) -> WalkForwardConfig:
        """Create default walk-forward configuration"""
        
        return WalkForwardConfig(
            start_date=self.config.backtest_config.start_date,
            end_date=self.config.backtest_config.end_date,
            training_window=252 * 2,  # 2 years
            testing_window=63,        # 3 months
            step_size=30,             # 1 month
            optimization_metric="sharpe_ratio"
        )
    
    def _create_default_monte_carlo_config(self) -> MonteCarloConfig:
        """Create default Monte Carlo configuration"""
        
        return MonteCarloConfig(
            num_simulations=1000,
            time_horizon=252,
            include_scenarios=True,
            parameter_uncertainty=True
        )
    
    def _create_default_validation_config(self) -> ValidationConfig:
        """Create default validation configuration"""
        
        return ValidationConfig(
            significance_level=0.05,
            confidence_level=0.95,
            bootstrap_samples=5000,
            multiple_testing_correction=True
        )


def explain_backtesting_pipeline():
    """
    Educational explanation of the complete backtesting pipeline
    """
    
    print("=== Complete Backtesting Pipeline Educational Guide ===\n")
    
    concepts = {
        'Pipeline Integration': "Combining all analysis components into a cohesive workflow",
        
        'Comprehensive Analysis': "Evaluating strategies from multiple angles for robust validation",
        
        'Standardized Methodology': "Ensuring consistent and reproducible analysis procedures",
        
        'Professional Reporting': "Generating institutional-quality reports and visualizations",
        
        'Risk Assessment': "Multi-layered risk evaluation across different time horizons",
        
        'Statistical Validation': "Rigorous testing to distinguish skill from luck",
        
        'Performance Attribution': "Understanding sources of returns and risk",
        
        'Robustness Testing': "Evaluating strategy stability across conditions",
        
        'Decision Support': "Providing evidence-based recommendations for deployment",
        
        'Documentation': "Complete record of analysis for compliance and review"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Pipeline Components ===")
    components = {
        "Basic Backtesting": "Core strategy testing with realistic market conditions",
        "Performance Analysis": "Detailed metrics and attribution analysis",
        "Walk-Forward Analysis": "Time-series validation with parameter optimization",
        "Monte Carlo Simulation": "Uncertainty analysis and scenario testing",
        "Statistical Validation": "Hypothesis testing and significance validation",
        "Visualization": "Professional charts and graphical analysis",
        "Reporting": "Comprehensive documentation and recommendations"
    }
    
    for component, description in components.items():
        print(f"{component}:")
        print(f"  {description}\n")
    
    print("=== Pipeline Best Practices ===")
    practices = [
        "1. Use consistent methodology across all analyses",
        "2. Validate results with multiple approaches",
        "3. Document all assumptions and parameters",
        "4. Use appropriate statistical significance levels",
        "5. Consider both statistical and practical significance",
        "6. Test across different market regimes",
        "7. Implement proper risk management controls",
        "8. Generate comprehensive documentation",
        "9. Review results with domain experts",
        "10. Update analysis regularly with new data"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n=== Pipeline Output ===")
    outputs = {
        "Performance Metrics": "30+ key performance and risk metrics",
        "Visual Charts": "Professional charts for all analysis aspects",
        "Statistical Reports": "Detailed validation and significance testing",
        "Risk Assessment": "Multi-dimensional risk analysis",
        "Recommendations": "Evidence-based improvement suggestions",
        "HTML Reports": "Comprehensive web-based analysis reports",
        "JSON Data": "Machine-readable results for integration",
        "Audit Trail": "Complete record of analysis methodology"
    }
    
    for output, description in outputs.items():
        print(f"{output}:")
        print(f"  {description}\n")


if __name__ == "__main__":
    # Example usage
    explain_backtesting_pipeline()
    
    print("\n=== Complete Backtesting Pipeline Example ===")
    print("To use the complete pipeline:")
    print("1. Configure pipeline with desired analysis components")
    print("2. Provide strategy class and parameters")
    print("3. Supply historical market data")
    print("4. Run complete analysis")
    print("5. Review comprehensive results and recommendations")
    
    # Example configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_cash=100000
    )
    
    pipeline_config = PipelineConfig(
        backtest_config=backtest_config,
        enable_walk_forward=True,
        enable_monte_carlo=True,
        enable_validation=True,
        enable_visualization=True,
        save_charts=True,
        generate_report=True
    )
    
    print(f"\nSample pipeline configuration:")
    print(f"  Walk-forward analysis: {pipeline_config.enable_walk_forward}")
    print(f"  Monte Carlo simulation: {pipeline_config.enable_monte_carlo}")
    print(f"  Statistical validation: {pipeline_config.enable_validation}")
    print(f"  Visualization: {pipeline_config.enable_visualization}")
    print(f"  Report generation: {pipeline_config.generate_report}")
    
    print("\nThe pipeline provides:")
    print("• End-to-end automated analysis")
    print("• Institutional-grade validation")
    print("• Professional visualization")
    print("• Comprehensive reporting")
    print("• Evidence-based recommendations")
    print("• Complete audit trail")
    print("• Reproducible results")
    print("• Risk assessment framework")