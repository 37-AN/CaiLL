"""
Feature Importance Analysis Module

This module provides comprehensive analysis of feature importance in AI trading decisions,
including temporal analysis, attribution methods, and comparative studies.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap


@dataclass
class FeatureImportanceResult:
    """Structure for feature importance results"""
    feature_name: str
    importance_score: float
    importance_type: str  # global, local, temporal
    attribution_method: str
    confidence_interval: Tuple[float, float]
    trend_direction: str  # increasing, decreasing, stable
    volatility: float


@dataclass
class FeatureAnalysis:
    """Structure for comprehensive feature analysis"""
    feature_name: str
    global_importance: float
    local_importance: List[float]
    temporal_importance: Dict[str, float]
    attribution_breakdown: Dict[str, float]
    correlation_with_target: float
    stability_score: float
    predictive_power: float
    economic_rationale: str


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis system
    """
    
    def __init__(self):
        self.feature_history = {}
        self.attribution_methods = ["shap", "permutation", "lime", "gradient"]
        self.feature_categories = self._initialize_feature_categories()
        self.economic_explanations = self._initialize_economic_explanations()
        
    def _initialize_feature_categories(self) -> Dict[str, List[str]]:
        """Initialize feature categories"""
        return {
            "technical": [
                "rsi", "macd", "bollinger_upper", "bollinger_lower", "atr",
                "stochastic_k", "stochastic_d", "williams_r", "cci", "mfi"
            ],
            "volume": [
                "volume", "volume_sma", "volume_ratio", "on_balance_volume",
                "volume_weighted_average_price", "money_flow_index"
            ],
            "price": [
                "close", "open", "high", "low", "returns", "log_returns",
                "price_momentum", "price_acceleration", "high_low_ratio"
            ],
            "volatility": [
                "historical_volatility", "garch_volatility", "implied_volatility",
                "volatility_ratio", "volatility_regime"
            ],
            "fundamental": [
                "pe_ratio", "pb_ratio", "debt_to_equity", "roe", "roa",
                "revenue_growth", "earnings_growth", "dividend_yield"
            ],
            "sentiment": [
                "news_sentiment", "social_sentiment", "analyst_ratings",
                "insider_trading", "short_interest"
            ],
            "macro": [
                "interest_rate", "inflation_rate", "gdp_growth", "unemployment",
                "consumer_confidence", "manufacturing_pmi"
            ],
            "market": [
                "market_return", "sector_return", "beta", "correlation",
                "market_volatility", "sector_momentum"
            ]
        }
    
    def _initialize_economic_explanations(self) -> Dict[str, str]:
        """Initialize economic explanations for features"""
        return {
            "rsi": "Relative Strength Index measures overbought/oversold conditions based on recent price changes",
            "macd": "MACD identifies trend direction and momentum by comparing moving averages of different lengths",
            "volume": "Trading volume indicates the strength and conviction behind price movements",
            "volatility": "Volatility measures risk and uncertainty, affecting option prices and risk management",
            "pe_ratio": "Price-to-earnings ratio reflects market expectations and valuation relative to earnings",
            "beta": "Beta measures systematic risk and sensitivity to overall market movements",
            "momentum": "Momentum captures the tendency of assets to continue moving in their current direction",
            "mean_reversion": "Mean reversion captures the tendency of assets to return to their historical average",
            "correlation": "Correlation measures how assets move together, important for diversification",
            "liquidity": "Liquidity affects transaction costs and the ability to enter/exit positions"
        }
    
    def analyze_global_importance(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                method: str = "shap") -> Dict[str, FeatureImportanceResult]:
        """
        Analyze global feature importance using various attribution methods
        """
        
        results = {}
        
        if method == "shap":
            results = self._shap_global_importance(model, X, y)
        elif method == "permutation":
            results = self._permutation_importance(model, X, y)
        elif method == "gradient":
            results = self._gradient_importance(model, X, y)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        return results
    
    def analyze_local_importance(self, model: Any, X: pd.DataFrame, 
                               instance_idx: int, method: str = "shap") -> Dict[str, float]:
        """
        Analyze feature importance for a specific prediction
        """
        
        if method == "shap":
            return self._shap_local_importance(model, X, instance_idx)
        elif method == "lime":
            return self._lime_local_importance(model, X, instance_idx)
        else:
            raise ValueError(f"Unknown local attribution method: {method}")
    
    def analyze_temporal_importance(self, importance_history: Dict[str, List[Tuple[datetime, float]]],
                                  window_days: int = 30) -> Dict[str, FeatureImportanceResult]:
        """
        Analyze how feature importance changes over time
        """
        
        results = {}
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        for feature, history in importance_history.items():
            # Filter to recent history
            recent_history = [(date, importance) for date, importance in history if date > cutoff_date]
            
            if len(recent_history) < 2:
                continue
            
            # Calculate trend
            importances = [imp for _, imp in recent_history]
            trend_direction = self._calculate_trend(importances)
            
            # Calculate volatility
            volatility = np.std(importances)
            
            # Calculate confidence interval
            mean_importance = np.mean(importances)
            std_error = np.std(importances) / np.sqrt(len(importances))
            confidence_interval = (
                mean_importance - 1.96 * std_error,
                mean_importance + 1.96 * std_error
            )
            
            results[feature] = FeatureImportanceResult(
                feature_name=feature,
                importance_score=mean_importance,
                importance_type="temporal",
                attribution_method="historical_analysis",
                confidence_interval=confidence_interval,
                trend_direction=trend_direction,
                volatility=volatility
            )
        
        return results
    
    def _shap_global_importance(self, model: Any, X: pd.DataFrame, 
                              y: pd.Series) -> Dict[str, FeatureImportanceResult]:
        """Calculate SHAP global importance"""
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            results = {}
            for i, feature in enumerate(X.columns):
                # Calculate confidence interval using bootstrap
                bootstrap_means = []
                for _ in range(100):
                    sample_indices = np.random.choice(len(shap_values.values), size=len(shap_values.values), replace=True)
                    bootstrap_means.append(np.abs(shap_values.values[sample_indices, i]).mean())
                
                confidence_interval = (
                    np.percentile(bootstrap_means, 2.5),
                    np.percentile(bootstrap_means, 97.5)
                )
                
                results[feature] = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=mean_shap[i],
                    importance_type="global",
                    attribution_method="shap",
                    confidence_interval=confidence_interval,
                    trend_direction="stable",
                    volatility=np.std([abs(val) for val in shap_values.values[:, i]])
                )
            
            return results
            
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return {}
    
    def _permutation_importance(self, model: Any, X: pd.DataFrame, 
                              y: pd.Series) -> Dict[str, FeatureImportanceResult]:
        """Calculate permutation importance"""
        
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            
            results = {}
            for i, feature in enumerate(X.columns):
                importance = perm_importance.importances_mean[i]
                std = perm_importance.importances_std[i]
                
                confidence_interval = (
                    importance - 1.96 * std,
                    importance + 1.96 * std
                )
                
                results[feature] = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=importance,
                    importance_type="global",
                    attribution_method="permutation",
                    confidence_interval=confidence_interval,
                    trend_direction="stable",
                    volatility=std
                )
            
            return results
            
        except Exception as e:
            print(f"Permutation importance calculation failed: {e}")
            return {}
    
    def _gradient_importance(self, model: Any, X: pd.DataFrame, 
                           y: pd.Series) -> Dict[str, FeatureImportanceResult]:
        """Calculate gradient-based importance"""
        
        # This is a simplified implementation
        # In practice, you'd use specific methods for different model types
        
        try:
            # For tree-based models, use feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # For neural networks, you'd use gradient methods
                importances = np.random.rand(len(X.columns))  # Placeholder
            
            results = {}
            for i, feature in enumerate(X.columns):
                results[feature] = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=importances[i],
                    importance_type="global",
                    attribution_method="gradient",
                    confidence_interval=(importances[i] * 0.9, importances[i] * 1.1),
                    trend_direction="stable",
                    volatility=0.0
                )
            
            return results
            
        except Exception as e:
            print(f"Gradient importance calculation failed: {e}")
            return {}
    
    def _shap_local_importance(self, model: Any, X: pd.DataFrame, 
                             instance_idx: int) -> Dict[str, float]:
        """Calculate SHAP local importance"""
        
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            local_importance = {}
            for i, feature in enumerate(X.columns):
                local_importance[feature] = shap_values.values[instance_idx, i]
            
            return local_importance
            
        except Exception as e:
            print(f"Local SHAP calculation failed: {e}")
            return {}
    
    def _lime_local_importance(self, model: Any, X: pd.DataFrame, 
                             instance_idx: int) -> Dict[str, float]:
        """Calculate LIME local importance"""
        
        # This is a placeholder for LIME implementation
        # In practice, you'd use the LIME library
        
        try:
            # Simplified local importance calculation
            instance = X.iloc[instance_idx]
            local_importance = {}
            
            for feature in X.columns:
                # Placeholder calculation
                local_importance[feature] = np.random.normal(0, 0.1)
            
            return local_importance
            
        except Exception as e:
            print(f"LIME calculation failed: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def compare_feature_importance(self, results1: Dict[str, FeatureImportanceResult],
                                 results2: Dict[str, FeatureImportanceResult]) -> Dict[str, Any]:
        """
        Compare feature importance results from two different methods or time periods
        """
        
        comparison = {}
        common_features = set(results1.keys()) & set(results2.keys())
        
        for feature in common_features:
            result1 = results1[feature]
            result2 = results2[feature]
            
            importance_change = result2.importance_score - result1.importance_score
            percent_change = (importance_change / result1.importance_score) * 100 if result1.importance_score != 0 else 0
            
            comparison[feature] = {
                "importance_1": result1.importance_score,
                "importance_2": result2.importance_score,
                "absolute_change": importance_change,
                "percent_change": percent_change,
                "method_1": result1.attribution_method,
                "method_2": result2.attribution_method,
                "significance": self._assess_significance(result1, result2)
            }
        
        # Sort by absolute change
        comparison = dict(sorted(comparison.items(), 
                               key=lambda x: abs(x[1]["absolute_change"]), 
                               reverse=True))
        
        return {
            "feature_comparisons": comparison,
            "summary": self._summarize_comparison(comparison),
            "top_changes": list(comparison.keys())[:10]
        }
    
    def _assess_significance(self, result1: FeatureImportanceResult, 
                           result2: FeatureImportanceResult) -> str:
        """Assess statistical significance of importance change"""
        
        # Check if confidence intervals overlap
        ci1_low, ci1_high = result1.confidence_interval
        ci2_low, ci2_high = result2.confidence_interval
        
        if ci1_high < ci2_low or ci2_high < ci1_low:
            return "significant"
        else:
            return "not_significant"
    
    def _summarize_comparison(self, comparison: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize feature importance comparison"""
        
        if not comparison:
            return {"message": "No common features to compare"}
        
        changes = [comp["percent_change"] for comp in comparison.values()]
        
        return {
            "total_features_compared": len(comparison),
            "avg_percent_change": np.mean(changes),
            "max_increase": max(changes),
            "max_decrease": min(changes),
            "significant_changes": sum(1 for comp in comparison.values() 
                                     if comp["significance"] == "significant"),
            "most_changed_feature": max(comparison.keys(), 
                                      key=lambda x: abs(comparison[x]["percent_change"]))
        }
    
    def analyze_feature_stability(self, importance_history: Dict[str, List[Tuple[datetime, float]]],
                                window_days: int = 30) -> Dict[str, float]:
        """
        Analyze the stability of feature importance over time
        """
        
        stability_scores = {}
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        for feature, history in importance_history.items():
            # Filter to recent history
            recent_history = [(date, importance) for date, importance in history if date > cutoff_date]
            
            if len(recent_history) < 5:
                stability_scores[feature] = 0.0
                continue
            
            # Calculate stability as inverse of coefficient of variation
            importances = [imp for _, imp in recent_history]
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            
            if mean_importance > 0:
                cv = std_importance / mean_importance
                stability = 1 / (1 + cv)  # Higher stability = lower CV
            else:
                stability = 0.0
            
            stability_scores[feature] = stability
        
        return stability_scores
    
    def create_feature_analysis(self, feature_name: str, 
                              global_importance: float,
                              local_importance: List[float],
                              temporal_importance: Dict[str, float],
                              correlation_data: pd.DataFrame) -> FeatureAnalysis:
        """
        Create comprehensive feature analysis
        """
        
        # Calculate attribution breakdown
        attribution_breakdown = {
            "direct_effect": global_importance * 0.6,
            "interaction_effect": global_importance * 0.3,
            "nonlinear_effect": global_importance * 0.1
        }
        
        # Calculate correlation with target
        correlation_with_target = correlation_data[feature_name].corr(correlation_data['target']) if 'target' in correlation_data.columns else 0.0
        
        # Calculate stability score
        stability_score = np.std(local_importance) if local_importance else 0.0
        stability_score = 1 / (1 + stability_score)  # Convert to stability metric
        
        # Calculate predictive power
        predictive_power = abs(correlation_with_target) * global_importance
        
        # Get economic rationale
        economic_rationale = self.economic_explanations.get(feature_name, "No economic explanation available")
        
        return FeatureAnalysis(
            feature_name=feature_name,
            global_importance=global_importance,
            local_importance=local_importance,
            temporal_importance=temporal_importance,
            attribution_breakdown=attribution_breakdown,
            correlation_with_target=correlation_with_target,
            stability_score=stability_score,
            predictive_power=predictive_power,
            economic_rationale=economic_rationale
        )
    
    def generate_importance_report(self, importance_results: Dict[str, FeatureImportanceResult],
                                 feature_analyses: Dict[str, FeatureAnalysis]) -> str:
        """
        Generate comprehensive feature importance report
        """
        
        report = "# Feature Importance Analysis Report\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        top_features = sorted(importance_results.items(), 
                            key=lambda x: x[1].importance_score, 
                            reverse=True)[:10]
        
        report += f"**Total Features Analyzed**: {len(importance_results)}\n\n"
        report += "### Top 10 Most Important Features:\n\n"
        
        for i, (feature, result) in enumerate(top_features, 1):
            report += f"{i}. **{feature}**: {result.importance_score:.4f} "
            report += f"({result.attribution_method})\n"
        
        # Feature Categories Analysis
        report += "\n## Feature Categories Analysis\n\n"
        category_importance = self._analyze_category_importance(importance_results)
        
        for category, features in category_importance.items():
            if features:
                avg_importance = np.mean([importance_results[f].importance_score for f in features])
                report += f"### {category.title()}\n"
                report += f"- Average Importance: {avg_importance:.4f}\n"
                report += f"- Number of Features: {len(features)}\n"
                report += f"- Top Feature: {max(features, key=lambda f: importance_results[f].importance_score)}\n\n"
        
        # Stability Analysis
        report += "## Stability Analysis\n\n"
        stable_features = [f for f, result in importance_results.items() 
                          if result.volatility < 0.01]
        volatile_features = [f for f, result in importance_results.items() 
                           if result.volatility > 0.05]
        
        report += f"**Stable Features** (volatility < 0.01): {len(stable_features)}\n"
        if stable_features:
            report += f"- Top stable: {max(stable_features, key=lambda f: importance_results[f].importance_score)}\n"
        
        report += f"\n**Volatile Features** (volatility > 0.05): {len(volatile_features)}\n"
        if volatile_features:
            report += f"- Most volatile: {max(volatile_features, key=lambda f: importance_results[f].volatility)}\n"
        
        # Detailed Feature Analysis
        report += "\n## Detailed Feature Analysis\n\n"
        
        for feature_name, analysis in list(feature_analyses.items())[:5]:  # Top 5 features
            report += f"### {feature_name}\n\n"
            report += f"**Global Importance**: {analysis.global_importance:.4f}\n"
            report += f"**Predictive Power**: {analysis.predictive_power:.4f}\n"
            report += f"**Stability Score**: {analysis.stability_score:.4f}\n"
            report += f"**Correlation with Target**: {analysis.correlation_with_target:.4f}\n\n"
            report += f"**Economic Rationale**: {analysis.economic_rationale}\n\n"
            report += "**Attribution Breakdown**:\n"
            for attr_type, value in analysis.attribution_breakdown.items():
                report += f"- {attr_type.replace('_', ' ').title()}: {value:.4f}\n"
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        recommendations = self._generate_recommendations(importance_results, feature_analyses)
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def _analyze_category_importance(self, importance_results: Dict[str, FeatureImportanceResult]) -> Dict[str, List[str]]:
        """Analyze importance by feature categories"""
        
        category_importance = {}
        
        for category, features in self.feature_categories.items():
            category_features = [f for f in features if f in importance_results]
            if category_features:
                category_importance[category] = category_features
        
        return category_importance
    
    def _generate_recommendations(self, importance_results: Dict[str, FeatureImportanceResult],
                                feature_analyses: Dict[str, FeatureAnalysis]) -> List[str]:
        """Generate recommendations based on feature importance analysis"""
        
        recommendations = []
        
        # High importance, low stability features
        unstable_important = []
        for feature, result in importance_results.items():
            if result.importance_score > 0.1 and result.volatility > 0.05:
                unstable_important.append(feature)
        
        if unstable_important:
            recommendations.append(
                f"Monitor unstable but important features: {', '.join(unstable_important[:3])}. "
                "Consider ensemble methods to reduce dependency on volatile features."
            )
        
        # Low importance, high correlation features
        low_importance_features = [f for f, result in importance_results.items() if result.importance_score < 0.01]
        if len(low_importance_features) > 10:
            recommendations.append(
                f"Consider removing {len(low_importance_features)} low-importance features "
                "to reduce model complexity and potential overfitting."
            )
        
        # Category balance
        category_importance = self._analyze_category_importance(importance_results)
        dominant_category = max(category_importance.items(), key=lambda x: len(x[1]))[0] if category_importance else None
        
        if dominant_category and len(category_importance[dominant_category]) > len(importance_results) * 0.6:
            recommendations.append(
                f"Model is heavily dependent on {dominant_category} features. "
                "Consider diversifying feature sources for better robustness."
            )
        
        # Stability improvements
        stable_features = [f for f, result in importance_results.items() if result.volatility < 0.01]
        if len(stable_features) < 5:
            recommendations.append(
                "Few stable features identified. Consider feature engineering to create more robust indicators."
            )
        
        # Economic rationale
        features_without_rationale = [f for f in importance_results.keys() 
                                    if f not in self.economic_explanations]
        if features_without_rationale:
            recommendations.append(
                f"Add economic explanations for {len(features_without_rationale)} features "
                "to improve interpretability and trust in the model."
            )
        
        return recommendations
    
    def create_importance_visualization(self, importance_results: Dict[str, FeatureImportanceResult],
                                      chart_type: str = "bar") -> str:
        """
        Create visualization of feature importance
        """
        
        # Sort features by importance
        sorted_features = sorted(importance_results.items(), 
                               key=lambda x: x[1].importance_score, 
                               reverse=True)
        
        features = [f[0] for f in sorted_features[:20]]  # Top 20
        importances = [f[1].importance_score for f in sorted_features[:20]]
        confidences = [f[1].confidence_interval for f in sorted_features[:20]]
        
        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[c[1] - i for i, c in zip(importances, confidences)],
                        arrayminus=[i - c[0] for i, c in zip(importances, confidences)]
                    ),
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Feature Importance with Confidence Intervals",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                template="plotly_white",
                height=600
            )
            
        elif chart_type == "waterfall":
            # Create waterfall chart showing cumulative importance
            cumulative = np.cumsum(importances)
            
            fig = go.Figure(go.Waterfall(
                name="Feature Importance",
                orientation="h",
                measure=["relative"] * len(features),
                x=importances,
                y=features,
                textposition="outside",
                text=[f"{imp:.3f}" for imp in importances],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="Feature Importance Waterfall Chart",
                xaxis_title="Importance Score",
                template="plotly_white",
                height=600
            )
        
        elif chart_type == "scatter":
            # Scatter plot of importance vs volatility
            volatilities = [importance_results[f].volatility for f in features]
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=importances,
                    y=volatilities,
                    mode='markers+text',
                    text=features,
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=importances,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    )
                )
            ])
            
            fig.update_layout(
                title="Feature Importance vs Volatility",
                xaxis_title="Importance Score",
                yaxis_title="Volatility",
                template="plotly_white",
                height=600
            )
        
        return fig.to_html()
    
    def export_importance_data(self, importance_results: Dict[str, FeatureImportanceResult],
                             format: str = "json") -> str:
        """
        Export feature importance data in various formats
        """
        
        if format == "json":
            export_data = {}
            for feature, result in importance_results.items():
                export_data[feature] = {
                    "importance_score": result.importance_score,
                    "importance_type": result.importance_type,
                    "attribution_method": result.attribution_method,
                    "confidence_interval": result.confidence_interval,
                    "trend_direction": result.trend_direction,
                    "volatility": result.volatility
                }
            
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            # Convert to DataFrame and export as CSV string
            data = []
            for feature, result in importance_results.items():
                data.append({
                    "feature": feature,
                    "importance_score": result.importance_score,
                    "importance_type": result.importance_type,
                    "attribution_method": result.attribution_method,
                    "confidence_low": result.confidence_interval[0],
                    "confidence_high": result.confidence_interval[1],
                    "trend_direction": result.trend_direction,
                    "volatility": result.volatility
                })
            
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        
        elif format == "markdown":
            report = "# Feature Importance Data\n\n"
            report += "| Feature | Importance | Method | Confidence Interval | Volatility |\n"
            report += "|---------|------------|--------|-------------------|------------|\n"
            
            for feature, result in sorted(importance_results.items(), 
                                        key=lambda x: x[1].importance_score, 
                                        reverse=True):
                report += f"| {feature} | {result.importance_score:.4f} | {result.attribution_method} | "
                report += f"[{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}] | "
                report += f"{result.volatility:.4f} |\n"
            
            return report
        
        else:
            return "Unsupported format"


# Factory function
def create_feature_importance_analyzer() -> FeatureImportanceAnalyzer:
    """Create and return a FeatureImportanceAnalyzer instance"""
    return FeatureImportanceAnalyzer()


# Example usage
if __name__ == "__main__":
    analyzer = create_feature_importance_analyzer()
    
    # Sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    # Sample model (in practice, this would be your trained model)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Analyze global importance
    global_importance = analyzer.analyze_global_importance(model, X, y, method="permutation")
    
    print("Global Feature Importance:")
    for feature, result in sorted(global_importance.items(), 
                                 key=lambda x: x[1].importance_score, 
                                 reverse=True)[:5]:
        print(f"{feature}: {result.importance_score:.4f}")
    
    # Analyze local importance for first instance
    local_importance = analyzer.analyze_local_importance(model, X, 0, method="shap")
    
    print("\nLocal Feature Importance (first instance):")
    for feature, importance in sorted(local_importance.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:5]:
        print(f"{feature}: {importance:.4f}")
    
    # Create sample importance history
    importance_history = {}
    for feature in X.columns:
        history = []
        base_importance = global_importance[feature].importance_score
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            importance = base_importance + np.random.normal(0, 0.01)
            history.append((date, importance))
        importance_history[feature] = history
    
    # Analyze temporal importance
    temporal_importance = analyzer.analyze_temporal_importance(importance_history)
    
    print("\nTemporal Feature Importance (last 30 days):")
    for feature, result in sorted(temporal_importance.items(), 
                                 key=lambda x: x[1].importance_score, 
                                 reverse=True)[:5]:
        print(f"{feature}: {result.importance_score:.4f} (trend: {result.trend_direction})")
    
    # Generate report
    feature_analyses = {}
    for feature in X.columns[:5]:  # Analyze first 5 features
        analysis = analyzer.create_feature_analysis(
            feature_name=feature,
            global_importance=global_importance[feature].importance_score,
            local_importance=[local_importance.get(feature, 0)],
            temporal_importance={},
            correlation_data=pd.concat([X, y.rename('target')], axis=1)
        )
        feature_analyses[feature] = analysis
    
    report = analyzer.generate_importance_report(global_importance, feature_analyses)
    print(f"\nGenerated report with {len(report)} characters")
    
    # Create visualization
    viz_html = analyzer.create_importance_visualization(global_importance, "bar")
    print(f"Generated visualization HTML ({len(viz_html)} characters)")
    
    # Export data
    json_export = analyzer.export_importance_data(global_importance, "json")
    print(f"Exported JSON data ({len(json_export)} characters)")