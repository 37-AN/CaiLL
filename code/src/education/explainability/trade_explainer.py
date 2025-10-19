"""
Trade Explainer - Real-time AI Trading Decision Explanation System

This module provides real-time explanations for AI trading decisions,
including feature importance, decision rationale, and risk analysis.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class TradeDecision:
    """Structure for trade decisions"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    position_size: float
    expected_return: float
    risk_score: float
    reasoning: Dict[str, Any]
    feature_importance: Dict[str, float]
    alternative_actions: List[Dict[str, Any]]


@dataclass
class ExplanationReport:
    """Structure for explanation reports"""
    decision: TradeDecision
    market_context: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    feature_analysis: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    what_if_scenarios: List[Dict[str, Any]]
    learning_insights: Dict[str, Any]


class TradeExplainer:
    """
    Comprehensive trade decision explanation system
    """
    
    def __init__(self):
        self.decision_history = []
        self.explanation_templates = self._initialize_templates()
        self.feature_descriptions = self._initialize_feature_descriptions()
        
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize explanation templates"""
        return {
            "buy_signal": """
            ## 🟢 BUY Decision Analysis
            
            **Confidence Level**: {confidence:.1%}
            **Position Size**: {position_size:.2f} shares
            **Expected Return**: {expected_return:.2%}
            **Risk Score**: {risk_score:.2f}/10
            
            ### Primary Reasoning
            {primary_reasoning}
            
            ### Key Technical Indicators
            {technical_analysis}
            
            ### Risk Considerations
            {risk_analysis}
            
            ### Feature Importance
            {feature_importance}
            """,
            
            "sell_signal": """
            ## 🔴 SELL Decision Analysis
            
            **Confidence Level**: {confidence:.1%}
            **Position Size**: {position_size:.2f} shares
            **Expected Return**: {expected_return:.2%}
            **Risk Score**: {risk_score:.2f}/10
            
            ### Primary Reasoning
            {primary_reasoning}
            
            ### Key Technical Indicators
            {technical_analysis}
            
            ### Risk Considerations
            {risk_analysis}
            
            ### Feature Importance
            {feature_importance}
            """,
            
            "hold_signal": """
            ## 🟡 HOLD Decision Analysis
            
            **Confidence Level**: {confidence:.1%}
            **Reason**: Uncertainty or insufficient signal strength
            
            ### Market Analysis
            {market_analysis}
            
            ### Waiting For
            {waiting_conditions}
            
            ### Risk Assessment
            {risk_assessment}
            """
        }
    
    def _initialize_feature_descriptions(self) -> Dict[str, str]:
        """Initialize feature descriptions for explanations"""
        return {
            "rsi": "Relative Strength Index - measures momentum and overbought/oversold conditions",
            "macd": "MACD - trend-following momentum indicator showing relationship between two moving averages",
            "sma_20": "20-day Simple Moving Average - smooths price data to identify trends",
            "sma_50": "50-day Simple Moving Average - longer-term trend indicator",
            "volume": "Trading Volume - indicates strength of price movement",
            "volatility": "Price Volatility - measures rate and magnitude of price changes",
            "price_momentum": "Price Momentum - rate of price change over recent period",
            "support_level": "Support Level - price level where buying pressure emerges",
            "resistance_level": "Resistance Level - price level where selling pressure emerges",
            "market_sentiment": "Market Sentiment - overall market mood and investor attitude",
            "implied_volatility": "Implied Volatility - market's expectation of future volatility",
            "correlation_risk": "Correlation Risk - how this asset moves with others in portfolio",
            "liquidity_score": "Liquidity Score - ease of buying/selling without affecting price",
            "fundamental_score": "Fundamental Score - company's financial health and valuation",
            "news_sentiment": "News Sentiment - recent news impact on price expectations",
            "sector_trend": "Sector Trend - performance of the industry sector",
            "market_regime": "Market Regime - current market condition (bull/bear/sideways)",
            "risk_reward_ratio": "Risk/Reward Ratio - potential profit vs potential loss",
            "position_size_risk": "Position Size Risk - risk contribution to overall portfolio"
        }
    
    def explain_trade_decision(self, decision: TradeDecision, 
                             market_data: Dict[str, Any],
                             portfolio_state: Dict[str, Any]) -> ExplanationReport:
        """
        Generate comprehensive explanation for a trade decision
        """
        
        # Analyze market context
        market_context = self._analyze_market_context(market_data, decision)
        
        # Technical analysis explanation
        technical_analysis = self._explain_technical_analysis(decision, market_data)
        
        # Risk assessment
        risk_assessment = self._assess_risk_explanation(decision, portfolio_state)
        
        # Feature importance analysis
        feature_analysis = self._analyze_feature_importance(decision)
        
        # Confidence breakdown
        confidence_breakdown = self._breakdown_confidence(decision)
        
        # What-if scenarios
        what_if_scenarios = self._generate_what_if_scenarios(decision, market_data)
        
        # Learning insights
        learning_insights = self._generate_learning_insights(decision)
        
        return ExplanationReport(
            decision=decision,
            market_context=market_context,
            technical_analysis=technical_analysis,
            risk_assessment=risk_assessment,
            feature_analysis=feature_analysis,
            confidence_breakdown=confidence_breakdown,
            what_if_scenarios=what_if_scenarios,
            learning_insights=learning_insights
        )
    
    def _analyze_market_context(self, market_data: Dict[str, Any], 
                              decision: TradeDecision) -> Dict[str, Any]:
        """Analyze market context for the decision"""
        
        current_price = market_data.get('current_price', 0)
        volume = market_data.get('volume', 0)
        market_sentiment = market_data.get('sentiment', 'neutral')
        volatility = market_data.get('volatility', 0)
        
        # Market regime analysis
        if market_data.get('sma_20', 0) > market_data.get('sma_50', 0):
            trend = "bullish"
        elif market_data.get('sma_20', 0) < market_data.get('sma_50', 0):
            trend = "bearish"
        else:
            trend = "sideways"
        
        # Volume analysis
        avg_volume = market_data.get('avg_volume', volume)
        volume_strength = "high" if volume > avg_volume * 1.5 else "normal" if volume > avg_volume * 0.5 else "low"
        
        # Volatility analysis
        if volatility > 0.3:
            vol_level = "high"
        elif volatility > 0.2:
            vol_level = "moderate"
        else:
            vol_level = "low"
        
        return {
            "trend": trend,
            "volume_strength": volume_strength,
            "volatility_level": vol_level,
            "market_sentiment": market_sentiment,
            "price_level": self._get_price_level(current_price, market_data),
            "time_in_market": self._get_time_context(),
            "overall_condition": self._summarize_market_condition(trend, vol_level, volume_strength)
        }
    
    def _explain_technical_analysis(self, decision: TradeDecision, 
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain technical analysis factors"""
        
        explanations = []
        
        # RSI analysis
        rsi = market_data.get('rsi', 50)
        if rsi > 70:
            explanations.append(f"RSI ({rsi:.1f}) indicates overbought conditions - suggests potential pullback")
        elif rsi < 30:
            explanations.append(f"RSI ({rsi:.1f}) indicates oversold conditions - suggests potential bounce")
        else:
            explanations.append(f"RSI ({rsi:.1f}) is in neutral range")
        
        # MACD analysis
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        if macd > macd_signal:
            explanations.append("MACD is above signal line - bullish momentum")
        else:
            explanations.append("MACD is below signal line - bearish momentum")
        
        # Moving averages
        price = market_data.get('current_price', 0)
        sma_20 = market_data.get('sma_20', price)
        sma_50 = market_data.get('sma_50', price)
        
        if price > sma_20 > sma_50:
            explanations.append("Price is above both short and long-term moving averages - strong uptrend")
        elif price < sma_20 < sma_50:
            explanations.append("Price is below both moving averages - strong downtrend")
        else:
            explanations.append("Price is between moving averages - trend transition phase")
        
        # Volume confirmation
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        if decision.action in ['BUY', 'SELL'] and volume > avg_volume * 1.2:
            explanations.append(f"High volume ({volume:,.0f}) confirms the {decision.action.lower()} signal strength")
        
        return {
            "primary_signals": explanations[:3],
            "secondary_signals": explanations[3:],
            "overall_technical_score": self._calculate_technical_score(market_data),
            "key_levels": self._identify_key_levels(market_data)
        }
    
    def _assess_risk_explanation(self, decision: TradeDecision, 
                               portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess and explain risk factors"""
        
        risk_factors = []
        risk_level = "low"
        
        # Position size risk
        portfolio_value = portfolio_state.get('total_value', 100000)
        position_value = abs(decision.position_size * decision.expected_return)
        position_risk = position_value / portfolio_value
        
        if position_risk > 0.1:
            risk_factors.append(f"Large position size ({position_risk:.1%} of portfolio)")
            risk_level = "high"
        elif position_risk > 0.05:
            risk_factors.append(f"Moderate position size ({position_risk:.1%} of portfolio)")
            risk_level = "moderate"
        
        # Volatility risk
        if decision.risk_score > 7:
            risk_factors.append("High volatility environment increases uncertainty")
            risk_level = "high"
        elif decision.risk_score > 5:
            risk_factors.append("Moderate volatility requires careful monitoring")
        
        # Correlation risk
        current_exposure = portfolio_state.get('sector_exposure', {})
        symbol_sector = decision.symbol.split('.')[0]  # Simplified sector extraction
        
        if current_exposure.get(symbol_sector, 0) > 0.3:
            risk_factors.append(f"High exposure to {symbol_sector} sector")
        
        # Liquidity risk
        liquidity_score = decision.reasoning.get('liquidity_score', 0.8)
        if liquidity_score < 0.5:
            risk_factors.append("Low liquidity may affect execution")
            risk_level = "high"
        
        # Confidence risk
        if decision.confidence < 0.6:
            risk_factors.append("Low confidence in decision increases uncertainty")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "position_risk": position_risk,
            "volatility_risk": decision.risk_score,
            "liquidity_risk": liquidity_score,
            "correlation_risk": current_exposure.get(symbol_sector, 0),
            "recommended_stop_loss": self._calculate_stop_loss(decision),
            "recommended_position_size": self._recommend_position_size(decision, portfolio_state)
        }
    
    def _analyze_feature_importance(self, decision: TradeDecision) -> Dict[str, Any]:
        """Analyze and explain feature importance"""
        
        feature_importance = decision.feature_importance
        explanations = {}
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:5]:  # Top 5 features
            description = self.feature_descriptions.get(feature, f"Feature {feature}")
            
            # Generate explanation based on importance and value
            if importance > 0.2:
                strength = "very strong"
            elif importance > 0.1:
                strength = "strong"
            elif importance > 0.05:
                strength = "moderate"
            else:
                strength = "weak"
            
            explanations[feature] = {
                "importance": importance,
                "strength": strength,
                "description": description,
                "impact": self._explain_feature_impact(feature, decision)
            }
        
        return {
            "top_features": explanations,
            "feature_distribution": self._analyze_feature_distribution(feature_importance),
            "dominant_category": self._identify_dominant_feature_category(sorted_features),
            "feature_stability": self._assess_feature_stability(decision)
        }
    
    def _breakdown_confidence(self, decision: TradeDecision) -> Dict[str, float]:
        """Break down confidence by factor"""
        
        # This would be calculated based on the actual model's internal confidence components
        # For now, we'll simulate a breakdown
        
        base_confidence = decision.confidence
        
        # Simulate confidence components
        technical_confidence = base_confidence * 0.4
        fundamental_confidence = base_confidence * 0.3
        sentiment_confidence = base_confidence * 0.2
        risk_confidence = base_confidence * 0.1
        
        return {
            "technical_analysis": technical_confidence,
            "fundamental_analysis": fundamental_confidence,
            "market_sentiment": sentiment_confidence,
            "risk_assessment": risk_confidence,
            "total_confidence": base_confidence
        }
    
    def _generate_what_if_scenarios(self, decision: TradeDecision, 
                                  market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate what-if scenarios for the decision"""
        
        scenarios = []
        
        # Scenario 1: Market moves against position
        adverse_move = -0.05  # 5% adverse move
        adverse_pnl = decision.position_size * adverse_move
        scenarios.append({
            "scenario": "5% Market Move Against Position",
            "probability": 0.3,
            "impact": adverse_pnl,
            "action": "Consider tighter stop loss",
            "risk_level": "high"
        })
        
        # Scenario 2: Market moves favorably
        favorable_move = 0.05  # 5% favorable move
        favorable_pnl = decision.position_size * favorable_move
        scenarios.append({
            "scenario": "5% Market Move in Favor",
            "probability": 0.25,
            "impact": favorable_pnl,
            "action": "Consider taking partial profits",
            "risk_level": "low"
        })
        
        # Scenario 3: Volatility spike
        vol_spike = market_data.get('volatility', 0.2) * 2
        scenarios.append({
            "scenario": "Volatility Spike",
            "probability": 0.15,
            "impact": "Increased position risk and potential slippage",
            "action": "Reduce position size or tighten stops",
            "risk_level": "moderate"
        })
        
        # Scenario 4: No significant move
        scenarios.append({
            "scenario": "Sideways Market",
            "probability": 0.3,
            "impact": "Time decay and transaction costs",
            "action": "Consider holding period costs",
            "risk_level": "low"
        })
        
        return scenarios
    
    def _generate_learning_insights(self, decision: TradeDecision) -> Dict[str, Any]:
        """Generate learning insights from the decision"""
        
        insights = []
        
        # Pattern recognition
        if decision.confidence > 0.8:
            insights.append("High confidence suggests strong pattern recognition")
        
        # Risk management insights
        if decision.risk_score > 6:
            insights.append("High-risk environment - valuable learning opportunity for risk management")
        
        # Feature learning
        top_feature = max(decision.feature_importance.items(), key=lambda x: x[1])
        insights.append(f"Feature '{top_feature[0]}' was most influential - consider monitoring this indicator")
        
        # Decision consistency
        recent_decisions = self.decision_history[-5:] if len(self.decision_history) >= 5 else []
        if recent_decisions:
            consistent_actions = sum(1 for d in recent_decisions if d.action == decision.action)
            if consistent_actions >= 3:
                insights.append("Consistent decision pattern - strong conviction in strategy")
            else:
                insights.append("Changing decision pattern - adapting to new market conditions")
        
        return {
            "key_insights": insights,
            "learning_opportunities": self._identify_learning_opportunities(decision),
            "pattern_recognition": self._analyze_patterns(decision),
            "improvement_suggestions": self._suggest_improvements(decision)
        }
    
    def generate_explanation_report(self, explanation: ExplanationReport) -> str:
        """Generate human-readable explanation report"""
        
        decision = explanation.decision
        template = self.explanation_templates.get(f"{decision.action.lower()}_signal")
        
        if not template:
            template = "## Decision Analysis\n{analysis}"
        
        # Format technical analysis
        tech_analysis = "\n".join([f"• {signal}" for signal in explanation.technical_analysis["primary_signals"]])
        
        # Format feature importance
        feature_imp = "\n".join([
            f"• {feature}: {details['strength']} importance ({details['importance']:.1%})"
            for feature, details in explanation.feature_analysis["top_features"].items()
        ])
        
        # Format risk analysis
        risk_factors = "\n".join([f"• {factor}" for factor in explanation.risk_assessment["risk_factors"]])
        
        return template.format(
            confidence=decision.confidence,
            position_size=decision.position_size,
            expected_return=decision.expected_return,
            risk_score=decision.risk_score,
            primary_reasoning=decision.reasoning.get("primary_reason", "Based on comprehensive market analysis"),
            technical_analysis=tech_analysis,
            risk_analysis=risk_factors,
            feature_importance=feature_imp
        )
    
    def create_explanation_visualization(self, explanation: ExplanationReport) -> Dict[str, str]:
        """Create visualizations for the explanation"""
        
        visualizations = {}
        
        # Feature importance chart
        fig_importance = go.Figure(data=[
            go.Bar(
                x=list(explanation.feature_analysis["top_features"].keys()),
                y=[details["importance"] for details in explanation.feature_analysis["top_features"].values()],
                marker_color='lightblue'
            )
        ])
        fig_importance.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            template="plotly_white"
        )
        visualizations["feature_importance"] = fig_importance.to_html()
        
        # Confidence breakdown chart
        confidence_data = explanation.confidence_breakdown
        fig_confidence = go.Figure(data=[
            go.Pie(
                labels=list(confidence_data.keys())[:-1],  # Exclude total
                values=list(confidence_data.values())[:-1],
                hole=0.3
            )
        ])
        fig_confidence.update_layout(
            title="Confidence Breakdown",
            template="plotly_white"
        )
        visualizations["confidence_breakdown"] = fig_confidence.to_html()
        
        # Risk assessment chart
        risk_data = {
            "Position Risk": explanation.risk_assessment["position_risk"],
            "Volatility Risk": explanation.risk_assessment["volatility_risk"] / 10,
            "Liquidity Risk": 1 - explanation.risk_assessment["liquidity_risk"],
            "Correlation Risk": explanation.risk_assessment["correlation_risk"]
        }
        
        fig_risk = go.Figure(data=[
            go.Bar(
                x=list(risk_data.keys()),
                y=list(risk_data.values()),
                marker_color=['red' if v > 0.5 else 'orange' if v > 0.3 else 'green' for v in risk_data.values()]
            )
        ])
        fig_risk.update_layout(
            title="Risk Assessment",
            xaxis_title="Risk Type",
            yaxis_title="Risk Level",
            template="plotly_white"
        )
        visualizations["risk_assessment"] = fig_risk.to_html()
        
        # What-if scenarios chart
        scenarios = explanation.what_if_scenarios
        fig_scenarios = go.Figure(data=[
            go.Bar(
                x=[s["scenario"] for s in scenarios],
                y=[s["probability"] for s in scenarios],
                marker_color='lightgreen'
            )
        ])
        fig_scenarios.update_layout(
            title="What-If Scenario Probabilities",
            xaxis_title="Scenario",
            yaxis_title="Probability",
            template="plotly_white"
        )
        visualizations["what_if_scenarios"] = fig_scenarios.to_html()
        
        return visualizations
    
    def _get_price_level(self, current_price: float, market_data: Dict[str, Any]) -> str:
        """Determine price level relative to recent range"""
        
        high_52w = market_data.get('high_52w', current_price)
        low_52w = market_data.get('low_52w', current_price)
        
        percentile = (current_price - low_52w) / (high_52w - low_52w)
        
        if percentile > 0.8:
            return "near 52-week highs"
        elif percentile < 0.2:
            return "near 52-week lows"
        else:
            return "mid-range"
    
    def _get_time_context(self) -> str:
        """Get time context for the decision"""
        now = datetime.now()
        
        if now.hour < 10:
            return "pre-market"
        elif now.hour < 16:
            return "trading hours"
        else:
            return "after-hours"
    
    def _summarize_market_condition(self, trend: str, vol_level: str, volume: str) -> str:
        """Summarize overall market condition"""
        
        if trend == "bullish" and vol_level == "low" and volume == "normal":
            return "Stable bull market"
        elif trend == "bullish" and vol_level == "high":
            return "Volatile bull market"
        elif trend == "bearish" and vol_level == "high":
            return "Volatile bear market"
        elif trend == "sideways":
            return "Range-bound market"
        else:
            return f"{trend} market with {vol_level} volatility"
    
    def _calculate_technical_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall technical score"""
        
        score = 0.5  # Base score
        
        # RSI contribution
        rsi = market_data.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 0.1
        elif 30 <= rsi <= 70:
            score += 0.05
        
        # MACD contribution
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        if macd > macd_signal:
            score += 0.1
        
        # Moving average contribution
        price = market_data.get('current_price', 0)
        sma_20 = market_data.get('sma_20', price)
        if price > sma_20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_key_levels(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        
        current_price = market_data.get('current_price', 0)
        
        # Simplified key level identification
        support = current_price * 0.95  # 5% below current
        resistance = current_price * 1.05  # 5% above current
        
        return {
            "support": support,
            "resistance": resistance,
            "current_price": current_price
        }
    
    def _explain_feature_impact(self, feature: str, decision: TradeDecision) -> str:
        """Explain how a specific feature impacted the decision"""
        
        feature_value = decision.reasoning.get(f"{feature}_value", 0)
        
        if feature == "rsi":
            if decision.action == "BUY" and feature_value < 30:
                return "Oversold condition suggested buying opportunity"
            elif decision.action == "SELL" and feature_value > 70:
                return "Overbought condition suggested selling opportunity"
        
        elif feature == "macd":
            if decision.action == "BUY" and feature_value > 0:
                return "Positive MACD momentum supported buy decision"
            elif decision.action == "SELL" and feature_value < 0:
                return "Negative MACD momentum supported sell decision"
        
        elif feature == "volume":
            if feature_value > decision.reasoning.get("avg_volume", feature_value):
                return "High volume confirmed the strength of the signal"
        
        return f"Feature value of {feature_value:.2f} influenced the {decision.action.lower()} decision"
    
    def _analyze_feature_distribution(self, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the distribution of feature importance"""
        
        values = list(feature_importance.values())
        
        return {
            "total_features": len(feature_importance),
            "dominant_features": sum(1 for v in values if v > 0.1),
            "minor_features": sum(1 for v in values if v < 0.05),
            "concentration": max(values) / sum(values) if sum(values) > 0 else 0
        }
    
    def _identify_dominant_feature_category(self, sorted_features: List[Tuple[str, float]]) -> str:
        """Identify the dominant category of features"""
        
        # Simplified categorization
        technical_features = ['rsi', 'macd', 'sma_20', 'sma_50', 'volume', 'volatility']
        fundamental_features = ['fundamental_score', 'sector_trend']
        sentiment_features = ['market_sentiment', 'news_sentiment']
        
        categories = {"technical": 0, "fundamental": 0, "sentiment": 0}
        
        for feature, _ in sorted_features[:5]:  # Top 5 features
            if feature in technical_features:
                categories["technical"] += 1
            elif feature in fundamental_features:
                categories["fundamental"] += 1
            elif feature in sentiment_features:
                categories["sentiment"] += 1
        
        return max(categories, key=categories.get)
    
    def _assess_feature_stability(self, decision: TradeDecision) -> str:
        """Assess the stability of feature contributions"""
        
        # This would compare with historical feature importance
        # For now, return a simple assessment
        if decision.confidence > 0.8:
            return "stable"
        elif decision.confidence > 0.6:
            return "moderately stable"
        else:
            return "unstable"
    
    def _calculate_stop_loss(self, decision: TradeDecision) -> float:
        """Calculate recommended stop loss level"""
        
        # Simplified stop loss calculation
        risk_per_trade = 0.02  # 2% risk per trade
        
        if decision.action == "BUY":
            return decision.expected_return * (1 - risk_per_trade)
        elif decision.action == "SELL":
            return decision.expected_return * (1 + risk_per_trade)
        else:
            return decision.expected_return
    
    def _recommend_position_size(self, decision: TradeDecision, 
                               portfolio_state: Dict[str, Any]) -> float:
        """Recommend optimal position size"""
        
        portfolio_value = portfolio_state.get('total_value', 100000)
        risk_per_trade = 0.02  # 2% risk per trade
        stop_loss_distance = abs(decision.expected_return - self._calculate_stop_loss(decision))
        
        if stop_loss_distance > 0:
            recommended_size = (portfolio_value * risk_per_trade) / stop_loss_distance
        else:
            recommended_size = decision.position_size * 0.5  # Conservative
        
        return min(recommended_size, decision.position_size)
    
    def _identify_learning_opportunities(self, decision: TradeDecision) -> List[str]:
        """Identify learning opportunities from the decision"""
        
        opportunities = []
        
        if decision.confidence < 0.7:
            opportunities.append("Low confidence decisions - review feature selection and model parameters")
        
        if decision.risk_score > 6:
            opportunities.append("High-risk environments - study risk management techniques")
        
        if len(decision.feature_importance) > 10:
            opportunities.append("Complex feature interactions - consider feature reduction techniques")
        
        return opportunities
    
    def _analyze_patterns(self, decision: TradeDecision) -> Dict[str, Any]:
        """Analyze patterns in the decision"""
        
        return {
            "pattern_type": decision.reasoning.get("pattern_type", "unknown"),
            "pattern_strength": decision.confidence,
            "historical_performance": decision.reasoning.get("historical_performance", 0.5),
            "similar_situations": decision.reasoning.get("similar_situations", 0)
        }
    
    def _suggest_improvements(self, decision: TradeDecision) -> List[str]:
        """Suggest improvements for future decisions"""
        
        suggestions = []
        
        if decision.confidence < 0.8:
            suggestions.append("Consider additional features to improve confidence")
        
        if decision.risk_score > 5:
            suggestions.append("Implement stricter risk management for high-volatility environments")
        
        if len(decision.feature_importance) < 3:
            suggestions.append("Increase feature diversity for more robust decisions")
        
        return suggestions
    
    def add_decision_to_history(self, decision: TradeDecision):
        """Add decision to history for learning"""
        self.decision_history.append(decision)
        
        # Keep only last 100 decisions
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
    
    def get_decision_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of recent decisions"""
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        recent_decisions = [d for d in self.decision_history if d.timestamp > cutoff_date]
        
        if not recent_decisions:
            return {"message": "No recent decisions found"}
        
        # Calculate statistics
        total_decisions = len(recent_decisions)
        buy_decisions = sum(1 for d in recent_decisions if d.action == "BUY")
        sell_decisions = sum(1 for d in recent_decisions if d.action == "SELL")
        hold_decisions = sum(1 for d in recent_decisions if d.action == "HOLD")
        
        avg_confidence = np.mean([d.confidence for d in recent_decisions])
        avg_risk = np.mean([d.risk_score for d in recent_decisions])
        
        return {
            "period_days": days,
            "total_decisions": total_decisions,
            "buy_decisions": buy_decisions,
            "sell_decisions": sell_decisions,
            "hold_decisions": hold_decisions,
            "avg_confidence": avg_confidence,
            "avg_risk_score": avg_risk,
            "decision_distribution": {
                "BUY": buy_decisions / total_decisions,
                "SELL": sell_decisions / total_decisions,
                "HOLD": hold_decisions / total_decisions
            }
        }


# Factory function
def create_trade_explainer() -> TradeExplainer:
    """Create and return a TradeExplainer instance"""
    return TradeExplainer()


# Example usage
if __name__ == "__main__":
    explainer = create_trade_explainer()
    
    # Create a sample trade decision
    decision = TradeDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        action="BUY",
        confidence=0.75,
        position_size=100,
        expected_return=150.0,
        risk_score=4.5,
        reasoning={
            "primary_reason": "Strong technical indicators and positive sentiment",
            "rsi_value": 35,
            "macd_value": 2.5,
            "volume_value": 50000000,
            "avg_volume": 40000000
        },
        feature_importance={
            "rsi": 0.25,
            "macd": 0.20,
            "volume": 0.15,
            "market_sentiment": 0.12,
            "volatility": 0.10
        },
        alternative_actions=[
            {"action": "HOLD", "confidence": 0.15, "reason": "Waiting for better entry"},
            {"action": "SELL", "confidence": 0.10, "reason": "Taking profits on existing position"}
        ]
    )
    
    # Sample market data
    market_data = {
        "current_price": 148.50,
        "volume": 50000000,
        "avg_volume": 40000000,
        "rsi": 35,
        "macd": 2.5,
        "macd_signal": 2.0,
        "sma_20": 145.0,
        "sma_50": 140.0,
        "volatility": 0.25,
        "sentiment": "positive"
    }
    
    # Sample portfolio state
    portfolio_state = {
        "total_value": 100000,
        "sector_exposure": {"technology": 0.3, "healthcare": 0.2}
    }
    
    # Generate explanation
    explanation = explainer.explain_trade_decision(decision, market_data, portfolio_state)
    
    # Generate report
    report = explainer.generate_explanation_report(explanation)
    print("Trade Explanation Report:")
    print(report)
    
    # Create visualizations
    visualizations = explainer.create_explanation_visualization(explanation)
    print(f"\nGenerated {len(visualizations)} visualizations")
    
    # Add to history
    explainer.add_decision_to_history(decision)
    
    # Get summary
    summary = explainer.get_decision_summary(7)
    print(f"\nRecent decisions summary: {summary}")