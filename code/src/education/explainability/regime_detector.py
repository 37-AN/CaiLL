"""
Market Regime Detector - Identify and Explain Market Conditions

This module identifies different market regimes (bull, bear, sideways, volatile, etc.)
and provides explanations for regime transitions and characteristics.
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


@dataclass
class MarketRegime:
    """Structure for market regime definition"""
    name: str
    description: str
    characteristics: Dict[str, Any]
    typical_indicators: Dict[str, float]
    trading_implications: List[str]
    risk_level: str
    expected_duration: str
    transition_probabilities: Dict[str, float]


@dataclass
class RegimeDetectionResult:
    """Structure for regime detection results"""
    timestamp: datetime
    current_regime: str
    confidence: float
    regime_probability: Dict[str, float]
    key_indicators: Dict[str, float]
    transition_signals: List[str]
    historical_context: Dict[str, Any]
    expected_next_regime: str
    time_in_regime: int


class MarketRegimeDetector:
    """
    Comprehensive market regime detection and explanation system
    """
    
    def __init__(self):
        self.regime_definitions = self._initialize_regime_definitions()
        self.detection_models = {}
        self.regime_history = []
        self.transition_matrix = {}
        
    def _initialize_regime_definitions(self) -> Dict[str, MarketRegime]:
        """Initialize predefined market regimes"""
        
        return {
            "bull_market": MarketRegime(
                name="Bull Market",
                description="Market characterized by rising prices, optimism, and investor confidence",
                characteristics={
                    "price_trend": "strong_upward",
                    "volatility": "low_to_moderate",
                    "volume": "increasing",
                    "breadth": "strong",
                    "sentiment": "optimistic"
                },
                typical_indicators={
                    "sma_20_above_sma_50": True,
                    "rsi_avg": 55,
                    "volatility_ratio": 0.8,
                    "advance_decline_ratio": 1.5,
                    "volume_ratio": 1.2
                },
                trading_implications=[
                    "Buy dips and hold positions",
                    "Focus on growth stocks",
                    "Use trend-following strategies",
                    "Moderate position sizing"
                ],
                risk_level="moderate",
                expected_duration="months_to_years",
                transition_probabilities={
                    "bear_market": 0.1,
                    "sideways_market": 0.3,
                    "volatile_market": 0.2,
                    "bull_market": 0.4
                }
            ),
            
            "bear_market": MarketRegime(
                name="Bear Market",
                description="Market characterized by falling prices, pessimism, and investor fear",
                characteristics={
                    "price_trend": "strong_downward",
                    "volatility": "high",
                    "volume": "decreasing_then_spike",
                    "breadth": "weak",
                    "sentiment": "pessimistic"
                },
                typical_indicators={
                    "sma_20_below_sma_50": True,
                    "rsi_avg": 35,
                    "volatility_ratio": 1.5,
                    "advance_decline_ratio": 0.5,
                    "volume_ratio": 0.8
                },
                trading_implications=[
                    "Focus on capital preservation",
                    "Consider defensive sectors",
                    "Use short-selling strategies",
                    "Reduce position sizes"
                ],
                risk_level="high",
                expected_duration="months_to_years",
                transition_probabilities={
                    "bull_market": 0.15,
                    "sideways_market": 0.35,
                    "volatile_market": 0.25,
                    "bear_market": 0.25
                }
            ),
            
            "sideways_market": MarketRegime(
                name="Sideways Market",
                description="Market with no clear trend, trading in a range",
                characteristics={
                    "price_trend": "horizontal",
                    "volatility": "low",
                    "volume": "average",
                    "breadth": "mixed",
                    "sentiment": "neutral"
                },
                typical_indicators={
                    "sma_20_near_sma_50": True,
                    "rsi_avg": 50,
                    "volatility_ratio": 0.6,
                    "advance_decline_ratio": 1.0,
                    "volume_ratio": 1.0
                },
                trading_implications=[
                    "Use range-trading strategies",
                    "Sell at resistance, buy at support",
                    "Focus on mean reversion",
                    "Be patient with entries"
                ],
                risk_level="low",
                expected_duration="weeks_to_months",
                transition_probabilities={
                    "bull_market": 0.25,
                    "bear_market": 0.25,
                    "volatile_market": 0.2,
                    "sideways_market": 0.3
                }
            ),
            
            "volatile_market": MarketRegime(
                name="Volatile Market",
                description="Market with large price swings and uncertainty",
                characteristics={
                    "price_trend": "erratic",
                    "volatility": "very_high",
                    "volume": "high",
                    "breadth": "erratic",
                    "sentiment": "uncertain"
                },
                typical_indicators={
                    "sma_20_crossing_sma_50": True,
                    "rsi_avg": 45,
                    "volatility_ratio": 2.0,
                    "advance_decline_ratio": 1.0,
                    "volume_ratio": 1.5
                },
                trading_implications=[
                    "Reduce position sizes significantly",
                    "Use options for hedging",
                    "Focus on risk management",
                    "Consider staying in cash"
                ],
                risk_level="very_high",
                expected_duration="days_to_weeks",
                transition_probabilities={
                    "bull_market": 0.2,
                    "bear_market": 0.2,
                    "sideways_market": 0.3,
                    "volatile_market": 0.3
                }
            ),
            
            "transition_market": MarketRegime(
                name="Transition Market",
                description="Market in transition between regimes",
                characteristics={
                    "price_trend": "changing",
                    "volatility": "moderate_to_high",
                    "volume": "increasing",
                    "breadth": "diverging",
                    "sentiment": "shifting"
                },
                typical_indicators={
                    "sma_20_approaching_sma_50": True,
                    "rsi_avg": 48,
                    "volatility_ratio": 1.2,
                    "advance_decline_ratio": 1.1,
                    "volume_ratio": 1.3
                },
                trading_implications=[
                    "Wait for clear signals",
                    "Reduce exposure temporarily",
                    "Monitor for confirmation",
                    "Prepare for new regime"
                ],
                risk_level="moderate_to_high",
                expected_duration="days_to_weeks",
                transition_probabilities={
                    "bull_market": 0.3,
                    "bear_market": 0.3,
                    "sideways_market": 0.2,
                    "volatile_market": 0.2
                }
            )
        }
    
    def detect_current_regime(self, market_data: pd.DataFrame, 
                            method: str = "ensemble") -> RegimeDetectionResult:
        """
        Detect the current market regime using multiple methods
        """
        
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(market_data)
        
        # Detect regime using different methods
        if method == "ensemble":
            regime_probabilities = self._ensemble_detection(indicators)
        elif method == "hmm":
            regime_probabilities = self._hmm_detection(market_data)
        elif method == "clustering":
            regime_probabilities = self._clustering_detection(indicators)
        else:
            regime_probabilities = self._rule_based_detection(indicators)
        
        # Determine current regime
        current_regime = max(regime_probabilities.items(), key=lambda x: x[1])
        confidence = current_regime[1]
        
        # Analyze transition signals
        transition_signals = self._detect_transition_signals(indicators, market_data)
        
        # Calculate time in current regime
        time_in_regime = self._calculate_time_in_regime(current_regime[0])
        
        # Predict next regime
        expected_next_regime = self._predict_next_regime(current_regime[0], transition_signals)
        
        # Get historical context
        historical_context = self._get_historical_context(market_data)
        
        result = RegimeDetectionResult(
            timestamp=datetime.now(),
            current_regime=current_regime[0],
            confidence=confidence,
            regime_probability=regime_probabilities,
            key_indicators=indicators,
            transition_signals=transition_signals,
            historical_context=historical_context,
            expected_next_regime=expected_next_regime,
            time_in_regime=time_in_regime
        )
        
        # Add to history
        self.regime_history.append(result)
        
        return result
    
    def _calculate_regime_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators for regime detection"""
        
        indicators = {}
        
        # Price trend indicators
        if 'close' in market_data.columns:
            prices = market_data['close']
            
            # Moving averages
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                current_sma_20 = sma_20.iloc[-1]
                current_sma_50 = sma_50.iloc[-1]
                
                indicators['sma_20_above_sma_50'] = float(current_sma_20 > current_sma_50)
                indicators['sma_ratio'] = float(current_sma_20 / current_sma_50)
                indicators['price_above_sma_20'] = float(prices.iloc[-1] > current_sma_20)
                indicators['price_above_sma_50'] = float(prices.iloc[-1] > current_sma_50)
            
            # Returns and volatility
            returns = prices.pct_change().dropna()
            if len(returns) > 0:
                indicators['return_5d'] = float(returns.tail(5).mean())
                indicators['return_20d'] = float(returns.tail(20).mean())
                indicators['volatility_20d'] = float(returns.tail(20).std())
                indicators['volatility_ratio'] = float(returns.tail(20).std() / returns.tail(60).std()) if len(returns) > 60 else 1.0
        
        # Volume indicators
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            volume_sma = volume.rolling(20).mean()
            
            if len(volume_sma) > 0:
                indicators['volume_ratio'] = float(volume.iloc[-1] / volume_sma.iloc[-1])
                indicators['volume_trend'] = float(volume.tail(5).mean() / volume.tail(20).mean())
        
        # RSI indicator
        if 'close' in market_data.columns:
            rsi = self._calculate_rsi(market_data['close'])
            if len(rsi) > 0:
                indicators['rsi'] = float(rsi.iloc[-1])
                indicators['rsi_avg'] = float(rsi.tail(20).mean())
        
        # Breadth indicators (if available)
        if 'advance_decline' in market_data.columns:
            ad_ratio = market_data['advance_decline']
            if len(ad_ratio) > 0:
                indicators['advance_decline_ratio'] = float(ad_ratio.iloc[-1])
                indicators['breadth_trend'] = float(ad_ratio.tail(5).mean() / ad_ratio.tail(20).mean())
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _ensemble_detection(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Ensemble method for regime detection"""
        
        # Combine multiple detection methods
        rule_based_probs = self._rule_based_detection(indicators)
        
        # Weight different methods
        ensemble_probs = {}
        for regime in self.regime_definitions.keys():
            ensemble_probs[regime] = rule_based_probs.get(regime, 0.0)
        
        # Normalize probabilities
        total_prob = sum(ensemble_probs.values())
        if total_prob > 0:
            ensemble_probs = {k: v/total_prob for k, v in ensemble_probs.items()}
        
        return ensemble_probs
    
    def _rule_based_detection(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Rule-based regime detection"""
        
        probabilities = {}
        
        # Bull market conditions
        bull_score = 0
        if indicators.get('sma_20_above_sma_50', 0) > 0.5:
            bull_score += 0.3
        if indicators.get('return_20d', 0) > 0.02:
            bull_score += 0.3
        if indicators.get('rsi_avg', 50) > 52:
            bull_score += 0.2
        if indicators.get('volume_ratio', 1.0) > 1.1:
            bull_score += 0.2
        probabilities['bull_market'] = bull_score
        
        # Bear market conditions
        bear_score = 0
        if indicators.get('sma_20_above_sma_50', 1) < 0.5:
            bear_score += 0.3
        if indicators.get('return_20d', 0) < -0.02:
            bear_score += 0.3
        if indicators.get('rsi_avg', 50) < 48:
            bear_score += 0.2
        if indicators.get('volatility_ratio', 1.0) > 1.3:
            bear_score += 0.2
        probabilities['bear_market'] = bear_score
        
        # Sideways market conditions
        sideways_score = 0
        if 0.8 < indicators.get('sma_ratio', 1.0) < 1.2:
            sideways_score += 0.3
        if abs(indicators.get('return_20d', 0)) < 0.01:
            sideways_score += 0.3
        if 48 < indicators.get('rsi_avg', 50) < 52:
            sideways_score += 0.2
        if indicators.get('volatility_ratio', 1.0) < 0.8:
            sideways_score += 0.2
        probabilities['sideways_market'] = sideways_score
        
        # Volatile market conditions
        volatile_score = 0
        if indicators.get('volatility_ratio', 1.0) > 1.5:
            volatile_score += 0.4
        if abs(indicators.get('return_5d', 0)) > 0.05:
            volatile_score += 0.3
        if indicators.get('volume_ratio', 1.0) > 1.3:
            volatile_score += 0.3
        probabilities['volatile_market'] = volatile_score
        
        # Transition market conditions
        transition_score = 0
        if 0.9 < indicators.get('sma_ratio', 1.0) < 1.1:
            transition_score += 0.3
        if indicators.get('volume_trend', 1.0) > 1.2:
            transition_score += 0.3
        if abs(indicators.get('return_5d', 0)) > 0.03:
            transition_score += 0.2
        if indicators.get('breadth_trend', 1.0) != 1.0:
            transition_score += 0.2
        probabilities['transition_market'] = transition_score
        
        return probabilities
    
    def _hmm_detection(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Hidden Markov Model for regime detection"""
        
        # This is a simplified HMM implementation
        # In practice, you'd use more sophisticated features and tuning
        
        try:
            # Prepare features
            features = []
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                features.append(returns.values)
            
            if len(features) == 0 or len(features[0]) < 50:
                return self._rule_based_detection(self._calculate_regime_indicators(market_data))
            
            # Stack features
            X = np.column_stack(features)
            
            # Fit HMM with 5 states (one for each regime)
            model = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
            model.fit(X.reshape(-1, 1))
            
            # Get the most recent state
            recent_state = model.predict(X[-1:].reshape(-1, 1))[0]
            
            # Map HMM states to regimes (simplified mapping)
            regime_mapping = ['bull_market', 'bear_market', 'sideways_market', 
                            'volatile_market', 'transition_market']
            
            probabilities = {}
            current_regime = regime_mapping[recent_state % 5]
            probabilities[current_regime] = 0.7
            
            # Add some probability to neighboring regimes
            for regime in regime_mapping:
                if regime != current_regime:
                    probabilities[regime] = 0.075
            
            return probabilities
            
        except Exception as e:
            print(f"HMM detection failed: {e}")
            return self._rule_based_detection(self._calculate_regime_indicators(market_data))
    
    def _clustering_detection(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Clustering-based regime detection"""
        
        # This would use historical data to train clusters
        # For now, return rule-based detection
        
        return self._rule_based_detection(indicators)
    
    def _detect_transition_signals(self, indicators: Dict[str, float], 
                                 market_data: pd.DataFrame) -> List[str]:
        """Detect signals indicating potential regime transitions"""
        
        signals = []
        
        # Moving average crossover signals
        sma_ratio = indicators.get('sma_ratio', 1.0)
        if 0.95 < sma_ratio < 1.05:
            signals.append("Moving averages converging - potential trend change")
        
        # Volatility expansion
        vol_ratio = indicators.get('volatility_ratio', 1.0)
        if vol_ratio > 1.5:
            signals.append("Volatility expanding - potential regime shift")
        
        # Volume anomaly
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            signals.append("Unusual volume - possible institutional activity")
        
        # RSI divergence
        rsi = indicators.get('rsi', 50)
        if rsi > 70 or rsi < 30:
            signals.append("Extreme RSI levels - potential reversal")
        
        # Price momentum change
        return_5d = indicators.get('return_5d', 0)
        return_20d = indicators.get('return_20d', 0)
        if abs(return_5d - return_20d) > 0.05:
            signals.append("Momentum acceleration - trend strengthening or reversing")
        
        return signals
    
    def _calculate_time_in_regime(self, current_regime: str) -> int:
        """Calculate how long we've been in the current regime"""
        
        if not self.regime_history:
            return 0
        
        time_in_regime = 0
        for result in reversed(self.regime_history):
            if result.current_regime == current_regime:
                time_in_regime += 1
            else:
                break
        
        return time_in_regime
    
    def _predict_next_regime(self, current_regime: str, 
                           transition_signals: List[str]) -> str:
        """Predict the most likely next regime"""
        
        if current_regime in self.regime_definitions:
            transition_probs = self.regime_definitions[current_regime].transition_probabilities
            
            # Adjust probabilities based on transition signals
            adjusted_probs = transition_probs.copy()
            
            if "Moving averages converging" in " ".join(transition_signals):
                adjusted_probs['transition_market'] += 0.1
                adjusted_probs['sideways_market'] += 0.1
            
            if "Volatility expanding" in " ".join(transition_signals):
                adjusted_probs['volatile_market'] += 0.15
            
            # Normalize probabilities
            total_prob = sum(adjusted_probs.values())
            if total_prob > 0:
                adjusted_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
            
            # Return most likely next regime
            return max(adjusted_probs.items(), key=lambda x: x[1])[0]
        
        return "unknown"
    
    def _get_historical_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get historical context for current market conditions"""
        
        context = {}
        
        if 'close' in market_data.columns:
            prices = market_data['close']
            
            # 52-week high/low
            if len(prices) >= 252:
                high_52w = prices.tail(252).max()
                low_52w = prices.tail(252).min()
                current_price = prices.iloc[-1]
                
                context['percent_from_high'] = ((current_price - high_52w) / high_52w) * 100
                context['percent_from_low'] = ((current_price - low_52w) / low_52w) * 100
                context['price_percentile'] = ((current_price - low_52w) / (high_52w - low_52w)) * 100
            
            # Recent performance
            returns = prices.pct_change().dropna()
            if len(returns) > 0:
                context['return_1m'] = float(returns.tail(21).sum())
                context['return_3m'] = float(returns.tail(63).sum())
                context['return_6m'] = float(returns.tail(126).sum())
                context['return_1y'] = float(returns.tail(252).sum()) if len(returns) >= 252 else None
        
        return context
    
    def generate_regime_explanation(self, detection_result: RegimeDetectionResult) -> str:
        """Generate human-readable explanation of the regime detection"""
        
        regime = self.regime_definitions.get(detection_result.current_regime)
        if not regime:
            return "Unknown regime detected"
        
        explanation = f"# Market Regime Analysis\n\n"
        explanation += f"## Current Regime: {regime.name}\n\n"
        explanation += f"**Confidence**: {detection_result.confidence:.1%}\n"
        explanation += f"**Time in Regime**: {detection_result.time_in_regime} periods\n\n"
        
        explanation += f"### Description\n{regime.description}\n\n"
        
        explanation += "### Current Market Characteristics\n"
        for characteristic, value in regime.characteristics.items():
            explanation += f"- **{characteristic.replace('_', ' ').title()}**: {value.replace('_', ' ').title()}\n"
        explanation += "\n"
        
        explanation += "### Key Indicators\n"
        for indicator, value in detection_result.key_indicators.items():
            explanation += f"- **{indicator.replace('_', ' ').title()}**: {value:.3f}\n"
        explanation += "\n"
        
        if detection_result.transition_signals:
            explanation += "### Transition Signals\n"
            for signal in detection_result.transition_signals:
                explanation += f"- ⚠️ {signal}\n"
            explanation += "\n"
        
        explanation += "### Trading Implications\n"
        for implication in regime.trading_implications:
            explanation += f"- {implication}\n"
        explanation += "\n"
        
        explanation += f"### Risk Level: {regime.risk_level.replace('_', ' ').title()}\n\n"
        
        explanation += f"### Expected Duration: {regime.expected_duration.replace('_', ' ').title()}\n\n"
        
        if detection_result.expected_next_regime != "unknown":
            next_regime = self.regime_definitions.get(detection_result.expected_next_regime)
            if next_regime:
                explanation += f"### Expected Next Regime: {next_regime.name}\n"
                explanation += f"Based on transition patterns and current signals\n\n"
        
        explanation += "### Historical Context\n"
        for key, value in detection_result.historical_context.items():
            if value is not None:
                if 'percent' in key:
                    explanation += f"- **{key.replace('_', ' ').title()}**: {value:.1f}%\n"
                else:
                    explanation += f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n"
        
        return explanation
    
    def create_regime_visualization(self, market_data: pd.DataFrame, 
                                  detection_result: RegimeDetectionResult) -> Dict[str, str]:
        """Create visualizations for regime analysis"""
        
        visualizations = {}
        
        # Price chart with regime annotation
        if 'close' in market_data.columns:
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            ))
            
            # Add moving averages
            if len(market_data) >= 50:
                sma_20 = market_data['close'].rolling(20).mean()
                sma_50 = market_data['close'].rolling(50).mean()
                
                fig.add_trace(go.Scatter(
                    x=market_data.index,
                    y=sma_20,
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=market_data.index,
                    y=sma_50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', dash='dash')
                ))
            
            # Add regime annotation
            regime_colors = {
                'bull_market': 'lightgreen',
                'bear_market': 'lightcoral',
                'sideways_market': 'lightyellow',
                'volatile_market': 'lightpink',
                'transition_market': 'lightblue'
            }
            
            fig.add_vrect(
                x0=market_data.index[-20],
                x1=market_data.index[-1],
                fillcolor=regime_colors.get(detection_result.current_regime, 'lightgray'),
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text=detection_result.current_regime.replace('_', ' ').title()
            )
            
            fig.update_layout(
                title=f"Price Chart - Current Regime: {detection_result.current_regime.replace('_', ' ').title()}",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=500
            )
            
            visualizations["price_chart"] = fig.to_html()
        
        # Regime probability chart
        fig_prob = go.Figure(data=[
            go.Bar(
                x=list(detection_result.regime_probability.keys()),
                y=list(detection_result.regime_probability.values()),
                marker_color=['green' if k == detection_result.current_regime else 'lightblue' 
                             for k in detection_result.regime_probability.keys()]
            )
        ])
        
        fig_prob.update_layout(
            title="Regime Probability Distribution",
            xaxis_title="Regime",
            yaxis_title="Probability",
            template="plotly_white"
        )
        
        visualizations["probability_chart"] = fig_prob.to_html()
        
        # Indicator dashboard
        indicators = detection_result.key_indicators
        if indicators:
            fig_indicators = go.Figure(data=[
                go.Bar(
                    x=list(indicators.keys()),
                    y=list(indicators.values()),
                    marker_color='lightsteelblue'
                )
            ])
            
            fig_indicators.update_layout(
                title="Key Regime Indicators",
                xaxis_title="Indicator",
                yaxis_title="Value",
                template="plotly_white"
            )
            
            visualizations["indicators_chart"] = fig_indicators.to_html()
        
        return visualizations
    
    def analyze_regime_transitions(self, lookback_days: int = 252) -> Dict[str, Any]:
        """Analyze historical regime transitions"""
        
        if len(self.regime_history) < 2:
            return {"message": "Insufficient history for transition analysis"}
        
        # Filter to recent history
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = [r for r in self.regime_history if r.timestamp > cutoff_date]
        
        if len(recent_history) < 2:
            return {"message": "Insufficient recent history for transition analysis"}
        
        # Count transitions
        transitions = {}
        for i in range(1, len(recent_history)):
            from_regime = recent_history[i-1].current_regime
            to_regime = recent_history[i].current_regime
            
            if from_regime != to_regime:
                transition_key = f"{from_regime} -> {to_regime}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        # Calculate transition frequencies
        total_transitions = sum(transitions.values())
        transition_frequencies = {k: v/total_transitions for k, v in transitions.items()}
        
        # Calculate average duration in each regime
        regime_durations = {}
        current_regime = None
        current_duration = 0
        
        for result in recent_history:
            if result.current_regime != current_regime:
                if current_regime is not None:
                    regime_durations[current_regime] = regime_durations.get(current_regime, [])
                    regime_durations[current_regime].append(current_duration)
                current_regime = result.current_regime
                current_duration = 1
            else:
                current_duration += 1
        
        # Add final regime duration
        if current_regime is not None:
            regime_durations[current_regime] = regime_durations.get(current_regime, [])
            regime_durations[current_regime].append(current_duration)
        
        # Calculate average durations
        avg_durations = {}
        for regime, durations in regime_durations.items():
            avg_durations[regime] = np.mean(durations)
        
        return {
            "analysis_period_days": lookback_days,
            "total_transitions": total_transitions,
            "transition_frequencies": transition_frequencies,
            "average_regime_durations": avg_durations,
            "most_common_transition": max(transition_frequencies.items(), key=lambda x: x[1]) if transition_frequencies else None,
            "regime_stability": {regime: np.mean(durations) for regime, durations in regime_durations.items()}
        }
    
    def export_regime_data(self, format: str = "json") -> str:
        """Export regime detection data"""
        
        if not self.regime_history:
            return "No regime history available"
        
        export_data = []
        for result in self.regime_history:
            export_data.append({
                "timestamp": result.timestamp.isoformat(),
                "current_regime": result.current_regime,
                "confidence": result.confidence,
                "regime_probability": result.regime_probability,
                "key_indicators": result.key_indicators,
                "time_in_regime": result.time_in_regime
            })
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        elif format == "csv":
            df = pd.DataFrame(export_data)
            return df.to_csv(index=False)
        else:
            return "Unsupported format"


# Factory function
def create_market_regime_detector() -> MarketRegimeDetector:
    """Create and return a MarketRegimeDetector instance"""
    return MarketRegimeDetector()


# Example usage
if __name__ == "__main__":
    detector = create_market_regime_detector()
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Simulate price data with different regimes
    prices = [100]
    for i in range(1, n_days):
        # Bull market regime
        if i < n_days * 0.3:
            change = np.random.normal(0.001, 0.02)
        # Sideways market
        elif i < n_days * 0.6:
            change = np.random.normal(0.0, 0.01)
        # Bear market
        else:
            change = np.random.normal(-0.001, 0.025)
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create DataFrame
    market_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_days),
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices]
    }, index=dates)
    
    # Detect current regime
    detection_result = detector.detect_current_regime(market_data)
    
    print(f"Current Regime: {detection_result.current_regime}")
    print(f"Confidence: {detection_result.confidence:.1%}")
    print(f"Time in Regime: {detection_result.time_in_regime}")
    
    # Generate explanation
    explanation = detector.generate_regime_explanation(detection_result)
    print(f"\nExplanation generated ({len(explanation)} characters)")
    
    # Create visualizations
    visualizations = detector.create_regime_visualization(market_data, detection_result)
    print(f"Generated {len(visualizations)} visualizations")
    
    # Analyze transitions
    # Add some more history for transition analysis
    for _ in range(10):
        detector.detect_current_regime(market_data)
    
    transition_analysis = detector.analyze_regime_transitions()
    print(f"\nTransition Analysis: {transition_analysis}")
    
    # Export data
    export_data = detector.export_regime_data("json")
    print(f"Exported regime data ({len(export_data)} characters)")