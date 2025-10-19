"""
Feature Pipeline Module - Phase 2.3

This module creates a unified feature engineering pipeline that combines
technical indicators, sentiment analysis, and market microstructure features
into a comprehensive feature set for AI model training.

Educational Note:
Feature engineering is the process of transforming raw data into features
that better represent the underlying problem to the predictive models.
Good features are more important than complex algorithms.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from collections import defaultdict
import warnings

# Import our feature modules
from .technical_indicators import TechnicalIndicators, IndicatorResult, IndicatorType
from .sentiment_analyzer import FinancialSentimentAnalyzer, SentimentResult, MarketSentiment
from .market_microstructure import MarketMicrostructureAnalyzer, MicrostructureFeatures

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MICROSTRUCTURE = "microstructure"
    DERIVED = "derived"
    TARGET = "target"

class DataType(Enum):
    """Data types for features"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIME_SERIES = "time_series"
    TEXT = "text"

@dataclass
class Feature:
    """Individual feature definition"""
    name: str
    value: Union[float, int, str, List, np.ndarray]
    feature_type: FeatureType
    data_type: DataType
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureSet:
    """Collection of features for a specific symbol/time"""
    symbol: str
    timestamp: datetime
    features: Dict[str, Feature] = field(default_factory=dict)
    target_variables: Dict[str, float] = field(default_factory=dict)
    
    def add_feature(self, feature: Feature):
        """Add a feature to the set"""
        self.features[feature.name] = feature
    
    def get_feature_vector(self, feature_names: List[str]) -> np.ndarray:
        """Get feature vector as numpy array"""
        values = []
        for name in feature_names:
            if name in self.features:
                value = self.features[name].value
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, np.ndarray):
                    values.append(np.mean(value))  # Aggregate arrays
                else:
                    values.append(0)  # Default for non-numeric
            else:
                values.append(0)  # Default for missing features
        
        return np.array(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'features': {name: feature.value for name, feature in self.features.items()},
            'target_variables': self.target_variables
        }

@dataclass
class FeatureConfig:
    """Configuration for feature generation"""
    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma_20', 'sma_50', 'ema_20', 'ema_50', 'rsi', 'macd', 'bollinger_bands',
        'atr', 'adx', 'stochastic', 'williams_r', 'cci', 'roc', 'momentum',
        'obv', 'vwap', 'mfi', 'ad_line'
    ])
    
    # Sentiment analysis
    sentiment_sources: List[str] = field(default_factory=lambda: [
        'news', 'social_media', 'analyst'
    ])
    sentiment_window: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    # Microstructure features
    microstructure_enabled: bool = True
    order_book_depth: int = 10
    
    # Feature engineering
    create_derived_features: bool = True
    normalize_features: bool = True
    handle_missing_values: bool = True
    
    # Target variables
    target_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])  # periods ahead
    target_types: List[str] = field(default_factory=lambda: ['returns', 'volatility', 'direction'])

class FeaturePipeline:
    """
    Unified Feature Engineering Pipeline
    
    Educational Notes:
    - Combines multiple feature types for comprehensive analysis
    - Handles feature scaling and normalization
    - Creates derived features from combinations
    - Manages feature timing and alignment
    - Provides feature importance and selection
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Initialize feature generators
        self.technical_analyzer = TechnicalIndicators()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        
        # Feature storage
        self.feature_history: Dict[str, List[FeatureSet]] = defaultdict(list)
        self.feature_importance: Dict[str, float] = {}
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        
        # Feature cache for performance
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        logger.info("FeaturePipeline initialized")
    
    def _get_cache_key(self, symbol: str, timestamp: datetime, feature_type: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{timestamp.isoformat()}_{feature_type}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache[cache_key].get('timestamp', datetime.min)
        return (datetime.now() - cache_time).total_seconds() < self._cache_timeout
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache"""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['value']
        return None
    
    def _set_cache(self, cache_key: str, value: Any):
        """Set value in cache"""
        self._cache[cache_key] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def generate_technical_features(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Feature]:
        """
        Generate technical indicator features
        
        Educational: Technical indicators capture price patterns and momentum
        that have historically shown predictive power.
        """
        features = {}
        timestamp = datetime.now()
        
        try:
            # Validate market data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                logger.warning(f"Missing required columns for technical analysis: {symbol}")
                return features
            
            # Check cache
            cache_key = self._get_cache_key(symbol, timestamp, "technical")
            cached_features = self._get_from_cache(cache_key)
            if cached_features:
                return cached_features
            
            # Calculate indicators
            all_indicators = self.technical_analyzer.calculate_all_indicators(market_data)
            
            # Convert to features
            for indicator_name, indicator_result in all_indicators.items():
                if isinstance(indicator_result, dict):
                    # Handle grouped indicators (like MACD, Bollinger Bands)
                    for sub_name, sub_result in indicator_result.items():
                        if hasattr(sub_result, 'values') and len(sub_result.values) > 0:
                            latest_value = sub_result.values.iloc[-1]
                            confidence = getattr(sub_result, 'confidence', 1.0)
                            
                            feature = Feature(
                                name=f"{indicator_name}_{sub_name}",
                                value=latest_value,
                                feature_type=FeatureType.TECHNICAL,
                                data_type=DataType.NUMERIC,
                                timestamp=timestamp,
                                confidence=confidence,
                                metadata={
                                    'indicator_type': sub_result.type.value if hasattr(sub_result, 'type') else 'unknown',
                                    'full_history': sub_result.values.tolist() if hasattr(sub_result.values, 'tolist') else []
                                }
                            )
                            features[feature.name] = feature
                else:
                    # Handle single indicators
                    if hasattr(indicator_result, 'values') and len(indicator_result.values) > 0:
                        latest_value = indicator_result.values.iloc[-1]
                        confidence = getattr(indicator_result, 'confidence', 1.0)
                        
                        feature = Feature(
                            name=indicator_name,
                            value=latest_value,
                            feature_type=FeatureType.TECHNICAL,
                            data_type=DataType.NUMERIC,
                            timestamp=timestamp,
                            confidence=confidence,
                            metadata={
                                'indicator_type': indicator_result.type.value if hasattr(indicator_result, 'type') else 'unknown',
                                'signal': getattr(indicator_result, 'signal', None),
                                'full_history': indicator_result.values.tolist() if hasattr(indicator_result.values, 'tolist') else []
                            }
                        )
                        features[feature.name] = feature
            
            # Cache results
            self._set_cache(cache_key, features)
            
            logger.info(f"Generated {len(features)} technical features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating technical features for {symbol}: {e}")
        
        return features
    
    def generate_sentiment_features(self, symbol: str, news_data: List[Dict[str, Any]]) -> Dict[str, Feature]:
        """
        Generate sentiment analysis features
        
        Educational: Sentiment features capture market psychology and
        can be leading indicators of price movements.
        """
        features = {}
        timestamp = datetime.now()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(symbol, timestamp, "sentiment")
            cached_features = self._get_from_cache(cache_key)
            if cached_features:
                return cached_features
            
            if not news_data:
                logger.warning(f"No news data available for sentiment analysis: {symbol}")
                return features
            
            # Analyze sentiment
            sentiment_results = asyncio.run(self.sentiment_analyzer.analyze_batch(news_data))
            
            if sentiment_results:
                # Aggregate sentiment
                market_sentiment = self.sentiment_analyzer.aggregate_market_sentiment(
                    sentiment_results, self.config.sentiment_window
                )
                
                # Create sentiment features
                features['sentiment_overall'] = Feature(
                    name='sentiment_overall',
                    value=market_sentiment.overall_sentiment,
                    feature_type=FeatureType.SENTIMENT,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.8,
                    metadata={'source': 'aggregated', 'window': self.config.sentiment_window.total_seconds()}
                )
                
                features['sentiment_volume_weighted'] = Feature(
                    name='sentiment_volume_weighted',
                    value=market_sentiment.volume_weighted_sentiment,
                    feature_type=FeatureType.SENTIMENT,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.7
                )
                
                # Sentiment distribution
                for label, proportion in market_sentiment.sentiment_distribution.items():
                    features[f'sentiment_{label.name.lower()}'] = Feature(
                        name=f'sentiment_{label.name.lower()}',
                        value=proportion,
                        feature_type=FeatureType.SENTIMENT,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.6
                    )
                
                # Emotional state
                for emotion, score in market_sentiment.emotional_state.items():
                    features[f'emotion_{emotion}'] = Feature(
                        name=f'emotion_{emotion}',
                        value=score,
                        feature_type=FeatureType.SENTIMENT,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.5
                    )
                
                # Source breakdown
                for source, sentiment in market_sentiment.source_breakdown.items():
                    features[f'sentiment_{source.lower()}'] = Feature(
                        name=f'sentiment_{source.lower()}',
                        value=sentiment,
                        feature_type=FeatureType.SENTIMENT,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.6
                    )
                
                # Key topics
                for i, (topic, weight) in enumerate(market_sentiment.key_topics[:5]):
                    features[f'topic_{i+1}_weight'] = Feature(
                        name=f'topic_{i+1}_weight',
                        value=weight,
                        feature_type=FeatureType.SENTIMENT,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.4,
                        metadata={'topic': topic}
                    )
                
                # Sentiment trend
                if len(market_sentiment.sentiment_trend) >= 2:
                    trend_slope = np.polyfit(range(len(market_sentiment.sentiment_trend)), 
                                           market_sentiment.sentiment_trend, 1)[0]
                    features['sentiment_trend'] = Feature(
                        name='sentiment_trend',
                        value=trend_slope,
                        feature_type=FeatureType.SENTIMENT,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.6
                    )
            
            # Cache results
            self._set_cache(cache_key, features)
            
            logger.info(f"Generated {len(features)} sentiment features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating sentiment features for {symbol}: {e}")
        
        return features
    
    def generate_microstructure_features(self, symbol: str, order_book_data: Dict, 
                                      trade_data: List[Dict]) -> Dict[str, Feature]:
        """
        Generate market microstructure features
        
        Educational: Microstructure features capture real-time market dynamics
        and short-term supply/demand imbalances.
        """
        features = {}
        timestamp = datetime.now()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(symbol, timestamp, "microstructure")
            cached_features = self._get_from_cache(cache_key)
            if cached_features:
                return cached_features
            
            if not self.config.microstructure_enabled:
                return features
            
            # Update microstructure analyzer with latest data
            # Note: This would typically be done via real-time data feeds
            # For now, we'll calculate features from the provided data
            
            # Calculate microstructure features
            micro_features = self.microstructure_analyzer.calculate_all_features(symbol)
            
            # Convert to features
            feature_fields = [
                'bid_ask_spread', 'effective_spread', 'realized_spread', 'spread_volatility',
                'bid_depth', 'ask_depth', 'total_depth', 'depth_imbalance', 'liquidity_ratio',
                'order_flow_imbalance', 'trade_intensity', 'order_intensity', 'cancel_ratio',
                'price_impact', 'temporary_impact', 'permanent_impact',
                'microstructure_volatility', 'price_clustering', 'tick_frequency',
                'market_efficiency', 'information_share', 'adverse_selection'
            ]
            
            for field in feature_fields:
                value = getattr(micro_features, field, 0)
                if value is not None:
                    features[field] = Feature(
                        name=field,
                        value=value,
                        feature_type=FeatureType.MICROSTRUCTURE,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.7
                    )
            
            # Market regime
            regime = self.microstructure_analyzer.get_market_regime(symbol)
            features['market_regime'] = Feature(
                name='market_regime',
                value=regime,
                feature_type=FeatureType.MICROSTRUCTURE,
                data_type=DataType.CATEGORICAL,
                timestamp=timestamp,
                confidence=0.8
            )
            
            # Cache results
            self._set_cache(cache_key, features)
            
            logger.info(f"Generated {len(features)} microstructure features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating microstructure features for {symbol}: {e}")
        
        return features
    
    def generate_derived_features(self, symbol: str, base_features: Dict[str, Feature]) -> Dict[str, Feature]:
        """
        Generate derived features from combinations of base features
        
        Educational: Derived features capture relationships and interactions
        between individual features that may have predictive power.
        """
        features = {}
        timestamp = datetime.now()
        
        try:
            if not self.config.create_derived_features:
                return features
            
            # Feature interactions
            feature_names = list(base_features.keys())
            
            # Technical indicator interactions
            technical_features = {k: v for k, v in base_features.items() 
                                if v.feature_type == FeatureType.TECHNICAL}
            
            # RSI and price interaction
            if 'rsi' in technical_features and 'close' in base_features:
                rsi_value = technical_features['rsi'].value
                close_value = base_features['close'].value
                
                # RSI divergence
                features['rsi_price_divergence'] = Feature(
                    name='rsi_price_divergence',
                    value=rsi_value - 50,  # Distance from neutral
                    feature_type=FeatureType.DERIVED,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.6
                )
            
            # Moving average crossovers
            if 'sma_20' in technical_features and 'sma_50' in technical_features:
                sma_20 = technical_features['sma_20'].value
                sma_50 = technical_features['sma_50'].value
                
                features['sma_crossover_signal'] = Feature(
                    name='sma_crossover_signal',
                    value=1 if sma_20 > sma_50 else -1,
                    feature_type=FeatureType.DERIVED,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.7
                )
                
                features['sma_spread'] = Feature(
                    name='sma_spread',
                    value=(sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0,
                    feature_type=FeatureType.DERIVED,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.7
                )
            
            # Bollinger Band position
            if 'bb_upper' in technical_features and 'bb_lower' in technical_features and 'close' in base_features:
                bb_upper = technical_features['bb_upper'].value
                bb_lower = technical_features['bb_lower'].value
                close_value = base_features['close'].value
                
                bb_width = bb_upper - bb_lower
                if bb_width > 0:
                    bb_position = (close_value - bb_lower) / bb_width
                    features['bb_position'] = Feature(
                        name='bb_position',
                        value=bb_position,
                        feature_type=FeatureType.DERIVED,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.7
                    )
            
            # Volume-price interactions
            if 'volume' in base_features and 'atr' in technical_features:
                volume_value = base_features['volume'].value
                atr_value = technical_features['atr'].value
                
                if atr_value > 0:
                    features['volume_atr_ratio'] = Feature(
                        name='volume_atr_ratio',
                        value=volume_value / atr_value,
                        feature_type=FeatureType.DERIVED,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.6
                    )
            
            # Sentiment-technical interactions
            sentiment_features = {k: v for k, v in base_features.items() 
                                if v.feature_type == FeatureType.SENTIMENT}
            
            if 'sentiment_overall' in sentiment_features and 'rsi' in technical_features:
                sentiment_value = sentiment_features['sentiment_overall'].value
                rsi_value = technical_features['rsi'].value
                
                # Sentiment-RSI alignment
                alignment = sentiment_value * (rsi_value - 50) / 50
                features['sentiment_rsi_alignment'] = Feature(
                    name='sentiment_rsi_alignment',
                    value=alignment,
                    feature_type=FeatureType.DERIVED,
                    data_type=DataType.NUMERIC,
                    timestamp=timestamp,
                    confidence=0.5
                )
            
            # Microstructure-technical interactions
            micro_features = {k: v for k, v in base_features.items() 
                            if v.feature_type == FeatureType.MICROSTRUCTURE}
            
            if 'bid_ask_spread' in micro_features and 'atr' in technical_features:
                spread_value = micro_features['bid_ask_spread'].value
                atr_value = technical_features['atr'].value
                
                if atr_value > 0:
                    features['spread_atr_ratio'] = Feature(
                        name='spread_atr_ratio',
                        value=spread_value / atr_value,
                        feature_type=FeatureType.DERIVED,
                        data_type=DataType.NUMERIC,
                        timestamp=timestamp,
                        confidence=0.6
                    )
            
            # Time-based features
            current_time = datetime.now()
            features['hour_of_day'] = Feature(
                name='hour_of_day',
                value=current_time.hour,
                feature_type=FeatureType.DERIVED,
                data_type=DataType.NUMERIC,
                timestamp=timestamp,
                confidence=1.0
            )
            
            features['day_of_week'] = Feature(
                name='day_of_week',
                value=current_time.weekday(),
                feature_type=FeatureType.DERIVED,
                data_type=DataType.NUMERIC,
                timestamp=timestamp,
                confidence=1.0
            )
            
            logger.info(f"Generated {len(features)} derived features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating derived features for {symbol}: {e}")
        
        return features
    
    def calculate_target_variables(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate target variables for supervised learning
        
        Educational: Target variables are what we want to predict.
        Different targets require different strategies.
        """
        targets = {}
        
        try:
            if len(market_data) < max(self.config.target_horizons) + 1:
                logger.warning(f"Insufficient data for target calculation: {symbol}")
                return targets
            
            close_prices = market_data['close'].values
            
            for horizon in self.config.target_horizons:
                if len(close_prices) > horizon:
                    # Future returns
                    future_return = (close_prices[-1] - close_prices[-horizon-1]) / close_prices[-horizon-1]
                    targets[f'return_{horizon}'] = future_return
                    
                    # Future volatility
                    if len(close_prices) > horizon + 1:
                        period_returns = np.diff(close_prices[-horizon-1:]) / close_prices[-horizon-1:-1]
                        future_volatility = np.std(period_returns) * np.sqrt(252)  # Annualized
                        targets[f'volatility_{horizon}'] = future_volatility
                    
                    # Direction (binary classification)
                    targets[f'direction_{horizon}'] = 1 if future_return > 0 else 0
                    
                    # Magnitude (regression)
                    targets[f'magnitude_{horizon}'] = abs(future_return)
            
            # Risk-adjusted returns
            if 'return_5' in targets and 'volatility_5' in targets:
                if targets['volatility_5'] > 0:
                    targets['risk_adjusted_return_5'] = targets['return_5'] / targets['volatility_5']
            
            logger.info(f"Calculated {len(targets)} target variables for {symbol}")
            
        except Exception as e:
            logger.error(f"Error calculating target variables for {symbol}: {e}")
        
        return targets
    
    def normalize_features(self, features: Dict[str, Feature]) -> Dict[str, Feature]:
        """
        Normalize features to improve model performance
        
        Educational: Normalization ensures all features contribute equally
        and helps algorithms converge faster.
        """
        if not self.config.normalize_features:
            return features
        
        normalized_features = {}
        
        try:
            # Calculate statistics for numeric features
            numeric_features = {k: v for k, v in features.items() 
                              if v.data_type == DataType.NUMERIC and isinstance(v.value, (int, float))}
            
            for name, feature in numeric_features.items():
                value = feature.value
                
                # Update running statistics
                if name not in self.feature_statistics:
                    self.feature_statistics[name] = {
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0,
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                
                stats = self.feature_statistics[name]
                stats['count'] += 1
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
                
                # Online mean and standard deviation calculation
                delta = value - stats['mean']
                stats['mean'] += delta / stats['count']
                delta2 = value - stats['mean']
                stats['std'] += delta * delta2
                
                # Normalize if we have enough data
                if stats['count'] > 1:
                    std = np.sqrt(stats['std'] / (stats['count'] - 1))
                    if std > 0:
                        normalized_value = (value - stats['mean']) / std
                    else:
                        normalized_value = 0.0
                else:
                    normalized_value = value
                
                # Create normalized feature
                normalized_feature = Feature(
                    name=f"{name}_normalized",
                    value=normalized_value,
                    feature_type=feature.feature_type,
                    data_type=DataType.NUMERIC,
                    timestamp=feature.timestamp,
                    confidence=feature.confidence * 0.9,  # Slightly lower confidence for normalized
                    metadata={**feature.metadata, 'normalized': True}
                )
                normalized_features[f"{name}_normalized"] = normalized_feature
            
            # Add non-numeric features unchanged
            for name, feature in features.items():
                if feature.data_type != DataType.NUMERIC:
                    normalized_features[name] = feature
            
            logger.info(f"Normalized {len(numeric_features)} features")
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features
        
        return normalized_features
    
    def handle_missing_values(self, features: Dict[str, Feature]) -> Dict[str, Feature]:
        """
        Handle missing values in features
        
        Educational: Missing values are common in real-world data.
        Proper handling prevents model failures and bias.
        """
        if not self.config.handle_missing_values:
            return features
        
        handled_features = {}
        
        try:
            for name, feature in features.items():
                if feature.value is None or (isinstance(feature.value, float) and np.isnan(feature.value)):
                    # Handle missing based on feature type
                    if feature.feature_type == FeatureType.TECHNICAL:
                        # Use 0 for technical indicators (neutral position)
                        filled_value = 0
                        confidence = feature.confidence * 0.5
                    elif feature.feature_type == FeatureType.SENTIMENT:
                        # Use 0 for sentiment (neutral sentiment)
                        filled_value = 0
                        confidence = feature.confidence * 0.5
                    elif feature.feature_type == FeatureType.MICROSTRUCTURE:
                        # Use small positive value for microstructure (minimal activity)
                        filled_value = 0.001
                        confidence = feature.confidence * 0.3
                    else:
                        # Use 0 for derived features
                        filled_value = 0
                        confidence = feature.confidence * 0.5
                    
                    handled_feature = Feature(
                        name=name,
                        value=filled_value,
                        feature_type=feature.feature_type,
                        data_type=feature.data_type,
                        timestamp=feature.timestamp,
                        confidence=confidence,
                        metadata={**feature.metadata, 'missing_value_handled': True}
                    )
                    handled_features[name] = handled_feature
                else:
                    handled_features[name] = feature
            
            logger.info(f"Handled missing values in features")
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return features
        
        return handled_features
    
    def create_feature_set(self, symbol: str, market_data: pd.DataFrame, 
                          news_data: Optional[List[Dict[str, Any]]] = None,
                          order_book_data: Optional[Dict] = None,
                          trade_data: Optional[List[Dict]] = None) -> FeatureSet:
        """
        Create a complete feature set for a symbol
        
        Educational: This is the main method that orchestrates
        all feature generation steps.
        """
        timestamp = datetime.now()
        
        try:
            # Generate base features
            technical_features = self.generate_technical_features(symbol, market_data)
            
            sentiment_features = {}
            if news_data:
                sentiment_features = self.generate_sentiment_features(symbol, news_data)
            
            microstructure_features = {}
            if order_book_data or trade_data:
                microstructure_features = self.generate_microstructure_features(
                    symbol, order_book_data or {}, trade_data or []
                )
            
            # Add basic price features
            price_features = {}
            if not market_data.empty:
                latest_data = market_data.iloc[-1]
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in latest_data:
                        price_features[col] = Feature(
                            name=col,
                            value=latest_data[col],
                            feature_type=FeatureType.TECHNICAL,
                            data_type=DataType.NUMERIC,
                            timestamp=timestamp,
                            confidence=1.0
                        )
            
            # Combine all features
            all_features = {}
            all_features.update(price_features)
            all_features.update(technical_features)
            all_features.update(sentiment_features)
            all_features.update(microstructure_features)
            
            # Generate derived features
            derived_features = self.generate_derived_features(symbol, all_features)
            all_features.update(derived_features)
            
            # Handle missing values
            all_features = self.handle_missing_values(all_features)
            
            # Normalize features
            all_features = self.normalize_features(all_features)
            
            # Calculate target variables
            target_variables = self.calculate_target_variables(symbol, market_data)
            
            # Create feature set
            feature_set = FeatureSet(
                symbol=symbol,
                timestamp=timestamp,
                features=all_features,
                target_variables=target_variables
            )
            
            # Store in history
            self.feature_history[symbol].append(feature_set)
            
            # Keep only recent history
            if len(self.feature_history[symbol]) > 1000:
                self.feature_history[symbol] = self.feature_history[symbol][-1000:]
            
            logger.info(f"Created feature set for {symbol} with {len(all_features)} features")
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error creating feature set for {symbol}: {e}")
            return FeatureSet(symbol=symbol, timestamp=timestamp)
    
    def get_feature_names(self, symbol: Optional[str] = None) -> List[str]:
        """
        Get all available feature names
        """
        if symbol:
            if symbol in self.feature_history and self.feature_history[symbol]:
                return list(self.feature_history[symbol][-1].features.keys())
            else:
                return []
        else:
            # Get all unique feature names across all symbols
            all_features = set()
            for symbol_features in self.feature_history.values():
                if symbol_features:
                    all_features.update(symbol_features[-1].features.keys())
            return list(all_features)
    
    def get_feature_importance(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get feature importance scores
        """
        if symbol:
            return {k: v for k, v in self.feature_importance.items() if k.startswith(f"{symbol}_")}
        else:
            return self.feature_importance
    
    def export_features(self, symbol: str, format: str = 'csv') -> Union[pd.DataFrame, str]:
        """
        Export features for external use
        
        Educational: Exporting features allows for analysis in other tools
        and for model training in different environments.
        """
        try:
            if symbol not in self.feature_history or not self.feature_history[symbol]:
                logger.warning(f"No feature history available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            feature_dicts = []
            for feature_set in self.feature_history[symbol]:
                row = {
                    'timestamp': feature_set.timestamp,
                    'symbol': feature_set.symbol
                }
                
                # Add features
                for name, feature in feature_set.features.items():
                    if isinstance(feature.value, (int, float)):
                        row[name] = feature.value
                
                # Add targets
                for target_name, target_value in feature_set.target_variables.items():
                    row[f'target_{target_name}'] = target_value
                
                feature_dicts.append(row)
            
            df = pd.DataFrame(feature_dicts)
            
            if format == 'csv':
                return df.to_csv(index=False)
            elif format == 'json':
                return df.to_json(orient='records', date_format='iso')
            else:
                return df
                
        except Exception as e:
            logger.error(f"Error exporting features for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature statistics for monitoring and debugging
        """
        return self.feature_statistics

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Basic Feature Generation:
   pipeline = FeaturePipeline()
   feature_set = pipeline.create_feature_set(
       symbol="AAPL",
       market_data=price_data,
       news_data=news_articles
   )
   print(f"Generated {len(feature_set.features)} features")

2. Custom Configuration:
   config = FeatureConfig(
       technical_indicators=['rsi', 'macd', 'bollinger_bands'],
       sentiment_window=timedelta(hours=12),
       microstructure_enabled=True
   )
   pipeline = FeaturePipeline(config)

3. Real-time Feature Updates:
   while True:
       # Get latest market data
       market_data = get_latest_market_data("AAPL")
       news_data = get_latest_news("AAPL")
       
       # Generate features
       features = pipeline.create_feature_set("AAPL", market_data, news_data)
       
       # Use features for trading decisions
       if features.features['sentiment_overall'].value > 0.5:
           print("Bullish sentiment detected")
       
       time.sleep(60)  # Update every minute

4. Feature Export for Model Training:
   # Export features
   df = pipeline.export_features("AAPL", format='dataframe')
   
   # Split features and targets
   feature_cols = [col for col in df.columns if not col.startswith('target_')]
   target_cols = [col for col in df.columns if col.startswith('target_')]
   
   X = df[feature_cols]
   y = df[target_cols]
   
   # Train model
   model = train_model(X, y)

5. Feature Analysis:
   # Get feature statistics
   stats = pipeline.get_feature_statistics()
   
   # Get feature importance
   importance = pipeline.get_feature_importance("AAPL")
   
   # Analyze feature distributions
   for feature_name, stats in stats.items():
       print(f"{feature_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

Key Insights:
- Feature engineering is iterative and domain-specific
- Combining multiple feature types improves robustness
- Normalization and missing value handling are crucial
- Feature monitoring helps detect data quality issues
- Export features for external analysis and model training
- Feature importance guides feature selection and optimization
"""