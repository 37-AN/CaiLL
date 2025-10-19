"""
Features Module - Phase 2.3

This module provides comprehensive feature engineering capabilities for the AI Trading System.
It combines technical indicators, sentiment analysis, and market microstructure features
to create a rich feature set for machine learning models.

Components:
- TechnicalIndicators: 50+ technical indicators for price analysis
- FinancialSentimentAnalyzer: Sentiment analysis for news and social media
- MarketMicrostructureAnalyzer: Order book and trade flow analysis
- FeaturePipeline: Unified feature engineering pipeline

Educational Notes:
Feature engineering is the most critical part of any machine learning system.
Good features can make simple models perform well, while poor features
can make even the most sophisticated models fail.

Key Principles:
1. Domain expertise drives feature creation
2. Multiple feature types provide complementary signals
3. Feature quality is more important than quantity
4. Proper normalization and handling of missing values is essential
5. Feature monitoring ensures data quality over time
"""

from .technical_indicators import (
    TechnicalIndicators,
    IndicatorResult,
    IndicatorType,
    SentimentLabel
)

from .sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    SentimentResult,
    MarketSentiment,
    SentimentType
)

from .market_microstructure import (
    MarketMicrostructureAnalyzer,
    MicrostructureFeatures,
    MarketDepth,
    Order,
    Trade,
    OrderType,
    OrderSide
)

from .feature_pipeline import (
    FeaturePipeline,
    FeatureSet,
    Feature,
    FeatureConfig,
    FeatureType,
    DataType
)

__all__ = [
    # Technical Indicators
    'TechnicalIndicators',
    'IndicatorResult',
    'IndicatorType',
    'SentimentLabel',
    
    # Sentiment Analysis
    'FinancialSentimentAnalyzer',
    'SentimentResult',
    'MarketSentiment',
    'SentimentType',
    
    # Market Microstructure
    'MarketMicrostructureAnalyzer',
    'MicrostructureFeatures',
    'MarketDepth',
    'Order',
    'Trade',
    'OrderType',
    'OrderSide',
    
    # Feature Pipeline
    'FeaturePipeline',
    'FeatureSet',
    'Feature',
    'FeatureConfig',
    'FeatureType',
    'DataType'
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Trading System"
__description__ = "Comprehensive feature engineering for AI trading"

# Educational: Quick Start Guide
QUICK_START_GUIDE = """
Quick Start Guide for Feature Engineering

1. Basic Usage:
   from features import FeaturePipeline
   
   pipeline = FeaturePipeline()
   features = pipeline.create_feature_set(
       symbol="AAPL",
       market_data=price_dataframe,
       news_data=news_list
   )

2. Technical Indicators Only:
   from features import TechnicalIndicators
   
   tech = TechnicalIndicators()
   rsi = tech.rsi(price_data, 14)
   sma = tech.sma(price_data, 20)

3. Sentiment Analysis:
   from features import FinancialSentimentAnalyzer
   
   analyzer = FinancialSentimentAnalyzer()
   sentiment = analyzer.analyze_single_text(
       "Stock rallies on positive earnings",
       source="news"
   )

4. Market Microstructure:
   from features import MarketMicrostructureAnalyzer
   
   analyzer = MarketMicrostructureAnalyzer()
   analyzer.update_order_book(market_depth)
   features = analyzer.calculate_all_features("AAPL")

5. Custom Configuration:
   config = FeatureConfig(
       technical_indicators=['rsi', 'macd'],
       sentiment_window=timedelta(hours=6),
       microstructure_enabled=True
   )
   pipeline = FeaturePipeline(config)

Key Features:
- 50+ technical indicators
- Multi-source sentiment analysis
- Real-time microstructure analysis
- Automated feature normalization
- Missing value handling
- Feature export capabilities
- Performance monitoring

Educational Notes:
- Start with a few key indicators and expand gradually
- Always validate feature quality before using in models
- Monitor feature statistics over time for data quality
- Combine multiple feature types for robust signals
- Use domain knowledge to guide feature selection
"""

# Module-level logging
import logging
logger = logging.getLogger(__name__)
logger.info("Features module initialized - Phase 2.3 Complete")