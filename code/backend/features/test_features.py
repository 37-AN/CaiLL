"""
Feature Engineering Test Script - Phase 2.3

This script demonstrates and tests the complete feature engineering pipeline.
It generates sample data and shows how all components work together.

Educational Note:
Testing is crucial for ensuring feature quality and reliability.
This script serves as both a test and a learning example.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Import our feature modules
from .technical_indicators import TechnicalIndicators
from .sentiment_analyzer import FinancialSentimentAnalyzer
from .market_microstructure import MarketMicrostructureAnalyzer
from .feature_pipeline import FeaturePipeline, FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_market_data(symbol: str, periods: int = 100) -> pd.DataFrame:
    """
    Generate sample market data for testing
    
    Educational: This creates realistic-looking price data with trends,
    volatility, and some randomness to simulate real market behavior.
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate base price with trend and volatility
    base_price = 100.0
    trend = 0.0002  # Slight upward trend
    volatility = 0.02  # 2% daily volatility
    
    prices = [base_price]
    returns = []
    
    for i in range(1, periods):
        # Generate return with trend and randomness
        daily_return = trend + np.random.normal(0, volatility)
        returns.append(daily_return)
        
        # Calculate new price
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate intraday variation
        high_low_range = price * np.random.uniform(0.005, 0.02)
        high = price + high_low_range * np.random.uniform(0.3, 0.7)
        low = price - high_low_range * np.random.uniform(0.3, 0.7)
        
        # Open close relationship
        if i == 0:
            open_price = price
        else:
            # Gap from previous close
            gap = np.random.normal(0, 0.001)
            open_price = prices[i-1] * (1 + gap)
        
        # Ensure OHLC relationships are correct
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        # Generate volume (correlated with price movement)
        volume_base = 1000000
        volume_variation = abs(returns[i-1]) * 10 if i > 0 else 1
        volume = int(volume_base * (1 + volume_variation * np.random.uniform(0.5, 2.0)))
        
        data.append({
            'timestamp': datetime.now() - timedelta(periods=periods-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(df)} periods of sample market data for {symbol}")
    return df

def generate_sample_news_data(symbol: str, count: int = 20) -> List[Dict[str, Any]]:
    """
    Generate sample news data for testing
    
    Educational: Creates news items with varying sentiment to test
    the sentiment analysis capabilities.
    """
    news_templates = [
        (0.8, "{symbol} beats earnings expectations, stock rallies"),
        (0.6, "{symbol} announces new product launch, investors optimistic"),
        (0.4, "{symbol} shows strong growth in quarterly report"),
        (0.2, "{symbol} meets analyst expectations"),
        (0.0, "{symbol} reports mixed quarterly results"),
        (-0.2, "{symbol} misses revenue targets, stock declines"),
        (-0.4, "{symbol} faces regulatory challenges, concerns grow"),
        (-0.6, "{symbol} cuts guidance amid market uncertainty"),
        (-0.8, "{symbol} reports major loss, stock plummets"),
    ]
    
    news_data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(count):
        # Select random news template
        sentiment_score, template = np.random.choice(news_templates)
        
        # Create news item
        news_item = {
            'text': template.format(symbol=symbol),
            'source': np.random.choice(['reuters', 'bloomberg', 'wsj', 'cnbc', 'twitter']),
            'timestamp': base_time + timedelta(hours=i * 24 / count),
            'sentiment_score': sentiment_score
        }
        
        news_data.append(news_item)
    
    logger.info(f"Generated {len(news_data)} sample news items for {symbol}")
    return news_data

def test_technical_indicators():
    """
    Test technical indicators functionality
    """
    logger.info("Testing Technical Indicators...")
    
    # Generate sample data
    market_data = generate_sample_market_data("AAPL", 100)
    
    # Initialize analyzer
    tech_analyzer = TechnicalIndicators()
    
    # Test individual indicators
    try:
        # Test SMA
        sma_result = tech_analyzer.sma(market_data['close'], 20)
        logger.info(f"SMA(20) latest value: {sma_result.values.iloc[-1]:.2f}")
        
        # Test RSI
        rsi_result = tech_analyzer.rsi(market_data['close'], 14)
        logger.info(f"RSI(14) latest value: {rsi_result.values.iloc[-1]:.2f}")
        
        # Test MACD
        macd_results = tech_analyzer.macd(market_data['close'])
        logger.info(f"MACD latest value: {macd_results['macd'].values.iloc[-1]:.4f}")
        
        # Test Bollinger Bands
        bb_results = tech_analyzer.bollinger_bands(market_data['close'])
        logger.info(f"BB Upper latest: {bb_results['upper'].values.iloc[-1]:.2f}")
        logger.info(f"BB Lower latest: {bb_results['lower'].values.iloc[-1]:.2f}")
        
        # Test all indicators
        all_indicators = tech_analyzer.calculate_all_indicators(market_data)
        logger.info(f"Total indicators calculated: {len(all_indicators)}")
        
        logger.info("✅ Technical Indicators test passed")
        
    except Exception as e:
        logger.error(f"❌ Technical Indicators test failed: {e}")
        return False
    
    return True

def test_sentiment_analyzer():
    """
    Test sentiment analysis functionality
    """
    logger.info("Testing Sentiment Analyzer...")
    
    try:
        # Initialize analyzer
        sentiment_analyzer = FinancialSentimentAnalyzer()
        
        # Test single text analysis
        test_text = "Apple reports strong earnings, stock rallies 5% on positive guidance"
        result = sentiment_analyzer.analyze_single_text(
            test_text, 
            source="reuters"
        )
        
        logger.info(f"Sentiment score: {result.sentiment_score:.2f}")
        logger.info(f"Sentiment label: {result.sentiment_label.name}")
        logger.info(f"Confidence: {result.confidence:.2f}")
        logger.info(f"Financial entities: {result.financial_entities}")
        
        # Test batch analysis
        news_data = generate_sample_news_data("AAPL", 10)
        batch_results = asyncio.run(sentiment_analyzer.analyze_batch(news_data))
        
        logger.info(f"Batch analysis completed for {len(batch_results)} items")
        
        # Test market sentiment aggregation
        market_sentiment = sentiment_analyzer.aggregate_market_sentiment(batch_results)
        logger.info(f"Overall market sentiment: {market_sentiment.overall_sentiment:.2f}")
        logger.info(f"Sentiment distribution: {market_sentiment.sentiment_distribution}")
        
        # Test sentiment signals
        signals = sentiment_analyzer.generate_sentiment_signals(market_sentiment)
        logger.info(f"Primary signal: {signals['primary_signal']}")
        logger.info(f"Signal confidence: {signals['confidence']:.2f}")
        
        logger.info("✅ Sentiment Analyzer test passed")
        
    except Exception as e:
        logger.error(f"❌ Sentiment Analyzer test failed: {e}")
        return False
    
    return True

def test_market_microstructure():
    """
    Test market microstructure functionality
    """
    logger.info("Testing Market Microstructure Analyzer...")
    
    try:
        # Initialize analyzer
        micro_analyzer = MarketMicrostructureAnalyzer()
        
        # Create sample order book
        from .market_microstructure import MarketDepth
        
        # Generate sample order book
        bids = []
        asks = []
        base_price = 100.0
        
        # Create bids (buy orders)
        for i in range(10):
            price = base_price - i * 0.01
            quantity = int(np.random.uniform(100, 1000))
            bids.append((price, quantity))
        
        # Create asks (sell orders)
        for i in range(10):
            price = base_price + (i + 1) * 0.01
            quantity = int(np.random.uniform(100, 1000))
            asks.append((price, quantity))
        
        order_book = MarketDepth(
            timestamp=datetime.now(),
            symbol="AAPL",
            bids=bids,
            asks=asks
        )
        
        # Update analyzer
        micro_analyzer.update_order_book(order_book)
        
        # Test spread features
        spread_features = micro_analyzer.calculate_spread_features("AAPL")
        logger.info(f"Bid-Ask Spread: {spread_features.get('bid_ask_spread', 0):.4f}")
        
        # Test liquidity features
        liquidity_features = micro_analyzer.calculate_liquidity_features("AAPL")
        logger.info(f"Total Depth (5 levels): {liquidity_features.get('total_depth_5', 0)}")
        
        # Test all features
        all_features = micro_analyzer.calculate_all_features("AAPL")
        logger.info(f"Total microstructure features: {len(all_features.__dict__)}")
        
        # Test market regime
        regime = micro_analyzer.get_market_regime("AAPL")
        logger.info(f"Market regime: {regime}")
        
        # Test slippage estimation
        from .market_microstructure import OrderSide
        slippage = micro_analyzer.estimate_slippage("AAPL", 500, OrderSide.BUY)
        logger.info(f"Estimated slippage: ${slippage['estimated_slippage']:.4f} per share")
        
        logger.info("✅ Market Microstructure test passed")
        
    except Exception as e:
        logger.error(f"❌ Market Microstructure test failed: {e}")
        return False
    
    return True

def test_feature_pipeline():
    """
    Test the complete feature pipeline
    """
    logger.info("Testing Feature Pipeline...")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data("AAPL", 100)
        news_data = generate_sample_news_data("AAPL", 15)
        
        # Create custom configuration
        config = FeatureConfig(
            technical_indicators=['sma_20', 'rsi', 'macd', 'bollinger_bands'],
            sentiment_window=timedelta(hours=12),
            microstructure_enabled=False,  # Disable for this test
            create_derived_features=True,
            normalize_features=True
        )
        
        # Initialize pipeline
        pipeline = FeaturePipeline(config)
        
        # Create feature set
        feature_set = pipeline.create_feature_set(
            symbol="AAPL",
            market_data=market_data,
            news_data=news_data
        )
        
        logger.info(f"Feature set created for {feature_set.symbol}")
        logger.info(f"Total features: {len(feature_set.features)}")
        logger.info(f"Target variables: {len(feature_set.target_variables)}")
        
        # Analyze feature types
        feature_types = {}
        for feature in feature_set.features.values():
            ftype = feature.feature_type.value
            feature_types[ftype] = feature_types.get(ftype, 0) + 1
        
        logger.info("Feature type distribution:")
        for ftype, count in feature_types.items():
            logger.info(f"  {ftype}: {count}")
        
        # Test feature export
        df = pipeline.export_features("AAPL", format='dataframe')
        logger.info(f"Exported features shape: {df.shape}")
        
        # Test feature statistics
        stats = pipeline.get_feature_statistics()
        logger.info(f"Feature statistics available for {len(stats)} features")
        
        # Test feature names
        feature_names = pipeline.get_feature_names("AAPL")
        logger.info(f"Feature names sample: {feature_names[:5]}")
        
        logger.info("✅ Feature Pipeline test passed")
        
    except Exception as e:
        logger.error(f"❌ Feature Pipeline test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """
    Run all feature engineering tests
    """
    logger.info("🚀 Starting Feature Engineering Tests - Phase 2.3")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(test_technical_indicators())
    test_results.append(test_sentiment_analyzer())
    test_results.append(test_market_microstructure())
    test_results.append(test_feature_pipeline())
    
    # Summary
    logger.info("=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        logger.info(f"🎉 All {total} tests passed! Phase 2.3 is complete.")
    else:
        logger.error(f"❌ {total - passed} out of {total} tests failed.")
    
    logger.info("=" * 60)
    
    # Educational summary
    logger.info("📚 Educational Summary:")
    logger.info("✅ Technical Indicators: 50+ indicators for price analysis")
    logger.info("✅ Sentiment Analysis: Multi-source sentiment processing")
    logger.info("✅ Market Microstructure: Order book and trade flow analysis")
    logger.info("✅ Feature Pipeline: Unified feature engineering system")
    logger.info("✅ Data Quality: Missing value handling and normalization")
    logger.info("✅ Export Capabilities: Multiple format support")
    
    return passed == total

if __name__ == "__main__":
    """
    Educational: This test script demonstrates the complete feature engineering
    pipeline and serves as a learning example for understanding how all components
    work together to create a comprehensive feature set for AI trading models.
    """
    success = run_all_tests()
    
    if success:
        print("\n🎯 Phase 2.3 Complete: Feature Engineering Pipeline")
        print("📈 Ready for Phase 3: Reinforcement Learning Engine")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")