"""
Technical Indicators Module - Phase 2.3

This module implements 50+ technical indicators used in quantitative trading.
Each indicator is optimized for performance and includes proper validation.

Educational Note:
Technical indicators are mathematical calculations based on historical price data.
They help traders identify trends, momentum, volatility, and potential reversal points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OVERLAP = "overlap"
    OSCILLATOR = "oscillator"

@dataclass
class IndicatorResult:
    """Result of technical indicator calculation"""
    name: str
    values: Union[float, np.ndarray, pd.Series]
    type: IndicatorType
    signal: Optional[str] = None
    confidence: Optional[float] = None

class TechnicalIndicators:
    """
    Comprehensive Technical Indicators Calculator
    
    Educational Notes:
    - Trend indicators: Help identify direction of price movement
    - Momentum indicators: Measure speed and strength of price changes
    - Volatility indicators: Measure price fluctuation magnitude
    - Volume indicators: Analyze trading volume patterns
    """
    
    def __init__(self):
        self.indicators = {}
        logger.info("TechnicalIndicators initialized")
    
    # ==================== TREND INDICATORS ====================
    
    def sma(self, data: pd.Series, period: int = 20) -> IndicatorResult:
        """
        Simple Moving Average
        
        Educational: The most basic trend indicator that smooths price data
        by creating a constantly updated average price.
        
        Formula: SMA = Sum(Close, N) / N
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for SMA({period})")
        
        values = data.rolling(window=period).mean()
        return IndicatorResult(
            name=f"SMA_{period}",
            values=values,
            type=IndicatorType.TREND
        )
    
    def ema(self, data: pd.Series, period: int = 20) -> IndicatorResult:
        """
        Exponential Moving Average
        
        Educational: Gives more weight to recent prices, making it more
        responsive to new information than SMA.
        
        Formula: EMA = (Close × Multiplier) + (EMA_previous × (1 - Multiplier))
        Multiplier = 2 / (period + 1)
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for EMA({period})")
        
        values = data.ewm(span=period, adjust=False).mean()
        return IndicatorResult(
            name=f"EMA_{period}",
            values=values,
            type=IndicatorType.TREND
        )
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, IndicatorResult]:
        """
        Moving Average Convergence Divergence
        
        Educational: Trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.
        
        Components:
        - MACD Line: EMA(fast) - EMA(slow)
        - Signal Line: EMA(MACD, signal_period)
        - Histogram: MACD - Signal
        """
        if len(data) < slow:
            raise ValueError(f"Insufficient data for MACD({fast},{slow},{signal})")
        
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': IndicatorResult(f"MACD_{fast}_{slow}", macd_line, IndicatorType.MOMENTUM),
            'signal': IndicatorResult(f"MACD_Signal_{signal}", signal_line, IndicatorType.MOMENTUM),
            'histogram': IndicatorResult(f"MACD_Histogram", histogram, IndicatorType.MOMENTUM)
        }
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Average Directional Index
        
        Educational: Measures trend strength without indicating trend direction.
        Values above 25 indicate a strong trend, below 20 indicate weak trend.
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for ADX({period})")
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate ADX
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return IndicatorResult(
            name=f"ADX_{period}",
            values=adx,
            type=IndicatorType.TREND
        )
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def rsi(self, data: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Relative Strength Index
        
        Educational: Momentum oscillator that measures the speed and change of price movements.
        Ranges from 0 to 100. Overbought > 70, Oversold < 30.
        
        Formula: RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for RSI({period})")
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signal = np.where(rsi > 70, "overbought", 
                         np.where(rsi < 30, "oversold", "neutral"))
        
        return IndicatorResult(
            name=f"RSI_{period}",
            values=rsi,
            type=IndicatorType.MOMENTUM,
            signal=signal[-1] if len(signal) > 0 else None
        )
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, IndicatorResult]:
        """
        Stochastic Oscillator
        
        Educational: Momentum indicator comparing a particular closing price
        of a security to a range of its prices over a certain period of time.
        
        Formula: %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        """
        if len(high) < k_period:
            raise ValueError(f"Insufficient data for Stochastic({k_period},{d_period})")
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': IndicatorResult(f"Stoch_K_{k_period}", k_percent, IndicatorType.OSCILLATOR),
            'd': IndicatorResult(f"Stoch_D_{d_period}", d_percent, IndicatorType.OSCILLATOR)
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Williams %R
        
        Educational: Momentum indicator that measures overbought and oversold levels.
        Ranges from -100 to 0. Readings above -20 are overbought, below -80 are oversold.
        
        Formula: %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Williams_R({period})")
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return IndicatorResult(
            name=f"Williams_R_{period}",
            values=wr,
            type=IndicatorType.OSCILLATOR
        )
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> IndicatorResult:
        """
        Commodity Channel Index
        
        Educational: Unbounded oscillator that helps identify cyclical trends
        and overbought/oversold conditions.
        
        Formula: CCI = (Typical Price - SMA(TP, period)) / (0.015 * Mean Deviation)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for CCI({period})")
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return IndicatorResult(
            name=f"CCI_{period}",
            values=cci,
            type=IndicatorType.OSCILLATOR
        )
    
    def roc(self, data: pd.Series, period: int = 12) -> IndicatorResult:
        """
        Rate of Change
        
        Educational: Momentum indicator that measures the percentage change
        in price between the current price and the price from a certain number of periods ago.
        
        Formula: ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for ROC({period})")
        
        roc = ((data - data.shift(period)) / data.shift(period)) * 100
        
        return IndicatorResult(
            name=f"ROC_{period}",
            values=roc,
            type=IndicatorType.MOMENTUM
        )
    
    def momentum(self, data: pd.Series, period: int = 10) -> IndicatorResult:
        """
        Momentum Indicator
        
        Educational: Simple momentum indicator that measures the rate of change
        in price movements.
        
        Formula: Momentum = Current Price - Price N periods ago
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for Momentum({period})")
        
        momentum = data - data.shift(period)
        
        return IndicatorResult(
            name=f"Momentum_{period}",
            values=momentum,
            type=IndicatorType.MOMENTUM
        )
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, IndicatorResult]:
        """
        Bollinger Bands
        
        Educational: Volatility indicator that consists of a middle band (SMA)
        and two outer bands set at standard deviation levels above and below the middle band.
        
        Components:
        - Middle Band: SMA(period)
        - Upper Band: SMA + (std_dev × Standard Deviation)
        - Lower Band: SMA - (std_dev × Standard Deviation)
        """
        if len(data) < period:
            raise ValueError(f"Insufficient data for Bollinger_Bands({period},{std_dev})")
        
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # Calculate bandwidth and %B
        bandwidth = (upper - lower) / middle * 100
        percent_b = (data - lower) / (upper - lower)
        
        return {
            'middle': IndicatorResult(f"BB_Middle_{period}", middle, IndicatorType.VOLATILITY),
            'upper': IndicatorResult(f"BB_Upper_{period}", upper, IndicatorType.VOLATILITY),
            'lower': IndicatorResult(f"BB_Lower_{period}", lower, IndicatorType.VOLATILITY),
            'bandwidth': IndicatorResult(f"BB_Bandwidth_{period}", bandwidth, IndicatorType.VOLATILITY),
            'percent_b': IndicatorResult(f"BB_PercentB_{period}", percent_b, IndicatorType.OSCILLATOR)
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Average True Range
        
        Educational: Volatility measure that captures market volatility.
        It doesn't indicate price direction, only volatility.
        
        True Range = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
        ATR = SMA(True Range, period)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for ATR({period})")
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        return IndicatorResult(
            name=f"ATR_{period}",
            values=atr,
            type=IndicatorType.VOLATILITY
        )
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 20, multiplier: float = 2.0) -> Dict[str, IndicatorResult]:
        """
        Keltner Channels
        
        Educational: Volatility-based bands that use ATR instead of standard deviation.
        Often used to identify trend direction and potential reversals.
        
        Components:
        - Middle Line: EMA(close, period)
        - Upper Channel: EMA + (multiplier × ATR)
        - Lower Channel: EMA - (multiplier × ATR)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Keltner_Channels({period},{multiplier})")
        
        middle = close.ewm(span=period, adjust=False).mean()
        atr = self.atr(high, low, close, period).values
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return {
            'middle': IndicatorResult(f"KC_Middle_{period}", middle, IndicatorType.VOLATILITY),
            'upper': IndicatorResult(f"KC_Upper_{period}", upper, IndicatorType.VOLATILITY),
            'lower': IndicatorResult(f"KC_Lower_{period}", lower, IndicatorType.VOLATILITY)
        }
    
    # ==================== VOLUME INDICATORS ====================
    
    def obv(self, close: pd.Series, volume: pd.Series) -> IndicatorResult:
        """
        On-Balance Volume
        
        Educational: Cumulative volume-based indicator that uses volume flow
        to predict changes in stock price.
        
        Logic: Add volume on up days, subtract on down days
        """
        if len(close) != len(volume):
            raise ValueError("Close and volume must have same length")
        
        obv = np.where(close > close.shift(1), volume,
                      np.where(close < close.shift(1), -volume, 0))
        obv_cumulative = pd.Series(obv).cumsum()
        
        return IndicatorResult(
            name="OBV",
            values=obv_cumulative,
            type=IndicatorType.VOLUME
        )
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
             volume: pd.Series) -> IndicatorResult:
        """
        Volume Weighted Average Price
        
        Educational: Trading benchmark that gives the average price a security
        has traded at throughout the day, based on both volume and price.
        
        Formula: VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        if not all(len(series) == len(volume) for series in [high, low, close]):
            raise ValueError("All price series must have same length as volume")
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return IndicatorResult(
            name="VWAP",
            values=vwap,
            type=IndicatorType.VOLUME
        )
    
    def money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Money Flow Index
        
        Educational: Momentum indicator that incorporates both price and volume.
        Ranges from 0 to 100. Overbought > 80, Oversold < 20.
        
        Formula: MFI = 100 - (100 / (1 + Money Flow Ratio))
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for MFI({period})")
        
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        # Determine positive and negative money flow
        positive_mf = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_mf = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        
        # Calculate money flow ratio
        positive_mf_sum = pd.Series(positive_mf).rolling(window=period).sum()
        negative_mf_sum = pd.Series(negative_mf).rolling(window=period).sum()
        
        money_flow_ratio = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return IndicatorResult(
            name=f"MFI_{period}",
            values=mfi,
            type=IndicatorType.VOLUME
        )
    
    def accumulation_distribution(self, high: pd.Series, low: pd.Series, 
                                 close: pd.Series, volume: pd.Series) -> IndicatorResult:
        """
        Accumulation/Distribution Line
        
        Educational: Volume-based indicator that measures the cumulative flow of money.
        Uses volume and price relationship to determine buying/selling pressure.
        
        Formula: A/D = Previous A/D + ((Close - Low) - (High - Close)) / (High - Low) × Volume
        """
        if not all(len(series) == len(volume) for series in [high, low, close]):
            raise ValueError("All price series must have same length as volume")
        
        clv = ((close - low) - (high - close)) / (high - low)
        ad = (clv * volume).cumsum()
        
        return IndicatorResult(
            name="AD_Line",
            values=ad,
            type=IndicatorType.VOLUME
        )
    
    # ==================== OVERLAP STUDIES ====================
    
    def ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, IndicatorResult]:
        """
        Ichimoku Cloud
        
        Educational: Comprehensive indicator that defines support and resistance,
        identifies trend direction, gauges momentum, and provides trading signals.
        
        Components:
        - Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 (9 periods)
        - Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 (26 periods)
        - Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 (26 periods ahead)
        - Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 (52 periods ahead)
        - Chikou Span (Lagging Span): Close (26 periods behind)
        """
        if len(high) < 52:
            raise ValueError("Insufficient data for Ichimoku Cloud (requires 52 periods)")
        
        # Tenkan-sen (9 periods)
        tenkan_high = high.rolling(window=9).max()
        tenkan_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (26 periods)
        kijun_high = high.rolling(window=26).max()
        kijun_low = low.rolling(window=26).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (26 periods ahead)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (52 periods ahead)
        senkou_high = high.rolling(window=52).max()
        senkou_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(52)
        
        # Chikou Span (26 periods behind)
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': IndicatorResult("Ichimoku_Tenkan", tenkan_sen, IndicatorType.TREND),
            'kijun_sen': IndicatorResult("Ichimoku_Kijun", kijun_sen, IndicatorType.TREND),
            'senkou_span_a': IndicatorResult("Ichimoku_SpanA", senkou_span_a, IndicatorType.TREND),
            'senkou_span_b': IndicatorResult("Ichimoku_SpanB", senkou_span_b, IndicatorType.TREND),
            'chikou_span': IndicatorResult("Ichimoku_Chikou", chikou_span, IndicatorType.TREND)
        }
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, IndicatorResult]:
        """
        Pivot Points
        
        Educational: Intraday indicator that determines potential support and resistance levels.
        Based on previous day's high, low, and close prices.
        
        Formula:
        - Pivot Point (PP) = (High + Low + Close) / 3
        - Resistance 1 (R1) = (2 × PP) - Low
        - Support 1 (S1) = (2 × PP) - High
        - Resistance 2 (R2) = PP + (High - Low)
        - Support 2 (S2) = PP - (High - Low)
        """
        if len(high) < 1:
            raise ValueError("Insufficient data for Pivot Points")
        
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        
        return {
            'pivot': IndicatorResult("Pivot_PP", pp, IndicatorType.OVERLAP),
            'r1': IndicatorResult("Pivot_R1", r1, IndicatorType.OVERLAP),
            's1': IndicatorResult("Pivot_S1", s1, IndicatorType.OVERLAP),
            'r2': IndicatorResult("Pivot_R2", r2, IndicatorType.OVERLAP),
            's2': IndicatorResult("Pivot_S2", s2, IndicatorType.OVERLAP)
        }
    
    # ==================== ADVANCED INDICATORS ====================
    
    def fibonacci_retracements(self, high: pd.Series, low: pd.Series, period: int = 100) -> Dict[str, IndicatorResult]:
        """
        Fibonacci Retracement Levels
        
        Educational: Horizontal lines that indicate where support and resistance
        are likely to occur. Based on Fibonacci ratios.
        
        Key Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Fibonacci({period})")
        
        rolling_high = high.rolling(window=period).max()
        rolling_low = low.rolling(window=period).min()
        
        diff = rolling_high - rolling_low
        
        fib_236 = rolling_high - (0.236 * diff)
        fib_382 = rolling_high - (0.382 * diff)
        fib_500 = rolling_high - (0.500 * diff)
        fib_618 = rolling_high - (0.618 * diff)
        fib_786 = rolling_high - (0.786 * diff)
        
        return {
            'fib_236': IndicatorResult("Fib_23.6", fib_236, IndicatorType.OVERLAP),
            'fib_382': IndicatorResult("Fib_38.2", fib_382, IndicatorType.OVERLAP),
            'fib_500': IndicatorResult("Fib_50.0", fib_500, IndicatorType.OVERLAP),
            'fib_618': IndicatorResult("Fib_61.8", fib_618, IndicatorType.OVERLAP),
            'fib_786': IndicatorResult("Fib_78.6", fib_786, IndicatorType.OVERLAP)
        }
    
    def elder_ray(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> Dict[str, IndicatorResult]:
        """
        Elder Ray Index
        
        Educational: Oscillator that measures the relative power of bulls and bears.
        Consists of Bull Power and Bear Power indicators.
        
        Components:
        - Bull Power: High - EMA(close)
        - Bear Power: Low - EMA(close)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Elder Ray({period})")
        
        ema = close.ewm(span=period, adjust=False).mean()
        bull_power = high - ema
        bear_power = low - ema
        
        return {
            'bull_power': IndicatorResult(f"Elder_Bull_{period}", bull_power, IndicatorType.OSCILLATOR),
            'bear_power': IndicatorResult(f"Elder_Bear_{period}", bear_power, IndicatorType.OSCILLATOR)
        }
    
    def force_index(self, close: pd.Series, volume: pd.Series, period: int = 13) -> IndicatorResult:
        """
        Force Index
        
        Educational: Oscillator that measures the force behind price movements.
        Combines price direction, extent of price change, and trading volume.
        
        Formula: Force Index = (Current Close - Previous Close) × Volume
        EMA Force Index = EMA(Force Index, period)
        """
        if len(close) < period:
            raise ValueError(f"Insufficient data for Force Index({period})")
        
        force = (close.diff()) * volume
        force_ema = force.ewm(span=period, adjust=False).mean()
        
        return IndicatorResult(
            name=f"Force_Index_{period}",
            values=force_ema,
            type=IndicatorType.MOMENTUM
        )
    
    def ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Ease of Movement
        
        Educational: Volume-based oscillator that measures the relationship
        between price change and volume. High values indicate prices are moving
        upward with minimal volume resistance.
        
        Formula: EoM = ((High + Low)/2 - (Previous High + Previous Low)/2) / Volume
        SMA EoM = SMA(EoM, period)
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Ease of Movement({period})")
        
        distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_height = high - low
        
        eom = (distance_moved / box_height) / volume
        eom_sma = eom.rolling(window=period).mean()
        
        return IndicatorResult(
            name=f"EoM_{period}",
            values=eom_sma,
            type=IndicatorType.VOLUME
        )
    
    def mass_index(self, high: pd.Series, low: pd.Series, period: int = 25, ema_period: int = 9) -> IndicatorResult:
        """
        Mass Index
        
        Educational: Developed to identify trend reversals based on the concept
        that reversals are more likely when the range widens.
        
        Formula: MI = Σ(EMA(High-Low, 9) / EMA(EMA(High-Low, 9), 9)) over period
        """
        if len(high) < period:
            raise ValueError(f"Insufficient data for Mass Index({period},{ema_period})")
        
        high_low = high - low
        ema1 = high_low.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        
        ratio = ema1 / ema2
        mass_index = ratio.rolling(window=period).sum()
        
        return IndicatorResult(
            name=f"Mass_Index_{period}",
            values=mass_index,
            type=IndicatorType.VOLATILITY
        )
    
    def negative_volume_index(self, close: pd.Series, volume: pd.Series) -> IndicatorResult:
        """
        Negative Volume Index
        
        Educational: Focuses on days when volume decreases from the previous day.
        Based on the idea that smart money trades on quiet days.
        
        Logic: Only calculate when volume decreases
        """
        if len(close) != len(volume):
            raise ValueError("Close and volume must have same length")
        
        nvi = np.where(volume < volume.shift(1), 
                      close.pct_change() + 1, 1)
        nvi_cumulative = pd.Series(nvi).cumprod()
        
        return IndicatorResult(
            name="NVI",
            values=nvi_cumulative,
            type=IndicatorType.VOLUME
        )
    
    def positive_volume_index(self, close: pd.Series, volume: pd.Series) -> IndicatorResult:
        """
        Positive Volume Index
        
        Educational: Focuses on days when volume increases from the previous day.
        Based on the idea that uninformed crowd trades on active days.
        
        Logic: Only calculate when volume increases
        """
        if len(close) != len(volume):
            raise ValueError("Close and volume must have same length")
        
        pvi = np.where(volume > volume.shift(1), 
                      close.pct_change() + 1, 1)
        pvi_cumulative = pd.Series(pvi).cumprod()
        
        return IndicatorResult(
            name="PVI",
            values=pvi_cumulative,
            type=IndicatorType.VOLUME
        )
    
    def ultimate_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period1: int = 7, period2: int = 14, period3: int = 28) -> IndicatorResult:
        """
        Ultimate Oscillator
        
        Educational: Momentum oscillator designed to capture momentum across
        three different timeframes, reducing volatility and false signals.
        
        Formula: UO = 100 × [(4 × Avg7) + (2 × Avg14) + Avg28] / (4 + 2 + 1)
        """
        if len(high) < period3:
            raise ValueError(f"Insufficient data for Ultimate Oscillator({period1},{period2},{period3})")
        
        # Calculate Buying Pressure (BP) and True Range (TR)
        bp = close - np.minimum(low, close.shift(1))
        tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
        
        # Calculate averages for each period
        avg1 = (bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum())
        avg2 = (bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum())
        avg3 = (bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum())
        
        # Calculate Ultimate Oscillator
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        
        return IndicatorResult(
            name=f"UO_{period1}_{period2}_{period3}",
            values=uo,
            type=IndicatorType.OSCILLATOR
        )
    
    # ==================== UTILITY METHODS ====================
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Union[IndicatorResult, Dict[str, IndicatorResult]]]:
        """
        Calculate all available indicators for the given data
        
        Educational: This method provides a comprehensive analysis by calculating
        all indicators. In practice, you'd select indicators based on your strategy.
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Data must contain: open, high, low, close, volume columns")
        
        indicators = {}
        
        try:
            # Trend Indicators
            indicators['sma_20'] = self.sma(data['close'], 20)
            indicators['sma_50'] = self.sma(data['close'], 50)
            indicators['ema_20'] = self.ema(data['close'], 20)
            indicators['ema_50'] = self.ema(data['close'], 50)
            indicators.update(self.macd(data['close']))
            indicators['adx'] = self.adx(data['high'], data['low'], data['close'])
            
            # Momentum Indicators
            indicators['rsi'] = self.rsi(data['close'])
            indicators.update(self.stochastic(data['high'], data['low'], data['close']))
            indicators['williams_r'] = self.williams_r(data['high'], data['low'], data['close'])
            indicators['cci'] = self.cci(data['high'], data['low'], data['close'])
            indicators['roc'] = self.roc(data['close'])
            indicators['momentum'] = self.momentum(data['close'])
            
            # Volatility Indicators
            indicators.update(self.bollinger_bands(data['close']))
            indicators['atr'] = self.atr(data['high'], data['low'], data['close'])
            indicators.update(self.keltner_channels(data['high'], data['low'], data['close']))
            
            # Volume Indicators
            indicators['obv'] = self.obv(data['close'], data['volume'])
            indicators['vwap'] = self.vwap(data['high'], data['low'], data['close'], data['volume'])
            indicators['mfi'] = self.money_flow_index(data['high'], data['low'], data['close'], data['volume'])
            indicators['ad_line'] = self.accumulation_distribution(data['high'], data['low'], data['close'], data['volume'])
            
            # Overlap Studies
            indicators.update(self.ichimoku_cloud(data['high'], data['low'], data['close']))
            indicators.update(self.pivot_points(data['high'], data['low'], data['close']))
            
            # Advanced Indicators
            indicators.update(self.fibonacci_retracements(data['high'], data['low']))
            indicators.update(self.elder_ray(data['high'], data['low'], data['close']))
            indicators['force_index'] = self.force_index(data['close'], data['volume'])
            indicators['eom'] = self.ease_of_movement(data['high'], data['low'], data['volume'])
            indicators['mass_index'] = self.mass_index(data['high'], data['low'])
            indicators['nvi'] = self.negative_volume_index(data['close'], data['volume'])
            indicators['pvi'] = self.positive_volume_index(data['close'], data['volume'])
            indicators['ultimate_oscillator'] = self.ultimate_oscillator(data['high'], data['low'], data['close'])
            
            logger.info(f"Calculated {len(indicators)} indicator groups")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
        
        return indicators
    
    def get_indicator_summary(self, indicators: Dict[str, Union[IndicatorResult, Dict[str, IndicatorResult]]]) -> Dict[str, Dict]:
        """
        Get summary statistics for all calculated indicators
        """
        summary = {}
        
        for name, indicator in indicators.items():
            if isinstance(indicator, dict):
                # Handle grouped indicators (like MACD, Bollinger Bands)
                for sub_name, sub_indicator in indicator.items():
                    if hasattr(sub_indicator.values, 'describe'):
                        summary[f"{name}_{sub_name}"] = sub_indicator.values.describe().to_dict()
            else:
                # Handle single indicators
                if hasattr(indicator.values, 'describe'):
                    summary[name] = indicator.values.describe().to_dict()
        
        return summary

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Basic Trend Analysis:
   indicators = TechnicalIndicators()
   sma_20 = indicators.sma(price_data, 20)
   ema_20 = indicators.ema(price_data, 20)
   
   # When EMA crosses above SMA, it's a bullish signal
   # When EMA crosses below SMA, it's a bearish signal

2. Momentum Trading:
   rsi = indicators.rsi(price_data, 14)
   macd = indicators.macd(price_data)
   
   # RSI > 70: Overbought (potential sell signal)
   # RSI < 30: Oversold (potential buy signal)
   # MACD line crossing above signal line: Bullish
   # MACD line crossing below signal line: Bearish

3. Volatility Trading:
   bb = indicators.bollinger_bands(price_data)
   atr = indicators.atr(high_data, low_data, close_data)
   
   # Price touching upper band: Potential reversal down
   # Price touching lower band: Potential reversal up
   # High ATR: High volatility (wider stops)
   # Low ATR: Low volatility (tighter stops)

4. Volume Analysis:
   obv = indicators.obv(close_data, volume_data)
   vwap = indicators.vwap(high_data, low_data, close_data, volume_data)
   
   # OBV rising with price: Confirms uptrend
   # OBV falling with price: Confirms downtrend
   # Price above VWAP: Bullish
   # Price below VWAP: Bearish

5. Comprehensive Analysis:
   all_indicators = indicators.calculate_all_indicators(market_data)
   
   # Use multiple indicators for confirmation
   # Example: Buy signal when:
   # - RSI < 30 (oversold)
   # - Price touches lower Bollinger Band
   # - MACD shows bullish crossover
   # - Volume confirms the move
"""