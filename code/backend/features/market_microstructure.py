"""
Market Microstructure Features Module - Phase 2.3

This module implements market microstructure analysis features that capture
the dynamics of order flow, liquidity, and market mechanics.

Educational Note:
Market microstructure studies how trades occur in financial markets.
Understanding order flow, spread dynamics, and liquidity patterns is crucial
for high-frequency trading and market timing strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    ALGORITHM = "algorithm"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"
    BUY_COVER = "buy_cover"
    SELL_SHORT = "sell_short"

@dataclass
class Order:
    """Order representation"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    is_hidden: bool = False
    is_iceberg: bool = False
    exchange: str = "unknown"
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity

@dataclass
class Trade:
    """Trade representation"""
    trade_id: str
    timestamp: datetime
    symbol: str
    price: float
    quantity: int
    side: OrderSide
    exchange: str = "unknown"
    buyer_order_id: Optional[str] = None
    seller_order_id: Optional[str] = None

@dataclass
class MarketDepth:
    """Market depth (order book) snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # (price, quantity)
    asks: List[Tuple[float, int]]  # (price, quantity)
    exchange: str = "unknown"
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

@dataclass
class MicrostructureFeatures:
    """Container for microstructure features"""
    timestamp: datetime
    symbol: str
    
    # Spread features
    bid_ask_spread: float = 0.0
    effective_spread: float = 0.0
    realized_spread: float = 0.0
    spread_volatility: float = 0.0
    
    # Liquidity features
    bid_depth: int = 0
    ask_depth: int = 0
    total_depth: int = 0
    depth_imbalance: float = 0.0
    liquidity_ratio: float = 0.0
    
    # Order flow features
    order_flow_imbalance: float = 0.0
    trade_intensity: float = 0.0
    order_intensity: float = 0.0
    cancel_ratio: float = 0.0
    
    # Price impact features
    price_impact: float = 0.0
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0
    
    # Volatility features
    microstructure_volatility: float = 0.0
    price_clustering: float = 0.0
    tick_frequency: float = 0.0
    
    # Market quality features
    market_efficiency: float = 0.0
    information_share: float = 0.0
    adverse_selection: float = 0.0

class MarketMicrostructureAnalyzer:
    """
    Market Microstructure Analyzer
    
    Educational Notes:
    - Market microstructure reveals hidden supply/demand dynamics
    - Order flow often precedes price movements
    - Spread patterns indicate market conditions
    - Liquidity analysis helps estimate slippage
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.order_book = {}  # symbol -> MarketDepth
        self.recent_trades = {}  # symbol -> deque of Trades
        self.recent_orders = {}  # symbol -> deque of Orders
        self.spread_history = {}  # symbol -> deque of spreads
        self.depth_history = {}  # symbol -> deque of depth measurements
        
        logger.info("MarketMicrostructureAnalyzer initialized")
    
    def update_order_book(self, market_depth: MarketDepth):
        """
        Update order book with new market depth data
        """
        symbol = market_depth.symbol
        
        # Store current order book
        self.order_book[symbol] = market_depth
        
        # Update spread history
        if market_depth.spread is not None:
            if symbol not in self.spread_history:
                self.spread_history[symbol] = deque(maxlen=self.window_size)
            self.spread_history[symbol].append(market_depth.spread)
        
        # Update depth history
        bid_depth = sum(quantity for _, quantity in market_depth.bids[:10])
        ask_depth = sum(quantity for _, quantity in market_depth.asks[:10])
        total_depth = bid_depth + ask_depth
        
        if symbol not in self.depth_history:
            self.depth_history[symbol] = deque(maxlen=self.window_size)
        self.depth_history[symbol].append({
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'timestamp': market_depth.timestamp
        })
    
    def add_trade(self, trade: Trade):
        """
        Add a new trade to the history
        """
        symbol = trade.symbol
        
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = deque(maxlen=self.window_size)
        
        self.recent_trades[symbol].append(trade)
    
    def add_order(self, order: Order):
        """
        Add a new order to the history
        """
        symbol = order.symbol
        
        if symbol not in self.recent_orders:
            self.recent_orders[symbol] = deque(maxlen=self.window_size)
        
        self.recent_orders[symbol].append(order)
    
    def calculate_spread_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate spread-related features
        
        Educational: Spreads indicate market liquidity and information asymmetry.
        Tight spreads suggest liquid markets, wide spreads suggest uncertainty.
        """
        if symbol not in self.order_book or symbol not in self.spread_history:
            return {}
        
        current_book = self.order_book[symbol]
        spread_history = list(self.spread_history[symbol])
        
        features = {}
        
        # Current bid-ask spread
        if current_book.spread is not None:
            features['bid_ask_spread'] = current_book.spread
            
            # Relative spread (as percentage of mid price)
            if current_book.mid_price is not None:
                features['relative_spread'] = current_book.spread / current_book.mid_price
        
        # Spread statistics
        if len(spread_history) > 1:
            features['spread_mean'] = np.mean(spread_history)
            features['spread_std'] = np.std(spread_history)
            features['spread_volatility'] = features['spread_std'] / features['spread_mean'] if features['spread_mean'] > 0 else 0
            
            # Spread trend
            recent_spreads = spread_history[-10:]
            if len(recent_spreads) >= 2:
                features['spread_trend'] = (recent_spreads[-1] - recent_spreads[0]) / len(recent_spreads)
        
        # Effective spread (based on recent trades)
        if symbol in self.recent_trades and len(self.recent_trades[symbol]) > 0:
            recent_trades = list(self.recent_trades[symbol])[-20:]  # Last 20 trades
            
            if current_book.mid_price is not None:
                effective_spreads = []
                for trade in recent_trades:
                    effective_spread.append(abs(trade.price - current_book.mid_price))
                
                if effective_spreads:
                    features['effective_spread'] = np.mean(effective_spreads)
                    features['effective_spread_std'] = np.std(effective_spreads)
        
        return features
    
    def calculate_liquidity_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate liquidity-related features
        
        Educational: Liquidity affects transaction costs and price impact.
        Deep markets absorb large orders with minimal price movement.
        """
        if symbol not in self.order_book or symbol not in self.depth_history:
            return {}
        
        current_book = self.order_book[symbol]
        depth_history = list(self.depth_history[symbol])
        
        features = {}
        
        # Current depth measures
        if current_book.bids and current_book.asks:
            # Depth at best levels
            features['bid_depth_best'] = current_book.bids[0][1]
            features['ask_depth_best'] = current_book.asks[0][1]
            
            # Depth at top 5 levels
            features['bid_depth_5'] = sum(quantity for _, quantity in current_book.bids[:5])
            features['ask_depth_5'] = sum(quantity for _, quantity in current_book.asks[:5])
            
            # Depth at top 10 levels
            features['bid_depth_10'] = sum(quantity for _, quantity in current_book.bids[:10])
            features['ask_depth_10'] = sum(quantity for _, quantity in current_book.asks[:10])
            
            # Total depth
            features['total_depth_5'] = features['bid_depth_5'] + features['ask_depth_5']
            features['total_depth_10'] = features['bid_depth_10'] + features['ask_depth_10']
            
            # Depth imbalance
            if features['total_depth_5'] > 0:
                features['depth_imbalance_5'] = (features['bid_depth_5'] - features['ask_depth_5']) / features['total_depth_5']
            
            if features['total_depth_10'] > 0:
                features['depth_imbalance_10'] = (features['bid_depth_10'] - features['ask_depth_10']) / features['total_depth_10']
            
            # Liquidity ratio (depth relative to spread)
            if current_book.spread is not None and current_book.spread > 0:
                features['liquidity_ratio_5'] = features['total_depth_5'] / current_book.spread
                features['liquidity_ratio_10'] = features['total_depth_10'] / current_book.spread
        
        # Depth dynamics
        if len(depth_history) > 1:
            recent_depths = depth_history[-20:]  # Last 20 measurements
            
            # Depth volatility
            total_depths = [d['total_depth'] for d in recent_depths]
            if total_depths:
                features['depth_volatility'] = np.std(total_depths) / np.mean(total_depths) if np.mean(total_depths) > 0 else 0
            
            # Depth trend
            if len(total_depths) >= 2:
                features['depth_trend'] = (total_depths[-1] - total_depths[0]) / len(total_depths)
        
        return features
    
    def calculate_order_flow_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate order flow features
        
        Educational: Order flow reveals buying/selling pressure before it's reflected in prices.
        Imbalance in order flow often predicts short-term price movements.
        """
        if symbol not in self.recent_orders and symbol not in self.recent_trades:
            return {}
        
        features = {}
        
        # Trade flow analysis
        if symbol in self.recent_trades and len(self.recent_trades[symbol]) > 0:
            recent_trades = list(self.recent_trades[symbol])[-50:]  # Last 50 trades
            
            # Buy vs sell volume
            buy_volume = sum(t.quantity for t in recent_trades if t.side in [OrderSide.BUY, OrderSide.BUY_COVER])
            sell_volume = sum(t.quantity for t in recent_trades if t.side in [OrderSide.SELL, OrderSide.SELL_SHORT])
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                features['trade_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
                features['buy_ratio'] = buy_volume / total_volume
                features['sell_ratio'] = sell_volume / total_volume
            
            # Trade intensity (trades per time unit)
            if len(recent_trades) >= 2:
                time_span = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
                if time_span > 0:
                    features['trade_intensity'] = len(recent_trades) / time_span * 60  # trades per minute
            
            # Volume-weighted average price (VWAP) deviation
            if symbol in self.order_book and self.order_book[symbol].mid_price is not None:
                vwap = sum(t.price * t.quantity for t in recent_trades) / sum(t.quantity for t in recent_trades)
                mid_price = self.order_book[symbol].mid_price
                features['vwap_deviation'] = (vwap - mid_price) / mid_price if mid_price > 0 else 0
        
        # Order flow analysis
        if symbol in self.recent_orders and len(self.recent_orders[symbol]) > 0:
            recent_orders = list(self.recent_orders[symbol])[-100:]  # Last 100 orders
            
            # Order submission vs cancellation
            submitted_orders = [o for o in recent_orders if o.filled_quantity == 0]
            cancelled_orders = [o for o in recent_orders if o.remaining_quantity < o.quantity and o.filled_quantity < o.quantity]
            
            if len(submitted_orders) > 0:
                features['cancel_ratio'] = len(cancelled_orders) / len(submitted_orders)
            
            # Order intensity
            if len(recent_orders) >= 2:
                time_span = (recent_orders[-1].timestamp - recent_orders[0].timestamp).total_seconds()
                if time_span > 0:
                    features['order_intensity'] = len(recent_orders) / time_span * 60  # orders per minute
            
            # Order type distribution
            market_orders = [o for o in recent_orders if o.order_type == OrderType.MARKET]
            limit_orders = [o for o in recent_orders if o.order_type == OrderType.LIMIT]
            
            if len(recent_orders) > 0:
                features['market_order_ratio'] = len(market_orders) / len(recent_orders)
                features['limit_order_ratio'] = len(limit_orders) / len(recent_orders)
        
        return features
    
    def calculate_price_impact_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate price impact features
        
        Educational: Price impact measures how much prices move due to trading activity.
        Understanding impact helps estimate trading costs and optimize execution.
        """
        if symbol not in self.recent_trades or len(self.recent_trades[symbol]) < 2:
            return {}
        
        features = {}
        recent_trades = list(self.recent_trades[symbol])[-50:]  # Last 50 trades
        
        if len(recent_trades) < 2:
            return features
        
        # Calculate price changes
        prices = [t.price for t in recent_trades]
        quantities = [t.quantity for t in recent_trades]
        
        # Price impact per unit volume
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        volumes = quantities[1:]  # Align with price changes
        
        if len(price_changes) > 0 and len(volumes) > 0:
            # Simple price impact
            impacts = [abs(pc) / vol if vol > 0 else 0 for pc, vol in zip(price_changes, volumes)]
            features['price_impact'] = np.mean(impacts) if impacts else 0
            features['price_impact_std'] = np.std(impacts) if impacts else 0
            
            # Temporary vs permanent impact
            if len(recent_trades) >= 10:
                # Temporary impact: immediate price reversal
                temporary_impacts = []
                permanent_impacts = []
                
                for i in range(1, min(len(recent_trades)-5, len(recent_trades))):
                    trade_price = recent_trades[i].price
                    prev_price = recent_trades[i-1].price
                    future_price = recent_trades[i+5].price  # Price 5 trades later
                    
                    immediate_change = trade_price - prev_price
                    permanent_change = future_price - prev_price
                    temporary_change = immediate_change - permanent_change
                    
                    if quantities[i] > 0:
                        temporary_impacts.append(abs(temporary_change) / quantities[i])
                        permanent_impacts.append(abs(permanent_change) / quantities[i])
                
                if temporary_impacts:
                    features['temporary_impact'] = np.mean(temporary_impacts)
                    features['permanent_impact'] = np.mean(permanent_impacts)
                    features['impact_permanence'] = features['permanent_impact'] / (features['temporary_impact'] + features['permanent_impact']) if (features['temporary_impact'] + features['permanent_impact']) > 0 else 0
        
        # Market depth impact
        if symbol in self.order_book and self.order_book[symbol].total_depth > 0:
            total_depth = sum(q for _, q in self.order_book[symbol].bids[:5]) + sum(q for _, q in self.order_book[symbol].asks[:5])
            
            if total_depth > 0 and recent_trades:
                avg_trade_size = np.mean([t.quantity for t in recent_trades[-10:]])
                features['depth_utilization'] = avg_trade_size / total_depth
        
        return features
    
    def calculate_volatility_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate microstructure volatility features
        
        Educational: Microstructure volatility differs from price volatility.
        It captures high-frequency price fluctuations and noise.
        """
        if symbol not in self.recent_trades or len(self.recent_trades[symbol]) < 2:
            return {}
        
        features = {}
        recent_trades = list(self.recent_trades[symbol])[-100:]  # Last 100 trades
        
        if len(recent_trades) < 2:
            return features
        
        # Price returns at trade frequency
        prices = [t.price for t in recent_trades]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        
        if returns:
            features['microstructure_volatility'] = np.std(returns) * np.sqrt(len(returns))  # Annualized equivalent
            
            # High-frequency volatility
            if len(returns) >= 10:
                features['hf_volatility_10'] = np.std(returns[-10:])
                features['hf_volatility_trend'] = np.std(returns[-10:]) - np.std(returns[-20:-10]) if len(returns) >= 20 else 0
        
        # Price clustering (tendency for prices to cluster at certain levels)
        if len(prices) >= 10:
            price_rounded = [round(p, 2) for p in prices]  # Round to cents
            unique_prices = len(set(price_rounded))
            features['price_clustering'] = 1 - (unique_prices / len(price_rounded))  # Higher = more clustering
        
        # Tick frequency (how often price changes by minimum tick)
        if len(prices) >= 2:
            tick_changes = 0
            for i in range(1, len(prices)):
                if abs(prices[i] - prices[i-1]) > 0:
                    tick_changes += 1
            
            features['tick_frequency'] = tick_changes / len(prices)
        
        # Spread volatility (if available)
        if symbol in self.spread_history and len(self.spread_history[symbol]) > 1:
            spreads = list(self.spread_history[symbol])
            features['spread_volatility'] = np.std(spreads) / np.mean(spreads) if np.mean(spreads) > 0 else 0
        
        return features
    
    def calculate_market_quality_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate market quality features
        
        Educational: Market quality measures how efficiently markets incorporate information.
        High-quality markets have low transaction costs and fast information incorporation.
        """
        features = {}
        
        # Market efficiency (price discovery)
        if symbol in self.recent_trades and len(self.recent_trades[symbol]) >= 10:
            recent_trades = list(self.recent_trades[symbol])[-50:]
            
            # Price efficiency: how quickly prices revert to equilibrium
            prices = [t.price for t in recent_trades]
            if len(prices) >= 10:
                # Autocorrelation of returns (lower = more efficient)
                returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
                if len(returns) >= 10:
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    features['market_efficiency'] = 1 - abs(autocorr) if not np.isnan(autocorr) else 0
        
        # Information share (how much each venue contributes to price discovery)
        if symbol in self.recent_trades:
            recent_trades = list(self.recent_trades[symbol])[-100:]
            
            # Group by exchange
            exchange_trades = {}
            for trade in recent_trades:
                if trade.exchange not in exchange_trades:
                    exchange_trades[trade.exchange] = []
                exchange_trades[trade.exchange].append(trade)
            
            # Calculate information share based on volume and timing
            total_volume = sum(t.quantity for t in recent_trades)
            for exchange, trades in exchange_trades.items():
                exchange_volume = sum(t.quantity for t in trades)
                features[f'info_share_{exchange}'] = exchange_volume / total_volume if total_volume > 0 else 0
        
        # Adverse selection risk
        if symbol in self.order_book and symbol in self.recent_trades:
            current_book = self.order_book[symbol]
            recent_trades = list(self.recent_trades[symbol])[-20:]
            
            if current_book.mid_price is not None and recent_trades:
                # Measure how often trades occur at unfavorable prices
                unfavorable_trades = 0
                for trade in recent_trades:
                    if trade.side in [OrderSide.BUY, OrderSide.BUY_COVER]:
                        # Buy trades above mid price are unfavorable
                        if trade.price > current_book.mid_price:
                            unfavorable_trades += 1
                    else:
                        # Sell trades below mid price are unfavorable
                        if trade.price < current_book.mid_price:
                            unfavorable_trades += 1
                
                features['adverse_selection'] = unfavorable_trades / len(recent_trades)
        
        # Order processing cost
        if symbol in self.spread_history and len(self.spread_history[symbol]) > 0:
            spreads = list(self.spread_history[symbol])
            features['order_processing_cost'] = np.mean(spreads) / 2  # Half-spread as proxy
        
        return features
    
    def calculate_all_features(self, symbol: str) -> MicrostructureFeatures:
        """
        Calculate all microstructure features for a symbol
        """
        timestamp = datetime.now()
        
        # Calculate all feature groups
        spread_features = self.calculate_spread_features(symbol)
        liquidity_features = self.calculate_liquidity_features(symbol)
        order_flow_features = self.calculate_order_flow_features(symbol)
        impact_features = self.calculate_price_impact_features(symbol)
        volatility_features = self.calculate_volatility_features(symbol)
        quality_features = self.calculate_market_quality_features(symbol)
        
        # Create feature container
        features = MicrostructureFeatures(timestamp=timestamp, symbol=symbol)
        
        # Update features with calculated values
        for feature_dict in [spread_features, liquidity_features, order_flow_features, 
                           impact_features, volatility_features, quality_features]:
            for key, value in feature_dict.items():
                if hasattr(features, key):
                    setattr(features, key, value)
        
        return features
    
    def get_market_regime(self, symbol: str) -> str:
        """
        Determine current market regime based on microstructure features
        
        Educational: Market regimes help adapt strategies to current conditions.
        Different regimes require different approaches.
        """
        features = self.calculate_all_features(symbol)
        
        # Define regime thresholds
        if features.spread_volatility > 0.5:
            return "high_volatility"
        elif features.depth_imbalance > 0.3:
            return "buy_pressure"
        elif features.depth_imbalance < -0.3:
            return "sell_pressure"
        elif features.liquidity_ratio < 100:
            return "low_liquidity"
        elif features.trade_intensity > 10:
            return "high_activity"
        elif features.adverse_selection > 0.6:
            return "high_risk"
        else:
            return "normal"
    
    def estimate_slippage(self, symbol: str, order_size: int, order_side: OrderSide) -> Dict[str, float]:
        """
        Estimate slippage for a given order size
        
        Educational: Slippage estimation helps set realistic expectations
        and optimize order execution strategies.
        """
        if symbol not in self.order_book:
            return {'estimated_slippage': 0.0, 'confidence': 0.0}
        
        current_book = self.order_book[symbol]
        features = self.calculate_all_features(symbol)
        
        slippage_estimate = 0.0
        remaining_size = order_size
        
        if order_side in [OrderSide.BUY, OrderSide.BUY_COVER]:
            # Buy order - walk through the book
            for price, quantity in current_book.asks:
                if remaining_size <= 0:
                    break
                
                filled_size = min(remaining_size, quantity)
                slippage_estimate += (price - current_book.mid_price) * filled_size if current_book.mid_price else 0
                remaining_size -= filled_size
        else:
            # Sell order - walk through the book
            for price, quantity in current_book.bids:
                if remaining_size <= 0:
                    break
                
                filled_size = min(remaining_size, quantity)
                slippage_estimate += (current_book.mid_price - price) * filled_size if current_book.mid_price else 0
                remaining_size -= filled_size
        
        # Average slippage per share
        avg_slippage = slippage_estimate / order_size if order_size > 0 else 0
        
        # Adjust based on market conditions
        volatility_adjustment = features.microstructure_volatility * 0.5
        liquidity_adjustment = max(0, (1 - features.liquidity_ratio / 1000)) * 0.1
        
        total_slippage = avg_slippage + volatility_adjustment + liquidity_adjustment
        
        # Confidence based on market quality
        confidence = features.market_efficiency * (1 - features.adverse_selection)
        
        return {
            'estimated_slippage': total_slippage,
            'confidence': max(0, min(1, confidence)),
            'market_regime': self.get_market_regime(symbol),
            'liquidity_ratio': features.liquidity_ratio,
            'spread_volatility': features.spread_volatility
        }

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Real-time Market Monitoring:
   analyzer = MarketMicrostructureAnalyzer()
   
   # Update with market data
   analyzer.update_order_book(market_depth_data)
   analyzer.add_trade(trade_data)
   analyzer.add_order(order_data)
   
   # Get current features
   features = analyzer.calculate_all_features("AAPL")
   print(f"Spread: {features.bid_ask_spread}, Liquidity: {features.liquidity_ratio}")

2. Market Regime Detection:
   regime = analyzer.get_market_regime("AAPL")
   if regime == "high_volatility":
       print("High volatility regime - widen stops")
   elif regime == "low_liquidity":
       print("Low liquidity - reduce position size")

3. Slippage Estimation:
   slippage = analyzer.estimate_slippage("AAPL", 1000, OrderSide.BUY)
   print(f"Estimated slippage: ${slippage['estimated_slippage']:.4f} per share")

4. Liquidity Analysis:
   liquidity_features = analyzer.calculate_liquidity_features("AAPL")
   if liquidity_features['depth_imbalance_5'] > 0.3:
       print("Strong buying pressure detected")

5. Order Flow Analysis:
   flow_features = analyzer.calculate_order_flow_features("AAPL")
   if flow_features['trade_flow_imbalance'] > 0.2:
       print("Net buying flow - bullish signal")

Key Insights:
- Market microstructure reveals hidden supply/demand dynamics
- Order flow often leads price movements
- Spread patterns indicate market conditions and risk
- Liquidity analysis is crucial for execution planning
- Market regimes help adapt strategies to current conditions
- Slippage estimation improves trade planning and cost control
"""