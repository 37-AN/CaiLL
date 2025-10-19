"""
Trading Fundamentals Module - Interactive Learning System

This module provides comprehensive education about trading fundamentals,
including market structure, order types, analysis methods, and risk management.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class Lesson:
    """Structure for educational lessons"""
    id: str
    title: str
    content: str
    difficulty: str  # beginner, intermediate, advanced
    duration_minutes: int
    interactive_elements: List[str]
    quiz_questions: List[Dict]
    examples: List[Dict]


class TradingBasicsModule:
    """
    Interactive trading fundamentals education module
    """
    
    def __init__(self):
        self.lessons = self._initialize_lessons()
        self.user_progress = {}
        
    def _initialize_lessons(self) -> Dict[str, Lesson]:
        """Initialize all trading fundamentals lessons"""
        
        lessons = {
            "market_structure": Lesson(
                id="market_structure",
                title="Understanding Market Structure",
                difficulty="beginner",
                duration_minutes=25,
                content=self._get_market_structure_content(),
                interactive_elements=["market_simulation", "order_book_visualization"],
                quiz_questions=self._get_market_structure_quiz(),
                examples=self._get_market_structure_examples()
            ),
            
            "order_types": Lesson(
                id="order_types",
                title="Order Types and Execution",
                difficulty="beginner",
                duration_minutes=30,
                content=self._get_order_types_content(),
                interactive_elements=["order_execution_simulator", "slippage_calculator"],
                quiz_questions=self._get_order_types_quiz(),
                examples=self._get_order_types_examples()
            ),
            
            "technical_analysis": Lesson(
                id="technical_analysis",
                title="Introduction to Technical Analysis",
                difficulty="intermediate",
                duration_minutes=45,
                content=self._get_technical_analysis_content(),
                interactive_elements=["chart_pattern_recognition", "indicator_calculator"],
                quiz_questions=self._get_technical_analysis_quiz(),
                examples=self._get_technical_analysis_examples()
            ),
            
            "fundamental_analysis": Lesson(
                id="fundamental_analysis",
                title="Fundamental Analysis Basics",
                difficulty="intermediate",
                duration_minutes=40,
                content=self._get_fundamental_analysis_content(),
                interactive_elements=["financial_statement_analyzer", "valuation_calculator"],
                quiz_questions=self._get_fundamental_analysis_quiz(),
                examples=self._get_fundamental_analysis_examples()
            ),
            
            "risk_management": Lesson(
                id="risk_management",
                title="Risk Management Principles",
                difficulty="beginner",
                duration_minutes=35,
                content=self._get_risk_management_content(),
                interactive_elements=["position_size_calculator", "risk_simulator"],
                quiz_questions=self._get_risk_management_quiz(),
                examples=self._get_risk_management_examples()
            ),
            
            "market_psychology": Lesson(
                id="market_psychology",
                title="Market Psychology and Behavioral Finance",
                difficulty="advanced",
                duration_minutes=30,
                content=self._get_market_psychology_content(),
                interactive_elements=["bias_detector", "sentiment_analyzer"],
                quiz_questions=self._get_market_psychology_quiz(),
                examples=self._get_market_psychology_examples()
            )
        }
        
        return lessons
    
    def _get_market_structure_content(self) -> str:
        """Get market structure lesson content"""
        return """
        # Understanding Market Structure
        
        ## What is a Market?
        A market is a place where buyers and sellers come together to exchange goods, services, or financial instruments. In financial markets, these instruments include stocks, bonds, currencies, and derivatives.
        
        ## Market Participants
        
        ### Market Makers
        - **Role**: Provide liquidity by being willing to buy at the bid price and sell at the ask price
        - **Profit**: Earn the bid-ask spread
        - **Example**: NYSE designated market makers, crypto market makers
        
        ### Retail Traders
        - **Role**: Individual investors trading for personal accounts
        - **Characteristics**: Smaller position sizes, longer holding periods
        - **Impact**: Provide liquidity and market depth
        
        ### Institutional Traders
        - **Role**: Large organizations managing substantial capital
        - **Types**: Hedge funds, mutual funds, pension funds, investment banks
        - **Impact**: Can move markets with large orders
        
        ### High-Frequency Traders (HFT)
        - **Role**: Use technology and algorithms to trade at high speeds
        - **Strategy**: Exploit small price inefficiencies
        - **Impact**: Provide liquidity but can increase volatility
        
        ## Market Microstructure
        
        ### Order Book
        The order book shows all current buy and sell orders at different price levels:
        
        ```
        ASK (Sell Orders)    |    BID (Buy Orders)
        ---------------------|---------------------
        100.05    100 shares | 100.00    200 shares
        100.04    150 shares | 99.99     300 shares
        100.03    200 shares | 99.98     250 shares
        ```
        
        ### Bid-Ask Spread
        - **Definition**: Difference between highest bid and lowest ask
        - **Significance**: Cost of immediate execution
        - **Factors affecting spread**: Liquidity, volatility, market conditions
        
        ### Market Depth
        - **Definition**: Quantity available at each price level
        - **Importance**: Indicates market liquidity and potential price impact
        
        ## Market Types
        
        ### Auction Markets
        - **Process**: All orders arrive simultaneously and are matched at a single price
        - **Example**: Opening auction on NYSE
        - **Advantage**: Price discovery and fairness
        
        ### Continuous Markets
        - **Process**: Orders can be placed and executed at any time during market hours
        - **Example**: Most electronic markets
        - **Advantage**: Immediate execution possible
        
        ## Key Concepts
        
        ### Liquidity
        - **Definition**: Ease of buying or selling without affecting price
        - **High liquidity**: Small spreads, minimal price impact
        - **Low liquidity**: Large spreads, significant price impact
        
        ### Price Discovery
        - **Definition**: Process of determining market price through buyer-seller interaction
        - **Factors**: Supply, demand, information, expectations
        
        ### Market Efficiency
        - **Efficient Market Hypothesis**: Prices reflect all available information
        - **Weak form**: Past prices don't predict future prices
        - **Semi-strong form**: Public information is reflected in prices
        - **Strong form**: All information (public and private) is reflected in prices
        """
    
    def _get_order_types_content(self) -> str:
        """Get order types lesson content"""
        return """
        # Order Types and Execution
        
        ## Market Orders
        
        ### Definition
        An order to buy or sell immediately at the current market price.
        
        ### Characteristics
        - **Guaranteed execution**: Will be filled
        - **Uncertain price**: May get worse than expected price
        - **Best for**: When execution speed is more important than price
        
        ### Example
        ```
        Market Order: Buy 100 shares of AAPL
        Result: Bought at $150.25 (current market price)
        ```
        
        ## Limit Orders
        
        ### Definition
        An order to buy or sell at a specific price or better.
        
        ### Characteristics
        - **Price guarantee**: Will not execute worse than specified price
        - **No execution guarantee**: May not fill if price doesn't reach limit
        - **Best for**: When price control is more important than speed
        
        ### Example
        ```
        Limit Order: Buy 100 shares of AAPL at $150.00 or lower
        Result: Will only execute if price drops to $150.00 or below
        ```
        
        ## Stop Orders
        
        ### Stop Loss Orders
        An order that becomes a market order when a specified price is reached.
        
        **Purpose**: Limit losses on existing positions
        
        **Example**:
        ```
        You own AAPL at $150.00
        Stop Loss: Sell if price drops to $145.00
        Result: Automatically sells if price hits $145.00
        ```
        
        ### Stop Entry Orders
        An order that becomes a market order when a specified price is reached.
        
        **Purpose**: Enter positions on breakout movements
        
        **Example**:
        ```
        AAPL is trading at $150.00
        Stop Entry: Buy if price rises to $155.00
        Result: Automatically buys if price breaks out to $155.00
        ```
        
        ## Advanced Order Types
        
        ### Stop Limit Orders
        Combines stop and limit order features.
        
        **Example**:
        ```
        Stop Price: $145.00 (triggers the order)
        Limit Price: $144.90 (worst execution price)
        Result: Becomes limit order when stop price is hit
        ```
        
        ### Trailing Stop Orders
        Stop price that follows the market price at a specified distance.
        
        **Example**:
        ```
        You own AAPL at $150.00
        Trailing Stop: $5.00 below current price
        If price rises to $160.00, stop becomes $155.00
        If price drops to $155.00, order triggers
        ```
        
        ### Iceberg Orders
        Large orders broken into smaller pieces to hide the true order size.
        
        **Purpose**: Minimize market impact
        
        ### Time-in-Force Instructions
        
        **Day**: Order valid only for the current trading day
        **GTC (Good Till Canceled)**: Order remains active until filled or canceled
        **IOC (Immediate or Cancel)**: Must execute immediately; unfilled portion is canceled
        **FOK (Fill or Kill)**: Must execute completely and immediately or be canceled
        
        ## Order Execution Process
        
        ### 1. Order Placement
        - Order enters the market
        - System checks for validity
        - Order is routed to appropriate venue
        
        ### 2. Order Matching
        - System looks for matching orders
        - Price-time priority determines execution order
        - Trades are executed when matches are found
        
        ### 3. Trade Confirmation
        - Execution details are recorded
        - Confirmation is sent to both parties
        - Settlement process begins
        
        ## Execution Quality Factors
        
        ### Slippage
        Difference between expected and actual execution price.
        
        **Causes**:
        - Market movement between order and execution
        - Large orders moving the market
        - Low liquidity conditions
        
        ### Fill Rate
        Percentage of orders that get executed.
        
        ### Execution Speed
        Time from order placement to execution.
        
        ## Best Execution Practices
        
        ### 1. Choose Appropriate Order Type
        - Market orders for speed
        - Limit orders for price control
        - Stop orders for risk management
        
        ### 2. Consider Market Conditions
        - High volatility: Use limit orders
        - Low liquidity: Break up large orders
        - Fast markets: Be prepared for slippage
        
        ### 3. Monitor Execution Quality
        - Track slippage
        - Compare to benchmarks
        - Adjust strategy based on results
        """
    
    def _get_technical_analysis_content(self) -> str:
        """Get technical analysis lesson content"""
        return """
        # Introduction to Technical Analysis
        
        ## What is Technical Analysis?
        
        Technical analysis is a method of evaluating securities by analyzing statistics generated by market activity, such as past prices and volume. Technical analysts believe that:
        
        1. **Price reflects everything**: All known information is already priced in
        2. **Prices move in trends**: Prices tend to move in identifiable trends
        3. **History repeats itself**: Market patterns are repetitive and predictable
        
        ## Chart Types
        
        ### Line Charts
        - **Construction**: Connect closing prices over time
        - **Use**: Simple trend identification
        - **Limitation**: Ignores price range within periods
        
        ### Bar Charts
        - **Components**: Open, High, Low, Close (OHLC)
        - **Advantage**: Shows price range and volatility
        - **Use**: Detailed price analysis
        
        ### Candlestick Charts
        - **Components**: OHLC with visual representation
        - **Color coding**: Green/white for up, red/black for down
        - **Use**: Pattern recognition and sentiment analysis
        
        ## Trend Analysis
        
        ### Uptrend
        - **Definition**: Series of higher highs and higher lows
        - **Confirmation**: Price above rising moving average
        - **Trading**: Look for buying opportunities on dips
        
        ### Downtrend
        - **Definition**: Series of lower highs and lower lows
        - **Confirmation**: Price below falling moving average
        - **Trading**: Look for selling opportunities on rallies
        
        ### Sideways Trend
        - **Definition**: Price moves in horizontal range
        - **Trading**: Range-bound strategies
        
        ## Support and Resistance
        
        ### Support
        - **Definition**: Price level where buying pressure overcomes selling pressure
        - **Significance**: Previous lows, moving averages, Fibonacci levels
        - **Trading**: Buy near support, place stop below
        
        ### Resistance
        - **Definition**: Price level where selling pressure overcomes buying pressure
        - **Significance**: Previous highs, moving averages, Fibonacci levels
        - **Trading**: Sell near resistance, place stop above
        
        ## Technical Indicators
        
        ### Trend Indicators
        
        **Moving Averages**
        - **Simple Moving Average (SMA)**: Average price over specified period
        - **Exponential Moving Average (EMA)**: Gives more weight to recent prices
        - **Use**: Trend identification, support/resistance levels
        
        **Moving Average Convergence Divergence (MACD)**
        - **Components**: MACD line, signal line, histogram
        - **Signals**: Crossovers, divergence, overbought/oversold
        
        ### Momentum Indicators
        
        **Relative Strength Index (RSI)**
        - **Range**: 0-100
        - **Overbought**: Above 70
        - **Oversold**: Below 30
        - **Use**: Identify potential reversals
        
        **Stochastic Oscillator**
        - **Range**: 0-100
        - **Overbought**: Above 80
        - **Oversold**: Below 20
        - **Use**: Momentum analysis
        
        ### Volatility Indicators
        
        **Bollinger Bands**
        - **Components**: Middle band (SMA), upper and lower bands (standard deviations)
        - **Use**: Volatility measurement, overbought/oversold levels
        
        **Average True Range (ATR)**
        - **Purpose**: Measure volatility
        - **Use**: Position sizing, stop placement
        
        ### Volume Indicators
        
        **On-Balance Volume (OBV)**
        - **Calculation**: Cumulative volume based on price direction
        - **Use**: Confirm price trends
        
        **Volume Profile**
        - **Purpose**: Show volume at price levels
        - **Use**: Identify significant price levels
        
        ## Chart Patterns
        
        ### Reversal Patterns
        
        **Head and Shoulders**
        - **Signal**: Trend reversal from up to down
        - **Confirmation**: Break below neckline
        
        **Double Top/Bottom**
        - **Signal**: Trend reversal
        - **Confirmation**: Break below/above support/resistance
        
        ### Continuation Patterns
        
        **Triangles**
        - **Types**: Ascending, descending, symmetrical
        - **Signal**: Trend continuation after breakout
        
        **Flags and Pennants**
        - **Signal**: Brief consolidation before trend continuation
        - **Duration**: Short-term patterns
        
        ## Technical Analysis Strategies
        
        ### Trend Following
        - **Logic**: Buy when price is above moving average, sell when below
        - **Indicators**: Moving averages, MACD, ADX
        - **Best for**: Markets with clear trends
        
        ### Mean Reversion
        - **Logic**: Buy when oversold, sell when overbought
        - **Indicators**: RSI, Stochastic, Bollinger Bands
        - **Best for**: Range-bound markets
        
        ### Breakout Trading
        - **Logic**: Buy when price breaks above resistance, sell below support
        - **Confirmation**: Volume increase, pattern completion
        - **Risk**: False breakouts
        
        ## Limitations of Technical Analysis
        
        1. **Subjectivity**: Different analysts may interpret charts differently
        2. **False signals**: No indicator is 100% accurate
        3. **Self-fulfilling prophecies**: Patterns may work because traders believe they work
        4. **Market changes**: Patterns may not work in all market conditions
        
        ## Best Practices
        
        1. **Use multiple indicators**: Confirm signals with different tools
        2. **Combine with other analysis**: Fundamental analysis, market sentiment
        3. **Risk management**: Always use stop losses
        4. **Backtesting**: Test strategies on historical data
        5. **Continuous learning**: Markets evolve, so should your analysis
        """
    
    def _get_fundamental_analysis_content(self) -> str:
        """Get fundamental analysis lesson content"""
        return """
        # Fundamental Analysis Basics
        
        ## What is Fundamental Analysis?
        
        Fundamental analysis is a method of evaluating a security's intrinsic value by examining related economic, financial, and other qualitative and quantitative factors. The goal is to determine whether a security is overvalued or undervalued.
        
        ## Core Principles
        
        ### 1. Intrinsic Value
        Every company has an underlying value based on its fundamentals, regardless of current market price.
        
        ### 2. Market Inefficiency
        Market prices can deviate from intrinsic value due to emotions, speculation, or information asymmetry.
        
        ### 3. Long-Term Perspective
        Fundamental analysis focuses on long-term value rather than short-term price movements.
        
        ## Financial Statements
        
        ### Income Statement
        Shows company's financial performance over a specific period.
        
        **Key Components**:
        - **Revenue**: Total sales of goods/services
        - **Cost of Goods Sold (COGS)**: Direct costs of producing goods
        - **Gross Profit**: Revenue - COGS
        - **Operating Expenses**: Sales, general, and administrative costs
        - **Operating Income**: Gross Profit - Operating Expenses
        - **Net Income**: Profit after all expenses and taxes
        
        **Important Metrics**:
        - **Gross Margin**: Gross Profit / Revenue
        - **Operating Margin**: Operating Income / Revenue
        - **Net Margin**: Net Income / Revenue
        
        ### Balance Sheet
        Snapshot of company's financial position at a specific point in time.
        
        **Assets**: What the company owns
        - **Current Assets**: Cash, accounts receivable, inventory (convertible to cash within 1 year)
        - **Non-Current Assets**: Property, plant, equipment, intangible assets
        
        **Liabilities**: What the company owes
        - **Current Liabilities**: Accounts payable, short-term debt (due within 1 year)
        - **Non-Current Liabilities**: Long-term debt, deferred tax liabilities
        
        **Equity**: Owners' residual interest
        - **Share Capital**: Money invested by shareholders
        - **Retained Earnings**: Accumulated profits not paid as dividends
        
        **Key Formula**: Assets = Liabilities + Equity
        
        ### Cash Flow Statement
        Shows how cash moves in and out of the company.
        
        **Operating Activities**: Cash from main business operations
        **Investing Activities**: Cash from buying/selling long-term assets
        **Financing Activities**: Cash from raising capital and debt
        
        **Key Metric**: Free Cash Flow = Operating Cash Flow - Capital Expenditures
        
        ## Financial Ratios
        
        ### Valuation Ratios
        
        **Price-to-Earnings (P/E) Ratio**
        ```
        P/E = Stock Price / Earnings per Share
        ```
        - **Interpretation**: How much investors pay for $1 of earnings
        - **Comparison**: Industry average, historical average
        
        **Price-to-Sales (P/S) Ratio**
        ```
        P/S = Stock Price / Sales per Share
        ```
        - **Use**: Useful for companies without earnings
        
        **Price-to-Book (P/B) Ratio**
        ```
        P/B = Stock Price / Book Value per Share
        ```
        - **Interpretation**: How much investors pay for company's net assets
        
        ### Profitability Ratios
        
        **Return on Equity (ROE)**
        ```
        ROE = Net Income / Shareholder's Equity
        ```
        - **Interpretation**: How efficiently company uses shareholder capital
        
        **Return on Assets (ROA)**
        ```
        ROA = Net Income / Total Assets
        ```
        - **Interpretation**: How efficiently company uses all assets
        
        **Return on Invested Capital (ROIC)**
        ```
        ROIC = NOPAT / Invested Capital
        ```
        - **Interpretation**: Return on capital invested in business
        
        ### Financial Health Ratios
        
        **Debt-to-Equity Ratio**
        ```
        D/E = Total Debt / Shareholder's Equity
        ```
        - **Interpretation**: Company's financial leverage
        
        **Current Ratio**
        ```
        Current Ratio = Current Assets / Current Liabilities
        ```
        - **Interpretation**: Ability to pay short-term obligations
        
        **Quick Ratio**
        ```
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        ```
        - **Interpretation**: Ability to pay short-term obligations without selling inventory
        
        ### Efficiency Ratios
        
        **Asset Turnover**
        ```
        Asset Turnover = Revenue / Total Assets
        ```
        - **Interpretation**: How efficiently company uses assets to generate revenue
        
        **Inventory Turnover**
        ```
        Inventory Turnover = COGS / Average Inventory
        ```
        - **Interpretation**: How quickly company sells inventory
        
        ## Valuation Methods
        
        ### Discounted Cash Flow (DCF) Analysis
        
        **Concept**: Company's value equals present value of future cash flows
        
        **Steps**:
        1. Project future cash flows (usually 5-10 years)
        2. Determine discount rate (WACC)
        3. Calculate terminal value
        4. Discount all cash flows to present value
        
        **Formula**:
        ```
        PV = CF1/(1+r)¹ + CF2/(1+r)² + ... + CFn/(1+r)ⁿ + TV/(1+r)ⁿ
        ```
        
        ### Dividend Discount Model (DDM)
        
        **Concept**: Stock value equals present value of future dividends
        
        **Gordon Growth Model**:
        ```
        Stock Value = D1 / (r - g)
        ```
        Where:
        - D1 = Expected dividend next year
        - r = Required rate of return
        - g = Dividend growth rate
        
        ### Comparable Company Analysis
        
        **Concept**: Compare company to similar public companies
        
        **Steps**:
        1. Identify comparable companies
        2. Calculate valuation multiples (P/E, P/S, EV/EBITDA)
        3. Apply multiples to target company
        4. Adjust for differences
        
        ## Economic Analysis
        
        ### Macroeconomic Factors
        
        **GDP Growth**: Economic growth affects corporate earnings
        **Interest Rates**: Affect borrowing costs and discount rates
        **Inflation**: Affects costs and pricing power
        **Employment**: Affects consumer spending
        
        ### Industry Analysis
        
        **Porter's Five Forces**:
        1. **Threat of New Entrants**: Barriers to entry
        2. **Bargaining Power of Buyers**: Customer concentration
        3. **Bargaining Power of Suppliers**: Supplier concentration
        4. **Threat of Substitutes**: Alternative products/services
        5. **Competitive Rivalry**: Industry competition
        
        ### Competitive Advantage
        
        **Types**:
        - **Cost Advantage**: Lower production costs
        - **Differentiation**: Unique products/services
        - **Network Effects**: Value increases with more users
        - **Switching Costs**: Cost for customers to switch
        
        ## Quality Indicators
        
        ### Earnings Quality
        - **Consistency**: Stable earnings growth
        - **Cash Conversion**: Earnings convert to cash
        - **Accounting Practices**: Conservative accounting
        
        ### Management Quality
        - **Experience**: Track record of success
        - **Capital Allocation**: Smart investment decisions
        - **Transparency**: Clear communication
        
        ## Limitations of Fundamental Analysis
        
        1. **Time-consuming**: Requires extensive research
        2. **Subjective**: Different analysts can reach different conclusions
        3. **Future uncertainty**: Projections may not materialize
        4. **Market irrationality**: Prices can remain irrational for long periods
        
        ## Best Practices
        
        1. **Comprehensive analysis**: Use multiple valuation methods
        2. **Margin of safety**: Buy below estimated intrinsic value
        3. **Long-term perspective**: Focus on business fundamentals
        4. **Continuous monitoring**: Regularly update analysis
        5. **Risk management**: Diversify across different companies/sectors
        """
    
    def _get_risk_management_content(self) -> str:
        """Get risk management lesson content"""
        return """
        # Risk Management Principles
        
        ## Understanding Risk in Trading
        
        Risk is the possibility of losing money or not achieving expected returns. In trading, risk cannot be eliminated, but it can be managed effectively.
        
        ## Types of Risk
        
        ### Market Risk
        Risk from market-wide factors affecting all securities.
        
        **Sources**:
        - Economic recessions
        - Interest rate changes
        - Political events
        - Natural disasters
        
        **Management**: Diversification, hedging, asset allocation
        
        ### Specific Risk
        Risk specific to individual securities or companies.
        
        **Sources**:
        - Company earnings disappointments
        - Management changes
        - Product failures
        - Legal issues
        
        **Management**: Diversification, research, stop losses
        
        ### Liquidity Risk
        Risk of not being able to sell at desired price.
        
        **Causes**:
        - Low trading volume
        - Market panic
        - Large position sizes
        
        **Management**: Position sizing, limit orders, trading liquid assets
        
        ### Leverage Risk
        Risk from using borrowed money to trade.
        
        **Dangers**:
        - Magnified losses
        - Margin calls
        - Forced liquidation
        
        **Management**: Conservative leverage, monitoring margin levels
        
        ## Position Sizing
        
        ### Fixed Fractional Position Sizing
        
        Risk a fixed percentage of capital per trade.
        
        ```
        Position Size = (Account Value × Risk Percentage) / Risk per Share
        ```
        
        **Example**:
        ```
        Account Value: $10,000
        Risk per Trade: 2% ($200)
        Risk per Share: $2 (difference between entry and stop loss)
        Position Size: $200 / $2 = 100 shares
        ```
        
        ### Kelly Criterion
        
        Mathematical formula for optimal position sizing.
        
        ```
        Kelly % = W - [(1 - W) / R]
        ```
        Where:
        - W = Win rate (probability of winning)
        - R = Win/Loss ratio (average win / average loss)
        
        **Example**:
        ```
        Win Rate: 60% (0.60)
        Win/Loss Ratio: 2.0
        Kelly % = 0.60 - [(1 - 0.60) / 2.0] = 0.40 or 40%
        ```
        
        **Important**: Many traders use half-Kelly for safety (20% in example)
        
        ### Volatility-Based Position Sizing
        
        Adjust position size based on market volatility.
        
        ```
        Position Size = Base Position Size × (Target Volatility / Current Volatility)
        ```
        
        ## Stop Loss Strategies
        
        ### Percentage Stop Loss
        
        Set stop loss at fixed percentage below entry.
        
        **Example**: 5% stop loss on $100 stock = stop at $95
        
        **Pros**: Simple, easy to implement
        **Cons**: Doesn't account for volatility or support levels
        
        ### Volatility Stop Loss
        
        Set stop based on market volatility (ATR).
        
        ```
        Stop Loss = Entry Price - (ATR × Multiplier)
        ```
        
        **Example**: ATR = $2, Multiplier = 2, Entry = $100
        Stop = $100 - ($2 × 2) = $96
        
        **Pros**: Adapts to market conditions
        **Cons**: May be too tight in high volatility
        
        ### Technical Stop Loss
        
        Place stop below technical support levels.
        
        **Levels to consider**:
        - Previous swing lows
        - Moving averages
        - Trend lines
        - Fibonacci levels
        
        **Pros**: Based on market structure
        **Cons**: Subjective, requires technical analysis skills
        
        ### Time Stop Loss
        
        Exit trade if it doesn't reach target within specified time.
        
        **Use**: Prevents capital from being tied up in non-performing trades
        
        ## Portfolio Risk Management
        
        ### Diversification
        
        **Asset Class Diversification**:
        - Stocks
        - Bonds
        - Real estate
        - Commodities
        - Currencies
        
        **Sector Diversification**:
        - Technology
        - Healthcare
        - Finance
        - Consumer goods
        - Industrial
        
        **Geographic Diversification**:
        - Domestic markets
        - International markets
        - Emerging markets
        
        ### Correlation Analysis
        
        Measure how different assets move together.
        
        **Correlation Coefficient**:
        - +1: Perfect positive correlation
        - 0: No correlation
        - -1: Perfect negative correlation
        
        **Goal**: Combine assets with low or negative correlation
        
        ### Portfolio Allocation
        
        **Modern Portfolio Theory**:
        - Optimize risk-return tradeoff
        - Efficient frontier
        - Minimum variance portfolio
        
        **Risk Parity**:
        - Equal risk contribution from all assets
        - Balance portfolio based on risk, not capital
        
        ## Risk Metrics
        
        ### Value at Risk (VaR)
        
        Maximum expected loss over specified time period at confidence level.
        
        ```
        VaR = Portfolio Value × Z-Score × Standard Deviation
        ```
        
        **Example**: $100,000 portfolio, 95% confidence, 1-day VaR
        If VaR = $2,000, there's 5% chance of losing more than $2,000 in one day
        
        ### Conditional Value at Risk (CVaR)
        
        Expected loss beyond VaR (also called Expected Shortfall).
        
        **Use**: Understand tail risk and worst-case scenarios
        
        ### Maximum Drawdown
        
        Maximum peak-to-trough decline in portfolio value.
        
        ```
        Drawdown = (Peak Value - Trough Value) / Peak Value
        ```
        
        **Importance**: Measures worst historical loss
        
        ### Sharpe Ratio
        
        Risk-adjusted return measure.
        
        ```
        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation
        ```
        
        **Interpretation**: Higher is better - more return per unit of risk
        
        ## Risk Management Rules
        
        ### The 2% Rule
        
        Never risk more than 2% of account on single trade.
        
        **Example**: $10,000 account = maximum $200 risk per trade
        
        ### The 6% Rule
        
        Never risk more than 6% of account in total at any time.
        
        **Example**: $10,000 account = maximum $600 total risk
        
        ### The 1% Rule (Conservative)
        
        Never risk more than 1% of account on single trade.
        
        **Use**: For beginners or high-volatility strategies
        
        ## Trading Psychology and Risk
        
        ### Fear and Greed
        
        **Fear**: Causes premature exits, missed opportunities
        **Greed**: Causes excessive risk-taking, overtrading
        
        **Management**: Stick to rules, use mechanical systems
        
        ### Loss Aversion
        
        Pain of losing is stronger than pleasure of winning.
        
        **Effect**: Hold losers too long, sell winners too early
        
        **Management**: Pre-defined exit rules, regular review
        
        ### Overconfidence Bias
        
        Overestimating ability and underestimating risk.
        
        **Effect**: Taking excessive risks, ignoring warning signs
        
        **Management**: Keep trading journal, seek feedback
        
        ## Risk Management Tools
        
        ### Trading Journal
        
        Track all trades with:
        - Entry and exit prices
        - Position size
        - Stop loss and target
        - Reason for trade
        - Outcome and lessons learned
        
        ### Risk Calculator
        
        Automated tools to calculate:
        - Position size
        - Stop loss levels
        - Portfolio risk
        - Risk/reward ratios
        
        ### Alert Systems
        
        Set up alerts for:
        - Price levels
        - Portfolio risk limits
        - Market conditions
        
        ## Emergency Procedures
        
        ### Market Crashes
        
        **Actions**:
        1. Assess portfolio risk
        2. Reduce position sizes
        3. Consider hedging strategies
        4. Avoid panic selling
        
        ### Margin Calls
        
        **Prevention**:
        - Monitor margin levels
        - Maintain cash buffer
        - Use conservative leverage
        
        **Response**:
        - Add more capital
        - Reduce positions
        - Contact broker immediately
        
        ### System Failures
        
        **Preparation**:
        - Have backup internet connection
        - Save broker contact information
        - Know manual order procedures
        
        ## Best Practices
        
        1. **Plan every trade**: Know entry, exit, and position size before trading
        2. **Use stop losses**: Always have predefined exit points
        3. **Diversify**: Don't put all eggs in one basket
        4. **Monitor risk**: Regularly check portfolio metrics
        5. **Learn from mistakes**: Review losing trades
        6. **Stay disciplined**: Stick to your risk management rules
        7. **Keep learning**: Markets change, so should your approach
        """
    
    def _get_market_psychology_content(self) -> str:
        """Get market psychology lesson content"""
        return """
        # Market Psychology and Behavioral Finance
        
        ## Introduction to Behavioral Finance
        
        Behavioral finance combines psychology and economics to explain why investors often make irrational decisions. It challenges the traditional assumption that markets are always efficient and investors are always rational.
        
        ## Cognitive Biases in Trading
        
        ### Confirmation Bias
        
        **Definition**: Tendency to seek information that confirms existing beliefs while ignoring contradictory evidence.
        
        **In Trading**:
        - Looking for news that supports your position
        - Ignoring warning signs about a losing trade
        - Following analysts who agree with your view
        
        **Impact**: Can lead to holding losing positions too long
        
        **Management**:
        - Actively seek opposing views
        - Use objective criteria for decisions
        - Keep trading journal to track biases
        
        ### Overconfidence Bias
        
        **Definition**: Overestimating one's abilities and underestimating risks.
        
        **Symptoms**:
        - Believing you can predict market movements
        - Taking excessive risks
        - Trading too frequently
        
        **Impact**: Poor risk management and large losses
        
        **Management**:
        - Track actual performance vs. perceived performance
        - Use mechanical systems
        - Set strict risk limits
        
        ### Anchoring Bias
        
        **Definition**: Relying too heavily on the first piece of information encountered.
        
        **Examples**:
        - Anchoring to purchase price when deciding to sell
        - Focusing on historical highs/lows
        - Sticking to initial price targets despite new information
        
        **Impact**: Poor entry/exit decisions
        
        **Management**:
        - Regularly reassess positions based on current information
        - Use multiple reference points
        - Focus on future prospects, not past prices
        
        ### Loss Aversion
        
        **Definition**: Pain of losing is psychologically twice as powerful as pleasure of winning.
        
        **Behaviors**:
        - Holding losing trades too long hoping to break even
        - Selling winning trades too early to lock in profits
        - Taking excessive risks to recover losses
        
        **Impact**: Poor risk/reward decisions
        
        **Management**:
        - Pre-defined exit rules
        - Focus on process, not individual outcomes
        - Regular portfolio review
        
        ### Sunk Cost Fallacy
        
        **Definition**: Continuing a behavior because of previously invested resources.
        
        **In Trading**:
        - Adding to losing positions because "I've already lost so much"
        - Refusing to admit a mistake
        - Averaging down without proper analysis
        
        **Impact**: Compounding losses
        
        **Management**:
        - Evaluate each decision independently
        - Set maximum loss limits
        - Regular portfolio rebalancing
        
        ### Herd Mentality
        
        **Definition**: Tendency to follow the crowd without independent analysis.
        
        **Manifestations**:
        - Buying stocks because everyone else is
        - Panic selling during market crashes
        - Following popular "gurus" blindly
        
        **Impact**: Buying high, selling low
        
        **Management**:
        - Develop independent analysis skills
        - Understand crowd psychology
        - Contrarian thinking when appropriate
        
        ### Recency Bias
        
        **Definition**: Giving more weight to recent events than historical patterns.
        
        **Examples**:
        - Assuming recent trends will continue indefinitely
        - Overreacting to recent news
        - Ignoring long-term historical patterns
        
        **Impact**: Poor market timing decisions
        
        **Management**:
        - Study historical market patterns
        - Use long-term data for analysis
        - Maintain perspective during extreme market conditions
        
        ### Gambler's Fallacy
        
        **Definition**: Believing that past random events affect future random events.
        
        **Examples**:
        - "I've had 5 losing trades, so I'm due for a winner"
        - "This stock has gone up 10 days in a row, it must go down soon"
        
        **Impact**: Poor position sizing and timing
        
        **Management**:
        - Understand that each trade is independent
        - Focus on probabilities, not patterns in random events
        - Use statistical analysis
        
        ## Emotional States in Trading
        
        ### Fear
        
        **Types**:
        - Fear of missing out (FOMO)
        - Fear of losing money
        - Fear of being wrong
        
        **Effects**:
        - Paralysis and missed opportunities
        - Panic selling at bottoms
        - Taking profits too early
        
        **Management**:
        - Pre-defined trading plan
        - Start with small positions
        - Focus on process, not outcomes
        
        ### Greed
        
        **Manifestations**:
        - Taking excessive risks
        - Overtrading
        - Ignoring risk management
        
        **Effects**:
        - Large losses
        - Poor decision making
        - Destroyed discipline
        
        **Management**:
        - Strict risk rules
        - Regular portfolio review
        - Set realistic expectations
        
        ### Hope
        
        **Danger of Hope**:
        - Holding losing positions hoping they'll recover
        - Ignoring objective analysis
        - Emotional decision making
        
        **Management**:
        - Replace hope with analysis
        - Set clear exit criteria
        - Accept losses quickly
        
        ### Regret
        
        **Types**:
        - Regret of commission (actions taken)
        - Regret of omission (opportunities missed)
        
        **Impact**:
        - Overtrading to "make up" for mistakes
        - Avoiding necessary risks
        
        **Management**:
        - Focus on learning from mistakes
        - Accept that losses are part of trading
        - Maintain long-term perspective
        
        ## Market Sentiment Analysis
        
        ### Fear and Greed Index
        
        **Components**:
        - Market momentum
        - Stock price breadth
        - Put/call ratio
        - Junk bond demand
        - Market volatility
        - Safe haven demand
        
        **Interpretation**:
        - High readings: Fear (potential buying opportunity)
        - Low readings: Greed (potential warning sign)
        
        ### Volatility Index (VIX)
        
        **Definition**: Measures market's expectation of 30-day volatility
        
        **Interpretation**:
        - High VIX: Fear and uncertainty
        - Low VIX: Complacency and confidence
        
        **Usage**:
        - Contrarian indicator
        - Risk management tool
        
        ### Put/Call Ratio
        
        **Definition**: Ratio of put options to call options traded
        
        **Interpretation**:
        - High ratio: Fear (more puts than calls)
        - Low ratio: Greed/complacency
        
        **Usage**:
        - Sentiment gauge
        - Contrarian signals
        
        ## Crowd Psychology Patterns
        
        ### Bubbles and Crashes
        
        **Bubble Characteristics**:
        - Rapid price appreciation
        - Widespread public participation
        - "This time is different" mentality
        - High leverage and speculation
        
        **Crash Patterns**:
        - Panic selling
        - Liquidity drying up
        - Margin calls
        - Capitulation
        
        **Trading Implications**:
        - Identify bubble conditions early
        - Use position sizing and risk management
        - Consider contrarian positions at extremes
        
        ### Market Cycles and Emotions
        
        **Optimism Phase**:
        - Rising prices
        - Increasing confidence
        - Media attention
        
        **Excitement Phase**:
        - Accelerating gains
        - Public participation
        - Euphoria
        
        **Anxiety Phase**:
        - Prices stall
        - Early warning signs
        - Some profit taking
        
        **Denial Phase**:
        - Prices decline
        - "Buy the dip" mentality
        - Ignoring warning signs
        
        **Fear Phase**:
        - Accelerating declines
        - Margin calls
        - Panic selling
        
        **Capitulation Phase**:
        - Maximum pessimism
        - Volume spike
        - Smart money buying
        
        **Hope Phase**:
        - Prices stabilize
        - Bargain hunting
        - Early recovery signs
        
        ## Behavioral Trading Strategies
        
        ### Contrarian Approach
        
        **Logic**: Go against crowd psychology at extremes
        
        **Implementation**:
        - Identify extreme sentiment readings
        - Use technical confirmation
        - Scale into positions gradually
        
        **Examples**:
        - Buy when Fear & Greed Index shows extreme fear
            - Sell when index shows extreme greed
        
        ### Momentum with Behavioral Overlay
        
        **Strategy**: Follow trends but be aware of behavioral extremes
        
        **Rules**:
        - Trade with momentum in normal conditions
        - Reduce or reverse positions at behavioral extremes
            - Use sentiment indicators as filters
        
        ### Behavioral Arbitrage
        
        **Concept**: Exploit behavioral biases of other traders
        
        **Examples**:
        - Buy stocks with unjustified pessimism
        - Sell stocks with irrational exuberance
        - Trade around earnings surprises
        
        ## Overcoming Behavioral Biases
        
        ### Systematic Approach
        
        **Mechanical Trading Systems**:
        - Remove emotion from decisions
        - Backtest strategies
        - Follow rules consistently
        
        **Checklists**:
            - Pre-trade checklist
            - Risk management checklist
            - Post-trade review checklist
        
        ### Mindfulness and Self-Awareness
        
        **Practices**:
        - Meditation and stress reduction
        - Regular self-reflection
        - Emotional state monitoring
        
        **Benefits**:
        - Better decision making under pressure
        - Reduced emotional reactions
        - Improved discipline
        
        ### Education and Experience
        
        **Continuous Learning**:
        - Study behavioral finance
        - Learn from mistakes
        - Stay updated on research
        
        **Experience**:
        - Practice with small positions
        - Keep detailed trading journal
        - Review performance regularly
        
        ### Accountability and Support
        
        **Trading Coach or Mentor**:
        - Objective feedback
        - Accountability partner
        - Experience and guidance
        
        **Trading Community**:
        - Share experiences
        - Learn from others
        - Emotional support
        
        ## Practical Applications
        
        ### Pre-Trading Routine
        
        1. **Market Sentiment Check**: Review sentiment indicators
        2. **Emotional State Assessment**: Are you fearful, greedy, or neutral?
        3. **Bias Check**: Are you confirming existing beliefs?
        4. **Risk Assessment**: Is position size appropriate?
        
        ### During Trading
        
        1. **Monitor Emotional State**: Regular check-ins
        2. **Follow Plan**: Stick to pre-defined rules
        3. **Avoid Impulse Decisions**: Take breaks if emotional
        4. **Record Decisions**: Note reasoning for each trade
        
        ### Post-Trading Review
        
        1. **Outcome Analysis**: What worked, what didn't?
        2. **Bias Identification**: Were biases present?
        3. **Emotional Review**: How did emotions affect decisions?
        4. **Learning Integration**: Update trading plan
        
        ## Key Takeaways
        
        1. **Awareness is First Step**: Recognize your biases
        2. **Systems Beat Emotions**: Use mechanical approaches
        3. **Psychology Matters**: Understand market sentiment
        4. **Continuous Improvement**: Learn from every trade
        5. **Balance is Key**: Combine analysis with psychology
        
        Remember: Successful trading is 20% strategy and 80% psychology. Master your mind, and the profits will follow.
        """
    
    def _get_market_structure_quiz(self) -> List[Dict]:
        """Get market structure quiz questions"""
        return [
            {
                "question": "What is the bid-ask spread?",
                "options": [
                    "The difference between the highest bid and lowest ask price",
                    "The total volume of shares traded",
                    "The price range over a trading day",
                    "The commission charged by brokers"
                ],
                "correct": 0,
                "explanation": "The bid-ask spread represents the cost of immediate execution and the difference between what buyers are willing to pay and sellers are asking for."
            },
            {
                "question": "Who provides liquidity in financial markets?",
                "options": [
                    "Only retail traders",
                    "Market makers and high-frequency traders",
                    "Only institutional investors",
                    "Regulatory bodies"
                ],
                "correct": 1,
                "explanation": "Market makers and high-frequency traders provide liquidity by being willing to buy at the bid and sell at the ask, though other participants also contribute to overall market liquidity."
            },
            {
                "question": "What does market efficiency mean?",
                "options": [
                    "Markets are always predictable",
                    "Prices reflect all available information",
                    "All traders make profits",
                    "There is no volatility in efficient markets"
                ],
                "correct": 1,
                "explanation": "Market efficiency suggests that prices quickly incorporate all available information, making it difficult to consistently achieve above-average returns."
            }
        ]
    
    def _get_order_types_quiz(self) -> List[Dict]:
        """Get order types quiz questions"""
        return [
            {
                "question": "When should you use a market order?",
                "options": [
                    "When price control is more important than speed",
                    "When execution speed is more important than price",
                    "When you want to specify exact price",
                    "When trading illiquid stocks"
                ],
                "correct": 1,
                "explanation": "Market orders guarantee execution but not price, making them suitable when speed is prioritized over price control."
            },
            {
                "question": "What is the main purpose of a stop-loss order?",
                "options": [
                    "To lock in profits",
                    "To limit potential losses",
                    "To buy at a specific price",
                    "To sell at the highest possible price"
                ],
                "correct": 1,
                "explanation": "Stop-loss orders are designed to automatically sell a position when it reaches a predetermined price, thereby limiting potential losses."
            },
            {
                "question": "What is slippage?",
                "options": [
                    "The commission charged by brokers",
                    "The difference between expected and actual execution price",
                    "The time delay in order execution",
                    "The spread between bid and ask prices"
                ],
                "correct": 1,
                "explanation": "Slippage occurs when the execution price differs from the expected price, often due to market movement between order placement and execution."
            }
        ]
    
    def _get_technical_analysis_quiz(self) -> List[Dict]:
        """Get technical analysis quiz questions"""
        return [
            {
                "question": "What does RSI measure?",
                "options": [
                    "Price volatility",
                    "Trading volume",
                    "Price momentum and overbought/oversold conditions",
                    "Market trend direction"
                ],
                "correct": 2,
                "explanation": "RSI (Relative Strength Index) measures the speed and change of price movements, indicating overbought conditions above 70 and oversold conditions below 30."
            },
            {
                "question": "What is a support level?",
                "options": [
                    "A price level where selling pressure overcomes buying pressure",
                    "A price level where buying pressure overcomes selling pressure",
                    "The highest price reached in a period",
                    "The average price over a period"
                ],
                "correct": 1,
                "explanation": "Support is a price level where buying interest is strong enough to overcome selling pressure, potentially causing a price bounce."
            },
            {
                "question": "What do Bollinger Bands indicate?",
                "options": [
                    "Only price direction",
                    "Only trading volume",
                    "Volatility and potential overbought/oversold levels",
                    "Company fundamentals"
                ],
                "correct": 2,
                "explanation": "Bollinger Bands consist of a moving average with upper and lower bands based on standard deviation, expanding during high volatility and contracting during low volatility."
            }
        ]
    
    def _get_fundamental_analysis_quiz(self) -> List[Dict]:
        """Get fundamental analysis quiz questions"""
        return [
            {
                "question": "What does the P/E ratio measure?",
                "options": [
                    "Company's debt level",
                    "How much investors pay for $1 of earnings",
                    "Trading volume relative to price",
                    "Company's growth rate"
                ],
                "correct": 1,
                "explanation": "The Price-to-Earnings ratio shows how much investors are willing to pay for each dollar of company earnings, helping assess relative valuation."
            },
            {
                "question": "What is Return on Equity (ROE)?",
                "options": [
                    "Net income divided by total assets",
                    "Revenue divided by total assets",
                    "Net income divided by shareholder's equity",
                    "Total debt divided by total assets"
                ],
                "correct": 2,
                "explanation": "ROE measures how efficiently a company uses shareholder equity to generate profits, indicating profitability and capital efficiency."
            },
            {
                "question": "What is intrinsic value?",
                "options": [
                    "The current market price of a stock",
                    "The theoretical true value based on fundamentals",
                    "The highest price a stock has reached",
                    "The average price over the past year"
                ],
                "correct": 1,
                "explanation": "Intrinsic value is the estimated true value of a company based on its fundamentals, regardless of current market price."
            }
        ]
    
    def _get_risk_management_quiz(self) -> List[Dict]:
        """Get risk management quiz questions"""
        return [
            {
                "question": "What is the 2% rule in risk management?",
                "options": [
                    "Risk 2% of total account value per trade",
                    "Risk 2% of potential profit per trade",
                    "Invest only 2% of account in one stock",
                    "Set stop loss 2% below entry price"
                ],
                "correct": 0,
                "explanation": "The 2% rule limits risk on any single trade to 2% of total account value, helping prevent catastrophic losses."
            },
            {
                "question": "What is Value at Risk (VaR)?",
                "options": [
                    "The maximum possible loss",
                    "The expected loss at a confidence level over a time period",
                    "The average loss over time",
                    "The total risk in portfolio"
                ],
                "correct": 1,
                "explanation": "VaR estimates the maximum expected loss over a specified time period at a given confidence level, helping quantify downside risk."
            },
            {
                "question": "Why is diversification important?",
                "options": [
                    "It guarantees profits",
                    "It eliminates all risk",
                    "It reduces specific risk through low correlation",
                    "It increases returns dramatically"
                ],
                "correct": 2,
                "explanation": "Diversification reduces portfolio risk by combining assets with low correlation, so losses in one area may be offset by gains in others."
            }
        ]
    
    def _get_market_psychology_quiz(self) -> List[Dict]:
        """Get market psychology quiz questions"""
        return [
            {
                "question": "What is confirmation bias?",
                "options": [
                    "The tendency to follow the crowd",
                    "Seeking information that confirms existing beliefs",
                    "Fear of missing out on profits",
                    "Overconfidence in trading abilities"
                ],
                "correct": 1,
                "explanation": "Confirmation bias causes traders to seek information that supports their existing beliefs while ignoring contradictory evidence."
            },
            {
                "question": "What is loss aversion?",
                "options": [
                    "Fear of taking any risks",
                    "The pain of losing being stronger than pleasure of winning",
                    "Preference for guaranteed small gains",
                    "Avoiding all losing trades"
                ],
                "correct": 1,
                "explanation": "Loss aversion is the psychological tendency where the pain of losing is about twice as powerful as the pleasure of winning."
            },
            {
                "question": "What does a high VIX reading typically indicate?",
                "options": [
                    "Market complacency and confidence",
                    "Fear and uncertainty in the market",
                    "Strong bull market conditions",
                    "Low volatility expectations"
                ],
                "correct": 1,
                "explanation": "The VIX (Volatility Index) often rises during periods of fear and uncertainty, making it a useful fear gauge for market sentiment."
            }
        ]
    
    def _get_market_structure_examples(self) -> List[Dict]:
        """Get market structure examples"""
        return [
            {
                "title": "Order Book Analysis",
                "description": "Understanding liquidity and price impact",
                "data": {
                    "symbol": "AAPL",
                    "bid": 150.00,
                    "ask": 150.02,
                    "bid_size": 1000,
                    "ask_size": 800,
                    "spread": 0.02,
                    "spread_percentage": 0.013
                }
            },
            {
                "title": "Market Impact Simulation",
                "description": "How large orders affect price",
                "data": {
                    "order_size": 10000,
                    "avg_daily_volume": 50000000,
                    "estimated_impact": 0.05,
                    "total_cost": 150.05
                }
            }
        ]
    
    def _get_order_types_examples(self) -> List[Dict]:
        """Get order types examples"""
        return [
            {
                "title": "Market Order Execution",
                "description": "Immediate execution example",
                "data": {
                    "order_type": "Market",
                    "symbol": "TSLA",
                    "quantity": 100,
                    "expected_price": 250.00,
                    "actual_price": 250.15,
                    "slippage": 0.15,
                    "total_cost": 25015.00
                }
            },
            {
                "title": "Limit Order Execution",
                "description": "Price control example",
                "data": {
                    "order_type": "Limit",
                    "symbol": "MSFT",
                    "quantity": 50,
                    "limit_price": 300.00,
                    "execution_price": 299.85,
                    "savings": 0.15,
                    "total_cost": 14992.50
                }
            }
        ]
    
    def _get_technical_analysis_examples(self) -> List[Dict]:
        """Get technical analysis examples"""
        return [
            {
                "title": "RSI Overbought Signal",
                "description": "Identifying potential reversal points",
                "data": {
                    "symbol": "NVDA",
                    "current_price": 450.00,
                    "rsi": 75.5,
                    "signal": "Overbought",
                    "suggested_action": "Consider taking profits or waiting for pullback"
                }
            },
            {
                "title": "Moving Average Crossover",
                "description": "Trend following signal",
                "data": {
                    "symbol": "AMD",
                    "current_price": 120.00,
                    "sma_50": 115.00,
                    "sma_200": 110.00,
                    "signal": "Bullish crossover",
                    "suggested_action": "Consider long position"
                }
            }
        ]
    
    def _get_fundamental_analysis_examples(self) -> List[Dict]:
        """Get fundamental analysis examples"""
        return [
            {
                "title": "P/E Ratio Comparison",
                "description": "Relative valuation analysis",
                "data": {
                    "company": "Apple",
                    "current_pe": 28.5,
                    "industry_average": 25.0,
                    "historical_average": 22.0,
                    "analysis": "Slightly overvalued compared to industry and history"
                }
            },
            {
                "title": "ROE Analysis",
                "description": "Profitability assessment",
                "data": {
                    "company": "Microsoft",
                    "roe": 35.2,
                    "industry_average": 20.0,
                    "trend": "Improving",
                    "analysis": "Excellent profitability and capital efficiency"
                }
            }
        ]
    
    def _get_risk_management_examples(self) -> List[Dict]:
        """Get risk management examples"""
        return [
            {
                "title": "Position Sizing Calculation",
                "description": "Applying the 2% rule",
                "data": {
                    "account_value": 10000,
                    "risk_percentage": 2,
                    "max_risk": 200,
                    "stop_distance": 2.50,
                    "position_size": 80,
                    "share_price": 50.00
                }
            },
            {
                "title": "Portfolio Diversification",
                "description": "Correlation analysis",
                "data": {
                    "stocks": ["AAPL", "MSFT", "GOOGL"],
                    "correlation_matrix": [[1.0, 0.7, 0.6], [0.7, 1.0, 0.8], [0.6, 0.8, 1.0]],
                    "analysis": "High correlation - consider adding uncorrelated assets"
                }
            }
        ]
    
    def _get_market_psychology_examples(self) -> List[Dict]:
        """Get market psychology examples"""
        return [
            {
                "title": "Fear and Greed Index Analysis",
                "description": "Market sentiment assessment",
                "data": {
                    "current_reading": 18,
                    "signal": "Extreme Fear",
                    "historical_context": "Often marks buying opportunities",
                    "recommended_action": "Consider contrarian positions with proper risk management"
                }
            },
            {
                "title": "Confirmation Bias Example",
                "description": "Identifying bias in decision making",
                "data": {
                    "scenario": "Holding losing position hoping for recovery",
                    "bias": "Confirmation bias + loss aversion",
                    "correct_action": "Objectively assess fundamentals and exit if thesis is broken"
                }
            }
        ]
    
    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Get a specific lesson by ID"""
        return self.lessons.get(lesson_id)
    
    def get_all_lessons(self) -> List[Lesson]:
        """Get all available lessons"""
        return list(self.lessons.values())
    
    def get_lessons_by_difficulty(self, difficulty: str) -> List[Lesson]:
        """Get lessons filtered by difficulty level"""
        return [lesson for lesson in self.lessons.values() if lesson.difficulty == difficulty]
    
    def calculate_progress(self, user_id: str) -> Dict[str, Any]:
        """Calculate user progress through lessons"""
        completed_lessons = self.user_progress.get(user_id, {}).get("completed", [])
        total_lessons = len(self.lessons)
        completed_count = len(completed_lessons)
        
        return {
            "total_lessons": total_lessons,
            "completed_lessons": completed_count,
            "progress_percentage": (completed_count / total_lessons) * 100 if total_lessons > 0 else 0,
            "next_lesson": self._get_next_lesson(completed_lessons)
        }
    
    def _get_next_lesson(self, completed_lessons: List[str]) -> Optional[str]:
        """Get the next lesson for user to complete"""
        lesson_order = [
            "market_structure",
            "order_types", 
            "risk_management",
            "technical_analysis",
            "fundamental_analysis",
            "market_psychology"
        ]
        
        for lesson_id in lesson_order:
            if lesson_id not in completed_lessons:
                return lesson_id
        
        return None
    
    def generate_certificate(self, user_id: str) -> Dict[str, Any]:
        """Generate completion certificate"""
        progress = self.calculate_progress(user_id)
        
        if progress["progress_percentage"] == 100:
            return {
                "certificate_id": f"TRADING_BASICS_{user_id}_{datetime.now().strftime('%Y%m%d')}",
                "completion_date": datetime.now().isoformat(),
                "user_id": user_id,
                "course": "Trading Fundamentals",
                "lessons_completed": progress["completed_lessons"],
                "achievement": "Master of Trading Fundamentals"
            }
        
        return {"error": "Course not completed yet"}
    
    def create_interactive_chart(self, lesson_id: str, chart_type: str) -> Dict[str, Any]:
        """Create interactive charts for lessons"""
        
        if lesson_id == "technical_analysis" and chart_type == "candlestick":
            # Generate sample candlestick data
            dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
            np.random.seed(42)
            
            prices = 100 + np.cumsum(np.random.randn(50) * 2)
            high = prices + np.random.uniform(0, 3, 50)
            low = prices - np.random.uniform(0, 3, 50)
            open_prices = prices + np.random.uniform(-2, 2, 50)
            
            fig = go.Figure(data=go.Candlestick(
                x=dates,
                open=open_prices,
                high=high,
                low=low,
                close=prices,
                name='Price'
            ))
            
            fig.update_layout(
                title='Sample Candlestick Chart',
                yaxis_title='Price',
                xaxis_title='Date',
                template='plotly_white'
            )
            
            return {
                "chart_html": fig.to_html(),
                "chart_json": fig.to_json()
            }
        
        elif lesson_id == "risk_management" and chart_type == "drawdown":
            # Generate sample drawdown data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            np.random.seed(42)
            
            returns = np.random.normal(0.001, 0.02, 100)
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'Drawdown'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=cumulative_returns, name='Portfolio Value'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=drawdown, name='Drawdown', fill='tonexty'),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Portfolio Drawdown Analysis',
                template='plotly_white',
                height=600
            )
            
            return {
                "chart_html": fig.to_html(),
                "chart_json": fig.to_json()
            }
        
        return {"error": "Chart not available"}
    
    def export_lesson_content(self, lesson_id: str, format: str = "json") -> str:
        """Export lesson content in different formats"""
        lesson = self.get_lesson(lesson_id)
        if not lesson:
            return "Lesson not found"
        
        if format == "json":
            return json.dumps({
                "id": lesson.id,
                "title": lesson.title,
                "content": lesson.content,
                "difficulty": lesson.difficulty,
                "duration": lesson.duration_minutes,
                "quiz_questions": lesson.quiz_questions,
                "examples": lesson.examples
            }, indent=2)
        
        elif format == "markdown":
            return f"""# {lesson.title}

**Difficulty:** {lesson.difficulty}  
**Duration:** {lesson.duration_minutes} minutes

{lesson.content}

## Quiz Questions
{self._format_quiz_for_markdown(lesson.quiz_questions)}

## Examples
{self._format_examples_for_markdown(lesson.examples)}
"""
        
        return "Format not supported"
    
    def _format_quiz_for_markdown(self, questions: List[Dict]) -> str:
        """Format quiz questions for markdown"""
        formatted = ""
        for i, q in enumerate(questions, 1):
            formatted += f"\n### Question {i}\n"
            formatted += f"{q['question']}\n\n"
            for j, option in enumerate(q['options']):
                marker = "✓" if j == q['correct'] else "○"
                formatted += f"{marker} {option}\n"
            formatted += f"\n**Explanation:** {q['explanation']}\n"
        return formatted
    
    def _format_examples_for_markdown(self, examples: List[Dict]) -> str:
        """Format examples for markdown"""
        formatted = ""
        for example in examples:
            formatted += f"\n### {example['title']}\n"
            formatted += f"{example['description']}\n\n"
            formatted += f"```json\n{json.dumps(example['data'], indent=2)}\n```\n"
        return formatted


# Factory function for easy initialization
def create_trading_basics_module() -> TradingBasicsModule:
    """Create and return a TradingBasicsModule instance"""
    return TradingBasicsModule()


# Example usage
if __name__ == "__main__":
    module = create_trading_basics_module()
    
    # Get all lessons
    lessons = module.get_all_lessons()
    print(f"Available lessons: {len(lessons)}")
    
    # Get a specific lesson
    market_structure_lesson = module.get_lesson("market_structure")
    if market_structure_lesson:
        print(f"Lesson: {market_structure_lesson.title}")
        print(f"Duration: {market_structure_lesson.duration_minutes} minutes")
    
    # Create interactive chart
    chart = module.create_interactive_chart("technical_analysis", "candlestick")
    if "chart_html" in chart:
        print("Chart created successfully")
    
    # Export lesson content
    content = module.export_lesson_content("market_structure", "markdown")
    print("Content exported successfully")