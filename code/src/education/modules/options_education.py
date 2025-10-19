"""
Options Trading Education Module - Comprehensive Learning System

This module provides in-depth education about options trading, including
Greeks, strategies, pricing models, and risk management.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from scipy.stats import norm


@dataclass
class OptionStrategy:
    """Structure for option strategies"""
    name: str
    description: str
    market_outlook: str
    risk_level: str
    max_profit: str
    max_loss: str
    breakeven_points: List[str]
    greeks_exposure: Dict[str, str]
    examples: List[Dict]


class OptionsEducationModule:
    """
    Comprehensive options trading education module
    """
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.greeks_calculator = GreeksCalculator()
        self.pricing_models = OptionsPricing()
        
    def _initialize_strategies(self) -> Dict[str, OptionStrategy]:
        """Initialize all options strategies"""
        
        strategies = {
            "long_call": OptionStrategy(
                name="Long Call",
                description="Buy a call option to profit from upward price movement",
                market_outlook="Bullish",
                risk_level="Limited",
                max_profit="Unlimited",
                max_loss="Premium paid",
                breakeven_points=["Strike price + premium"],
                greeks_exposure={
                    "Delta": "Positive (0 to 1)",
                    "Gamma": "Positive",
                    "Theta": "Negative (time decay)",
                    "Vega": "Positive (volatility helps)"
                },
                examples=self._get_long_call_examples()
            ),
            
            "long_put": OptionStrategy(
                name="Long Put",
                description="Buy a put option to profit from downward price movement",
                market_outlook="Bearish",
                risk_level="Limited",
                max_profit="Substantial (strike - premium)",
                max_loss="Premium paid",
                breakeven_points=["Strike price - premium"],
                greeks_exposure={
                    "Delta": "Negative (-1 to 0)",
                    "Gamma": "Positive",
                    "Theta": "Negative (time decay)",
                    "Vega": "Positive (volatility helps)"
                },
                examples=self._get_long_put_examples()
            ),
            
            "covered_call": OptionStrategy(
                name="Covered Call",
                description="Own stock and sell call options against it",
                market_outlook="Neutral to slightly bullish",
                risk_level="Limited",
                max_profit="Strike price - stock cost + premium",
                max_loss="Substantial (stock can go to zero)",
                breakeven_points=["Stock purchase price - premium received"],
                greeks_exposure={
                    "Delta": "Slightly positive",
                    "Gamma": "Slightly negative",
                    "Theta": "Positive (time decay helps)",
                    "Vega": "Slightly negative"
                },
                examples=self._get_covered_call_examples()
            ),
            
            "protective_put": OptionStrategy(
                name="Protective Put",
                description="Own stock and buy put options as insurance",
                market_outlook="Bullish with protection",
                risk_level="Limited",
                max_profit="Unlimited (less premium paid)",
                max_loss="Stock price - strike + premium",
                breakeven_points=["Stock purchase price + premium"],
                greeks_exposure={
                    "Delta": "Positive but reduced",
                    "Gamma": "Slightly positive",
                    "Theta": "Negative (time decay cost)",
                    "Vega": "Slightly positive"
                },
                examples=self._get_protective_put_examples()
            ),
            
            "iron_condor": OptionStrategy(
                name="Iron Condor",
                description="Sell credit spread on both calls and puts",
                market_outlook="Neutral (range-bound)",
                risk_level="Limited",
                max_profit="Net premium received",
                max_loss="Width of spreads - premium received",
                breakeven_points=["Two breakeven points around current price"],
                greeks_exposure={
                    "Delta": "Near zero",
                    "Gamma": "Negative",
                    "Theta": "Positive (time decay helps)",
                    "Vega": "Negative (volatility hurts)"
                },
                examples=self._get_iron_condor_examples()
            ),
            
            "straddle": OptionStrategy(
                name="Long Straddle",
                description="Buy both call and put at same strike",
                market_outlook="Expecting big move (direction uncertain)",
                risk_level="Limited",
                max_profit="Unlimited",
                max_loss="Total premium paid",
                breakeven_points=["Strike ± total premium"],
                greeks_exposure={
                    "Delta": "Near zero at start",
                    "Gamma": "High positive",
                    "Theta": "Negative (high time decay)",
                    "Vega": "High positive (volatility helps)"
                },
                examples=self._get_straddle_examples()
            ),
            
            "strangle": OptionStrategy(
                name="Long Strangle",
                description="Buy out-of-the-money call and put",
                market_outlook="Expecting big move (direction uncertain)",
                risk_level="Limited",
                max_profit="Unlimited",
                max_loss="Total premium paid",
                breakeven_points=["Call strike + premium, Put strike - premium"],
                greeks_exposure={
                    "Delta": "Near zero at start",
                    "Gamma": "Positive",
                    "Theta": "Negative (time decay)",
                    "Vega": "Positive (volatility helps)"
                },
                examples=self._get_strangle_examples()
            ),
            
            "butterfly_spread": OptionStrategy(
                name="Butterfly Spread",
                description="Complex strategy with limited risk/reward",
                market_outlook="Neutral (expecting price to stay near center strike)",
                risk_level="Limited",
                max_profit="Width of wings - premium paid",
                max_loss="Premium paid",
                breakeven_points=["Two points around center strike"],
                greeks_exposure={
                    "Delta": "Near zero",
                    "Gamma": "Varies (positive near center)",
                    "Theta": "Positive near center",
                    "Vega": "Negative near center"
                },
                examples=self._get_butterfly_examples()
            ),
            
            "calendar_spread": OptionStrategy(
                name="Calendar Spread",
                description="Sell near-term option, buy longer-term option",
                market_outlook="Neutral (benefits from time decay)",
                risk_level="Limited",
                max_profit="Maximum at short strike at expiration",
                max_loss="Premium paid",
                breakeven_points=["Two points around strike"],
                greeks_exposure={
                    "Delta": "Varies",
                    "Gamma": "Negative",
                    "Theta": "Positive (time decay helps)",
                    "Vega": "Positive (volatility helps)"
                },
                examples=self._get_calendar_examples()
            )
        }
        
        return strategies
    
    def get_options_basics_content(self) -> str:
        """Get comprehensive options basics content"""
        return """
        # Options Trading Fundamentals
        
        ## What is an Option?
        
        An option is a contract that gives the buyer the right, but not the obligation, to buy or sell an underlying asset at a specified price on or before a specific date.
        
        ## Key Terminology
        
        ### Call Option
        Gives the holder the right to **BUY** the underlying asset at the strike price.
        
        ### Put Option
        Gives the holder the right to **SELL** the underlying asset at the strike price.
        
        ### Strike Price (Exercise Price)
        The price at which the option holder can buy or sell the underlying asset.
        
        ### Expiration Date
        The date on which the option contract expires.
        
        ### Premium
        The price paid for the option contract.
        
        ## Option Moneyness
        
        ### In the Money (ITM)
        - **Call**: Stock price > Strike price
        - **Put**: Stock price < Strike price
        - Has intrinsic value
        
        ### At the Money (ATM)
        - Stock price = Strike price
        - No intrinsic value
        
        ### Out of the Money (OTM)
        - **Call**: Stock price < Strike price
        - **Put**: Stock price > Strike price
        - No intrinsic value, only time value
        
        ## Option Value Components
        
        ### Intrinsic Value
        The immediate value if exercised today.
        
        **Call Intrinsic Value** = Max(0, Stock Price - Strike Price)
        **Put Intrinsic Value** = Max(0, Strike Price - Stock Price)
        
        ### Time Value (Extrinsic Value)
        The value beyond intrinsic value, based on:
        - Time to expiration
        - Volatility
        - Interest rates
        - Dividends
        
        **Total Premium** = Intrinsic Value + Time Value
        
        ## Factors Affecting Option Prices
        
        ### 1. Underlying Price (Delta)
        - Call options: Value increases as stock price increases
        - Put options: Value decreases as stock price increases
        
        ### 2. Strike Price
        - Lower strike calls are more valuable
        - Higher strike puts are more valuable
        
        ### 3. Time to Expiration (Theta)
        - More time = more value
        - Time decay accelerates near expiration
        
        ### 4. Volatility (Vega)
        - Higher volatility = higher option prices
        - Both calls and puts benefit from increased volatility
        
        ### 5. Interest Rates (Rho)
        - Higher rates increase call values
        - Higher rates decrease put values
        
        ### 6. Dividends
        - Higher dividends decrease call values
        - Higher dividends increase put values
        
        ## Option Styles
        
        ### American Options
        Can be exercised at any time before expiration.
        
        ### European Options
        Can only be exercised at expiration.
        
        ## Basic Option Strategies
        
        ### Long Call
        - **When to use**: Bullish outlook
        - **Max profit**: Unlimited
        - **Max loss**: Premium paid
        - **Breakeven**: Strike + premium
        
        ### Long Put
        - **When to use**: Bearish outlook
        - **Max profit**: Substantial
        - **Max loss**: Premium paid
        - **Breakeven**: Strike - premium
        
        ### Covered Call
        - **When to use**: Neutral to slightly bullish
        - **Own**: 100 shares of stock
        - **Sell**: 1 call option per 100 shares
        - **Income**: Premium received
        
        ### Protective Put
        - **When to use**: Own stock, want protection
        - **Own**: 100 shares of stock
        - **Buy**: 1 put option per 100 shares
        - **Insurance**: Limits downside risk
        
        ## Understanding Leverage
        
        Options provide leverage, allowing control of more shares with less capital.
        
        **Example**:
        - Stock: $100 per share
        - Call option (90 strike, 30 days): $5 premium
        - Control 100 shares for $500 vs $10,000 for stock
        
        **Benefits**:
        - Limited risk (premium paid)
        - High potential returns
        - Flexibility in strategies
        
        **Risks**:
        - Time decay works against you
        - Can lose entire premium
        - Requires understanding of Greeks
        
        ## Assignment and Exercise
        
        ### Exercise
        Option holder uses their right to buy/sell the underlying.
        
        ### Assignment
        Option seller is required to fulfill their obligation.
        
        ### Automatic Exercise
        Options in the money at expiration are typically automatically exercised.
        
        ## Margin Requirements
        
        Selling options requires maintaining margin account with sufficient collateral.
        
        **Naked calls**: Highest margin requirement
        **Cash-secured puts**: Cash equal to strike price
        **Spreads**: Reduced margin due to defined risk
        
        ## Trading Considerations
        
        ### Liquidity
        - High volume = tight spreads
        - Low volume = wide spreads, difficult execution
        
        ### Bid-Ask Spread
        - Cost of entering/exiting positions
        - More significant for cheaper options
        
        ### Implied Volatility
        - Market's expectation of future volatility
        - High IV = expensive options
        - Low IV = cheap options
        
        ## Risk Management
        
        ### Position Sizing
        - Limit options trading to small portion of portfolio
        - Consider total risk, not just premium
        
        ### Understanding Greeks
        - Monitor Delta, Gamma, Theta, Vega exposure
        - Adjust positions based on Greek changes
        
        ### Time Decay
        - Remember options are wasting assets
        - Plan exit strategies before time decay accelerates
        
        ## Common Mistakes to Avoid
        
        1. **Buying OTM options close to expiration**
        2. **Ignoring time decay (Theta)**
        3. **Over-leveraging positions**
        4. **Not understanding assignment risk**
        5. **Trading without understanding Greeks**
        6. **Holding losing positions too long**
        7. **Not having exit strategies**
        
        ## Getting Started
        
        1. **Education**: Understand basics thoroughly
        2. **Paper Trading**: Practice without real money
        3. **Start Small**: Begin with defined-risk strategies
        4. **Focus on Learning**: Priority is education, not profits
        5. **Keep Detailed Records**: Track all trades and lessons learned
        """
    
    def get_greeks_content(self) -> str:
        """Get comprehensive Greeks education content"""
        return """
        # The Options Greeks - Complete Guide
        
        ## Introduction to Greeks
        
        The Greeks are measures of risk sensitivity in options trading. They help traders understand how different factors affect option prices and manage their positions effectively.
        
        ## Delta (Δ) - The First Derivative
        
        ### Definition
        Delta measures the rate of change in an option's price relative to the change in the underlying asset's price.
        
        ### Formula
        ```
        Delta = Change in Option Price / Change in Underlying Price
        ```
        
        ### Values and Interpretation
        
        **Call Options**:
        - Range: 0 to 1
        - Deep ITM: ~0.8 to 1.0
        - ATM: ~0.5
        - Deep OTM: ~0 to 0.2
        
        **Put Options**:
        - Range: -1 to 0
        - Deep ITM: ~-0.8 to -1.0
        - ATM: ~-0.5
        - Deep OTM: ~-0.2 to 0
        
        ### Practical Applications
        
        **1. Directional Exposure**
        - Positive delta = bullish exposure
        - Negative delta = bearish exposure
        - Delta of 0 = neutral exposure
        
        **2. Equivalent Stock Position**
        Delta represents the equivalent number of shares:
        - Call with delta 0.5 = equivalent to 50 shares
        - Put with delta -0.3 = equivalent to short 30 shares
        
        **3. Probability of Finishing ITM**
        For ATM options, delta approximates the probability of finishing ITM.
        - Delta 0.3 = ~30% chance of finishing ITM
        
        ### Delta Hedging
        
        Creating a delta-neutral position:
        ```
        Shares needed = -Total Delta of Options / Delta per Share
        ```
        
        Example:
        - Long 10 call options with delta 0.5 each
        - Total delta = 10 × 100 × 0.5 = 500
        - Sell 500 shares to hedge
        
        ## Gamma (Γ) - The Second Derivative
        
        ### Definition
        Gamma measures the rate of change in delta relative to the change in the underlying asset's price.
        
        ### Formula
        ```
        Gamma = Change in Delta / Change in Underlying Price
        ```
        
        ### Characteristics
        
        **Highest Gamma**:
        - ATM options
        - Short time to expiration
        - Lower volatility
        
        **Lowest Gamma**:
        - Deep ITM/OTM options
        - Long time to expiration
        - High volatility
        
        ### Practical Applications
        
        **1. Delta Stability**
        - High gamma = delta changes rapidly
        - Low gamma = delta remains stable
        
        **2. Gamma Scalping**
        - Profit from large price movements
        - Buy options with high gamma before expected moves
        
        **3. Risk Management**
        - High gamma positions require frequent adjustment
        - Gamma risk is highest near expiration
        
        ## Theta (Θ) - Time Decay
        
        ### Definition
        Theta measures the rate of change in an option's price relative to the passage of time.
        
        ### Formula
        ```
        Theta = Change in Option Price / Change in Time
        ```
        
        ### Characteristics
        
        **Always Negative for Long Options**:
        - Options lose value as time passes
        - Time decay accelerates near expiration
        
        **Theta Acceleration**:
        - Slow decay: Long-dated options
        - Fast decay: Short-dated ATM options
        
        ### Practical Applications
        
        **1. Income Generation**
        - Sell options to collect theta (time premium)
        - Calendar spreads benefit from theta differential
        
        **2. Risk Management**
        - Long positions fight against theta
        - Need underlying movement to overcome time decay
        
        **3. Strategy Selection**
        - High theta strategies: Iron condors, calendars
        - Low theta strategies: LEAPS, deep ITM options
        
        ### Theta Decay Curve
        
        Time decay is not linear:
        - Days 1-30: Minimal decay
        - Days 31-50: Moderate decay
        - Days 51-60: Rapid decay
        - Final week: Exponential decay
        
        ## Vega (ν) - Volatility Sensitivity
        
        ### Definition
        Vega measures the rate of change in an option's price relative to changes in implied volatility.
        
        ### Formula
        ```
        Vega = Change in Option Price / Change in Volatility
        ```
        
        ### Characteristics
        
        **Highest Vega**:
        - ATM options
        - Long time to expiration
        - Moderate volatility levels
        
        **Lowest Vega**:
        - Deep ITM/OTM options
        - Short time to expiration
        
        ### Practical Applications
        
        **1. Volatility Trading**
        - Long vega: Benefit from volatility increase
        - Short vega: Benefit from volatility decrease
        
        **2. Earnings Plays**
        - Implied volatility typically rises before earnings
        - Falls sharply after earnings (volatility crush)
        
        **3. Portfolio Management**
        - Monitor total vega exposure
        - Adjust based on volatility outlook
        
        ## Rho (ρ) - Interest Rate Sensitivity
        
        ### Definition
        Rho measures the rate of change in an option's price relative to changes in interest rates.
        
        ### Characteristics
        
        **Call Options**:
        - Positive rho
        - Benefit from rising interest rates
        
        **Put Options**:
        - Negative rho
        - Hurt by rising interest rates
        
        **Impact**:
        - Generally small compared to other Greeks
        - More significant for LEAPS (long-term options)
        
        ## Greek Interactions
        
        ### Delta-Gamma Relationship
        - Gamma drives delta changes
        - High gamma = unstable delta
        - Low gamma = stable delta
        
        ### Theta-Vega Relationship
        - Often inversely related
        - High theta strategies usually have negative vega
        - High vega strategies usually have negative theta
        
        ### Position Greeks
        
        **Calculating Position Greeks**:
        ```
        Position Greek = Contract Quantity × Option Greek × 100
        ```
        
        **Portfolio Greeks**:
        - Sum of all position Greeks
        - Determines overall risk exposure
        
        ## Greek Management Strategies
        
        ### Delta Neutral Strategies
        - Iron condors
        - Calendar spreads
        - Ratio spreads
        
        ### Positive Theta Strategies
        - Iron condors
        - Credit spreads
        - Calendar spreads
        
        ### Positive Vega Strategies
        - Long straddles/strangles
        - Calendar spreads
        - Debit spreads
        
        ## Practical Greek Calculations
        
        ### Example Position
        ```
        Long 10 ATM calls (30 DTE):
        - Delta: 0.5 × 10 × 100 = 500
        - Gamma: 0.05 × 10 × 100 = 50
        - Theta: -0.10 × 10 × 100 = -$100 per day
        - Vega: 0.20 × 10 × 100 = $200 per 1% vol change
        ```
        
        ### Greek Changes
        
        **Delta Changes**:
        - Stock moves $1: Delta changes by Gamma
        - New delta = Old delta + Gamma × Price change
        
        **Theta Decay**:
        - Daily theta loss
        - Accelerates as expiration approaches
        
        **Vega Impact**:
        - IV changes affect option price
        - Vega × IV change = price change
        
        ## Advanced Greek Concepts
        
        ### Volatility Smile
        - OTM options have higher implied volatility
        - Affects delta calculations
        - Important for accurate pricing
        
        ### Surface Greeks
        - Second-order Greeks
        - Vanna: Delta sensitivity to volatility
        - Charm: Delta sensitivity to time
        - Vomma: Vega sensitivity to volatility
        
        ### Portfolio Optimization
        - Balance Greeks for desired risk profile
        - Adjust positions based on Greek targets
        - Monitor Greek changes over time
        
        ## Common Greek Mistakes
        
        1. **Ignoring gamma risk in delta-neutral positions**
        2. **Underestimating theta decay near expiration**
        3. **Forgetting vega risk in volatile markets**
        4. **Not monitoring cumulative Greek exposure**
        5. **Failing to adjust Greeks as market conditions change**
        
        ## Best Practices
        
        1. **Calculate Greeks before entering positions**
        2. **Monitor Greek changes daily**
        3. **Set Greek limits for your portfolio**
        4. **Understand how Greeks interact**
        5. **Use Greeks for position adjustment decisions**
        6. **Practice Greek calculations with paper trading**
        
        Remember: Greeks are your risk management tools. Master them to become a successful options trader.
        """
    
    def get_advanced_strategies_content(self) -> str:
        """Get advanced options strategies content"""
        return """
        # Advanced Options Strategies
        
        ## Multi-Leg Strategies
        
        ### Iron Condor
        
        **Structure**:
        - Bull put spread (sell put, buy lower strike put)
        - Bear call spread (sell call, buy higher strike call)
        - Both spreads have same expiration
        
        **Best When**:
        - Market expected to stay in range
        - Low volatility environment
        - Income generation focus
        
        **Management**:
        - Close at 50% max profit
        - Adjust if price approaches short strikes
        - Consider rolling spreads
        
        ### Butterfly Spreads
        
        **Long Butterfly**:
        - Buy 1 lower strike
        - Sell 2 middle strikes
        - Buy 1 higher strike
        - Limited risk, limited reward
        
        **Iron Butterfly**:
        - Sell ATM straddle
        - Buy OTM strangle
        - Credit spread version
        
        **Best When**:
        - Expecting minimal price movement
        - High volatility (expensive options)
        - Precise range prediction
        
        ### Calendar Spreads (Time Spreads)
        
        **Structure**:
        - Sell near-term option
        - Buy longer-term option
        - Same strike price
        
        **Types**:
        - Call calendar: Bullish
        - Put calendar: Bearish
        - Neutral calendar: Range-bound
        
        **Best When**:
        - Expecting minimal price movement
        - Benefit from time decay differential
        - Volatility plays
        
        ## Volatility Strategies
        
        ### Straddle vs Strangle
        
        **Long Straddle**:
        - Buy ATM call + ATM put
        - High cost, high gamma
        - Requires any directional movement
        
        **Long Strangle**:
        - Buy OTM call + OTM put
        - Lower cost, lower gamma
        - Requires larger movement
        
        **Selection Criteria**:
        - Expected move size
        - Cost considerations
        - Risk tolerance
        
        ### Volatility Skew Trading
        
        **Understanding Skew**:
        - OTM puts typically more expensive
        - Reflects crash risk premium
        - Varies by market conditions
        
        **Strategies**:
        - Ratio spreads: Exploit skew differences
        - Risk reversals: Play skew changes
        - Vertical spreads: Skew-aware positioning
        
        ### Variance Trading
        
        **VIX Futures**:
        - Trade implied volatility directly
        - No delta exposure
        - Pure volatility play
        
        **Variance Swaps**:
        - OTC contracts
        - Realized vs implied volatility
        - Institutional product
        
        ## Ratio Spreads
        
        ### Call Ratio Spread
        
        **Structure**:
        - Buy 1 lower strike call
        - Sell 2+ higher strike calls
        - Net credit or small debit
        
        **Best When**:
        - Moderately bullish
        - Expect limited upside
        - Income generation
        
        **Risks**:
        - Unlimited risk above short strikes
        - Requires careful management
        
        ### Put Ratio Spread
        
        **Structure**:
        - Buy 1 higher strike put
        - Sell 2+ lower strike puts
        - Net credit or small debit
        
        **Best When**:
        - Moderately bearish
        - Expect limited downside
        - Income generation
        
        ## Exotic Strategies
        
        ### Diagonal Spreads
        
        **Structure**:
        - Different strikes AND different expirations
        - Combines vertical and calendar spread features
        
        **Types**:
        - Call diagonal: Bullish
        - Put diagonal: Bearish
        - Double diagonal: Range-bound
        
        **Advantages**:
        - Flexible risk/reward
        - Time decay benefits
        - Directional exposure
        
        ### Strip and Strap Strategies
        
        **Strip**:
        - Long 1 ATM call + Long 2 ATM puts
        - Bearish bias
        - Benefits from downward movement
        
        **Strap**:
            - Long 2 ATM calls + Long 1 ATM put
            - Bullish bias
            - Benefits from upward movement
        
        ## Dynamic Hedging
        
        ### Delta Hedging
        
        **Continuous Hedging**:
        - Adjust delta to neutral
        - Frequency depends on gamma
        - Transaction costs vs. risk reduction
        
        **Delta Bands**:
        - Allow delta to fluctuate within range
        - Reduce transaction costs
        - Accept some directional risk
        
        ### Gamma Scalping
        
        **Concept**:
        - Long gamma positions
        - Profit from volatility
        - Buy low, sell high automatically
        
        **Implementation**:
        - Maintain delta neutrality
        - Adjust as underlying moves
        - Capture gamma profits
        
        ## Portfolio Strategies
        
        ### Collars
        
        **Structure**:
        - Own 100 shares
        - Buy protective put (OTM)
        - Sell covered call (OTM)
        - Limited risk, limited reward
        
        **Variations**:
        - Zero-cost collars
        - Flexible collars
        - Dynamic collars
        
        ### Synthetic Positions
        
        **Synthetic Long Stock**:
        - Long ATM call
        - Short ATM put
        - Same risk/reward as stock
        
        **Synthetic Short Stock**:
        - Short ATM call
        - Long ATM put
        - Same risk/reward as short stock
        
        **Advantages**:
        - Leverage
        - No borrowing costs
        - Flexible strikes
        
        ## Risk Management for Advanced Strategies
        
        ### Position Sizing
        
        **Complex Positions**:
        - Consider total risk
        - Account for all legs
        - Margin requirements
        
        **Portfolio Impact**:
        - Greek exposure
        - Correlation with existing positions
        - Concentration risk
        
        ### Adjustment Strategies
        
        **Rolling**:
        - Extend duration
        - Change strikes
        - Maintain position characteristics
        
        **Converting**:
        - Change strategy type
        - Respond to market conditions
        - Manage risk/reward
        
        ### Stop Losses for Spreads
        
        **Percentage Stops**:
            - Close at predefined loss percentage
            - Simple to implement
            - May not account for Greeks
        
        **Greek-Based Stops**:
        - Close when Greeks exceed limits
        - More sophisticated
        - Requires monitoring
        
        ## Market Conditions and Strategy Selection
        
        ### High Volatility Environments
        
        **Preferred Strategies**:
        - Iron condors (high premium)
        - Calendar spreads (volatility selling)
        - Credit spreads
        
        **Avoid**:
        - Long premium strategies
        - High gamma positions
        
        ### Low Volatility Environments
        
        **Preferred Strategies**:
        - Long straddles/strangles
        - Debit spreads
        - Calendar spreads (buying volatility)
        
        **Avoid**:
        - Credit spreads (low premium)
        - Iron condors
        
        ### Trending Markets
        
        **Uptrend**:
        - Bull call spreads
        - Call calendars
        - Covered calls
        
        **Downtrend**:
        - Bear put spreads
        - Put calendars
        - Protective puts
        
        ### Range-Bound Markets
        
        **Preferred Strategies**:
        - Iron condors
        - Butterflies
        - Calendar spreads
        
        ## Advanced Concepts
        
        ### Implied Volatility Surface
        
        **Understanding**:
        - Volatility varies by strike and expiration
        - Creates 3D surface
        - Changes over time
        
        **Trading**:
        - Surface arbitrage
        - Volatility term structure
        - Skew trading
        
        ### Correlation Trading
        
        **Multi-Asset Strategies**:
        - Pairs trading with options
        - Basket options
        - Rainbow options
        
        **Dispersion Trading**:
        - Index options vs individual options
        - Correlation plays
        - Volatility dispersion
        
        ## Best Practices for Advanced Trading
        
        1. **Master basics first**: Don't skip fundamentals
        2. **Understand all risks**: Know maximum loss scenarios
        3. **Start with paper trading**: Test strategies without risk
        4. **Keep detailed records**: Track all adjustments and outcomes
        5. **Monitor Greeks continuously**: Real-time risk management
        6. **Have adjustment plans**: Know what to do when
        7. **Consider transaction costs**: Multiple legs increase costs
        8. **Stay disciplined**: Stick to your strategy rules
        
        Advanced options trading requires deep understanding, continuous learning, and disciplined execution. Start simple and gradually increase complexity as you gain experience.
        """
    
    def _get_long_call_examples(self) -> List[Dict]:
        """Get long call examples"""
        return [
            {
                "title": "Basic Long Call",
                "description": "Simple bullish call option purchase",
                "data": {
                    "stock_price": 100,
                    "strike_price": 105,
                    "premium": 3.50,
                    "days_to_expiration": 30,
                    "breakeven": 108.50,
                    "max_loss": 350,
                    "max_profit": "Unlimited",
                    "delta": 0.40,
                    "theta": -0.05,
                    "vega": 0.15
                }
            },
            {
                "title": "LEAPS Call",
                "description": "Long-term call option for strategic positioning",
                "data": {
                    "stock_price": 150,
                    "strike_price": 160,
                    "premium": 12.00,
                    "days_to_expiration": 365,
                    "breakeven": 172.00,
                    "max_loss": 1200,
                    "max_profit": "Unlimited",
                    "delta": 0.55,
                    "theta": -0.02,
                    "vega": 0.35
                }
            }
        ]
    
    def _get_long_put_examples(self) -> List[Dict]:
        """Get long put examples"""
        return [
            {
                "title": "Protective Put",
                "description": "Buying insurance for stock position",
                "data": {
                    "stock_price": 80,
                    "strike_price": 75,
                    "premium": 2.00,
                    "days_to_expiration": 45,
                    "breakeven": 73.00,
                    "max_loss": 200,
                    "max_profit": 7300,
                    "delta": -0.35,
                    "theta": -0.03,
                    "vega": 0.12
                }
            }
        ]
    
    def _get_covered_call_examples(self) -> List[Dict]:
        """Get covered call examples"""
        return [
            {
                "title": "Monthly Covered Call",
                "description": "Generating monthly income from stock holdings",
                "data": {
                    "stock_cost": 50.00,
                    "call_strike": 55,
                    "call_premium": 1.50,
                    "days_to_expiration": 30,
                    "max_profit": 650,
                    "breakeven": 48.50,
                    "assignment_price": 55.00,
                    "return_if_assigned": 13.0,
                    "annualized_return": 156.0
                }
            }
        ]
    
    def _get_protective_put_examples(self) -> List[Dict]:
        """Get protective put examples"""
        return [
            {
                "title": "Portfolio Insurance",
                "description": "Protecting long-term stock position",
                "data": {
                    "stock_cost": 120.00,
                    "put_strike": 110,
                    "put_premium": 3.00,
                    "days_to_expiration": 90,
                    "max_loss": 1300,
                    "protection_level": 110.00,
                    "total_cost": 123.00,
                    "insurance_cost": 2.5
                }
            }
        ]
    
    def _get_iron_condor_examples(self) -> List[Dict]:
        """Get iron condor examples"""
        return [
            {
                "title": "Standard Iron Condor",
                "description": "Income strategy with defined risk",
                "data": {
                    "current_price": 100,
                    "put_spread": [90, 95],
                    "call_spread": [105, 110],
                    "credit_received": 1.75,
                    "max_profit": 175,
                    "max_loss": 325,
                    "breakevens": [93.25, 106.75],
                    "probability_of_profit": 65,
                    "days_to_expiration": 30
                }
            }
        ]
    
    def _get_straddle_examples(self) -> List[Dict]:
        """Get straddle examples"""
        return [
            {
                "title": "Earnings Straddle",
                "description": "Playing earnings announcement with straddle",
                "data": {
                    "stock_price": 75,
                    "strike_price": 75,
                    "call_premium": 4.50,
                    "put_premium": 4.00,
                    "total_cost": 8.50,
                    "breakevens": [66.50, 83.50],
                    "max_loss": 850,
                    "max_profit": "Unlimited",
                    "implied_move": 11.3,
                    "days_to_expiration": 7
                }
            }
        ]
    
    def _get_strangle_examples(self) -> List[Dict]:
        """Get strangle examples"""
        return [
            {
                "title": "OTM Strangle",
                "description": "Lower cost alternative to straddle",
                "data": {
                    "stock_price": 100,
                    "call_strike": 105,
                    "put_strike": 95,
                    "call_premium": 2.25,
                    "put_premium": 2.00,
                    "total_cost": 4.25,
                    "breakevens": [90.75, 109.25],
                    "max_loss": 425,
                    "max_profit": "Unlimited",
                    "days_to_expiration": 30
                }
            }
        ]
    
    def _get_butterfly_examples(self) -> List[Dict]:
        """Get butterfly spread examples"""
        return [
            {
                "title": "Call Butterfly",
                "description": "Limited risk, limited reward strategy",
                "data": {
                    "current_price": 50,
                    "strikes": [45, 50, 55],
                    "cost": 1.25,
                    "max_profit": 375,
                    "max_loss": 125,
                    "breakevens": [46.25, 53.75],
                    "profit_zone": 6.25,
                    "days_to_expiration": 21
                }
            }
        ]
    
    def _get_calendar_examples(self) -> List[Dict]:
        """Get calendar spread examples"""
        return [
            {
                "title": "Call Calendar",
                "description": "Time spread with bullish bias",
                "data": {
                    "current_price": 80,
                    "strike": 80,
                    "short_dte": 30,
                    "long_dte": 60,
                    "short_premium": 2.50,
                    "long_premium": 3.75,
                    "net_cost": 1.25,
                    "max_profit": 150,
                    "max_loss": 125,
                    "optimal_price": 80.00
                }
            }
        ]
    
    def calculate_option_greeks(self, option_type: str, stock_price: float, strike: float, 
                              time_to_expiration: float, risk_free_rate: float, 
                              volatility: float) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes model"""
        return self.greeks_calculator.calculate_all_greeks(
            option_type, stock_price, strike, time_to_expiration, 
            risk_free_rate, volatility
        )
    
    def price_option(self, option_type: str, stock_price: float, strike: float,
                    time_to_expiration: float, risk_free_rate: float,
                    volatility: float) -> float:
        """Calculate option price using Black-Scholes model"""
        return self.pricing_models.black_scholes_price(
            option_type, stock_price, strike, time_to_expiration,
            risk_free_rate, volatility
        )
    
    def analyze_strategy_risk(self, strategy_name: str, underlying_price: float,
                            volatility: float, time_to_expiration: float) -> Dict[str, Any]:
        """Analyze risk characteristics of an option strategy"""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return {"error": "Strategy not found"}
        
        # This would contain detailed risk analysis logic
        return {
            "strategy": strategy.name,
            "risk_level": strategy.risk_level,
            "max_profit": strategy.max_profit,
            "max_loss": strategy.max_loss,
            "greeks_exposure": strategy.greeks_exposure,
            "current_analysis": "Detailed risk analysis would be implemented here"
        }


class GreeksCalculator:
    """Calculate option Greeks using various models"""
    
    def __init__(self):
        pass
    
    def calculate_all_greeks(self, option_type: str, stock_price: float, strike: float,
                           time_to_expiration: float, risk_free_rate: float,
                           volatility: float) -> Dict[str, float]:
        """Calculate all Greeks using Black-Scholes model"""
        
        d1 = self._calculate_d1(stock_price, strike, time_to_expiration, risk_free_rate, volatility)
        d2 = d1 - volatility * math.sqrt(time_to_expiration)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (stock_price * volatility * math.sqrt(time_to_expiration))
        
        # Theta
        if option_type.lower() == 'call':
            theta = (-stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_expiration))
                    - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)) / 365
        else:
            theta = (-stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_expiration))
                    + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiration) / 100
        
        # Rho
        if option_type.lower() == 'call':
            rho = strike * time_to_expiration * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2) / 100
        else:
            rho = -strike * time_to_expiration * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }
    
    def _calculate_d1(self, stock_price: float, strike: float, time_to_expiration: float,
                      risk_free_rate: float, volatility: float) -> float:
        """Calculate d1 parameter for Black-Scholes"""
        return (math.log(stock_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))


class OptionsPricing:
    """Options pricing models"""
    
    def __init__(self):
        pass
    
    def black_scholes_price(self, option_type: str, stock_price: float, strike: float,
                           time_to_expiration: float, risk_free_rate: float,
                           volatility: float) -> float:
        """Calculate option price using Black-Scholes model"""
        
        d1 = (math.log(stock_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
        d2 = d1 - volatility * math.sqrt(time_to_expiration)
        
        if option_type.lower() == 'call':
            price = stock_price * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)
        else:
            price = strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        
        return price
    
    def binomial_price(self, option_type: str, stock_price: float, strike: float,
                      time_to_expiration: float, risk_free_rate: float,
                      volatility: float, steps: int = 100) -> float:
        """Calculate option price using binomial tree model"""
        
        dt = time_to_expiration / steps
        u = math.exp(volatility * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(risk_free_rate * dt) - d) / (u - d)
        
        # Initialize stock prices at expiration
        stock_prices = [stock_price * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        
        # Calculate option values at expiration
        if option_type.lower() == 'call':
            option_values = [max(0, price - strike) for price in stock_prices]
        else:
            option_values = [max(0, strike - price) for price in stock_prices]
        
        # Work backwards through the tree
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = (p * option_values[j + 1] + (1 - p) * option_values[j]) * math.exp(-risk_free_rate * dt)
        
        return option_values[0]


# Factory function
def create_options_education_module() -> OptionsEducationModule:
    """Create and return an OptionsEducationModule instance"""
    return OptionsEducationModule()


# Example usage
if __name__ == "__main__":
    module = create_options_education_module()
    
    # Calculate Greeks for a sample option
    greeks = module.calculate_option_greeks(
        'call', 100, 105, 30/365, 0.05, 0.25
    )
    print(f"Option Greeks: {greeks}")
    
    # Price an option
    price = module.price_option(
        'call', 100, 105, 30/365, 0.05, 0.25
    )
    print(f"Option Price: {price:.2f}")
    
    # Get strategy information
    strategy = module.strategies.get('iron_condor')
    if strategy:
        print(f"Strategy: {strategy.name}")
        print(f"Risk Level: {strategy.risk_level}")