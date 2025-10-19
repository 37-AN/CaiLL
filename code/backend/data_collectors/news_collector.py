"""
News and Sentiment Data Collector

This module implements data collection for financial news and sentiment analysis
from multiple sources including NewsAPI, Twitter, and financial news websites.
It provides real-time news updates with sentiment analysis.
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import asdict

import pandas as pd
from textblob import TextBlob

from backend.data_collectors.base_collector import (
    BaseDataCollector, MarketData, NewsData, 
    DataType, DataFrequency, RateLimiter
)
from backend.core.config import settings
from backend.core.exceptions import DataCollectionError, APIError, RateLimitError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class NewsAPICollector(BaseDataCollector):
    """
    NewsAPI data collector for financial news.
    
    This collector provides access to news articles from various sources
    with filtering capabilities for financial news.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("newsapi", config)
        self.api_key = config.get("api_key", settings.NEWS_API_KEY)
        self.base_url = "https://newsapi.org/v2"
        self.session = None
        
    async def initialize(self) -> None:
        """Initialize the NewsAPI collector."""
        if not self.api_key:
            raise DataCollectionError("NewsAPI key is required", "newsapi")
        
        self.session = aiohttp.ClientSession()
        self.logger.info("NewsAPI collector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.logger.info("NewsAPI collector cleaned up")
    
    async def get_news(
        self,
        query: str = "stocks OR trading OR finance OR market",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100
    ) -> List[NewsData]:
        """
        Get news articles from NewsAPI.
        
        Args:
            query: Search query
            from_date: Start date for news
            to_date: End date for news
            language: Language code
            sort_by: Sort order
            page_size: Number of articles per page
            
        Returns:
            List of news articles
        """
        await self._apply_rate_limit()
        
        try:
            # Prepare parameters
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'apiKey': self.api_key
            }
            
            if from_date:
                params['from'] = from_date.strftime('%Y-%m-%d')
            if to_date:
                params['to'] = to_date.strftime('%Y-%m-%d')
            
            # Make API request
            url = f"{self.base_url}/everything"
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise APIError(f"NewsAPI error: {error_data.get('message', 'Unknown error')}", 
                                 "newsapi", response.status)
                
                data = await response.json()
            
            # Convert to NewsData objects
            news_articles = []
            for article in data.get('articles', []):
                try:
                    # Extract sentiment using TextBlob
                    sentiment = self._analyze_sentiment(article.get('description', '') + 
                                                      article.get('content', ''))
                    
                    # Extract relevant symbols from title and content
                    symbols = self._extract_symbols(article.get('title', '') + 
                                                   article.get('content', ''))
                    
                    news_data = NewsData(
                        symbol=symbols[0] if symbols else None,  # Use first symbol found
                        timestamp=self._parse_newsapi_date(article.get('publishedAt')),
                        title=article.get('title', ''),
                        content=article.get('content', article.get('description', '')),
                        source=article.get('source', {}).get('name', 'NewsAPI'),
                        url=article.get('url', ''),
                        sentiment_score=sentiment,
                        relevance_score=self._calculate_relevance(article, query)
                    )
                    
                    news_articles.append(news_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing news article: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(news_articles)} news articles")
            return news_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            raise DataCollectionError(f"Failed to fetch news: {e}", "newsapi")
    
    async def get_company_news(
        self,
        company_name: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[NewsData]:
        """
        Get news specifically about a company.
        
        Args:
            company_name: Company name to search for
            from_date: Start date
            to_date: End date
            
        Returns:
            List of news articles about the company
        """
        query = f'"{company_name}" AND (stocks OR finance OR market OR trading)'
        return await self.get_news(query, from_date, to_date)
    
    async def get_real_time_news(
        self,
        query: str = "stocks OR trading OR finance OR market"
    ) -> AsyncGenerator[NewsData, None]:
        """
        Get real-time news updates.
        
        Args:
            query: Search query
            
        Yields:
            Real-time news articles
        """
        last_fetch_time = datetime.now() - timedelta(minutes=5)
        
        while self.is_running:
            try:
                await self._apply_rate_limit()
                
                # Get news from the last 5 minutes
                current_time = datetime.now()
                news_articles = await self.get_news(
                    query=query,
                    from_date=last_fetch_time,
                    to_date=current_time,
                    page_size=20
                )
                
                # Yield new articles
                for article in news_articles:
                    yield article
                
                last_fetch_time = current_time
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 300))  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in real-time news: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _extract_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text.
        
        Args:
            text: Text to extract symbols from
            
        Returns:
            List of stock symbols found
        """
        # Simple regex to find stock symbols (e.g., $AAPL, AAPL)
        # This is a basic implementation - in practice, you'd use more sophisticated NLP
        symbol_pattern = r'\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\b'
        matches = re.findall(symbol_pattern, text.upper())
        
        # Flatten matches and remove duplicates
        symbols = list(set([match[0] or match[1] for match in matches]))
        
        # Filter out common words that might match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return [symbol for symbol in symbols if symbol not in common_words and len(symbol) >= 2]
    
    def _calculate_relevance(self, article: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance score for an article.
        
        Args:
            article: Article data
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        
        # Count query term matches
        query_terms = query.lower().split()
        total_text = f"{title} {description} {content}"
        
        matches = sum(1 for term in query_terms if term in total_text)
        relevance = min(matches / len(query_terms), 1.0)
        
        return relevance
    
    def _parse_newsapi_date(self, date_str: str) -> datetime:
        """
        Parse NewsAPI date string.
        
        Args:
            date_str: Date string from NewsAPI
            
        Returns:
            Parsed datetime
        """
        try:
            # NewsAPI uses ISO 8601 format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception as e:
            self.logger.warning(f"Error parsing date {date_str}: {e}")
            return datetime.now()
    
    async def _start_collection_loop(self, symbols: List[str]) -> None:
        """
        Start the news collection loop.
        
        Args:
            symbols: List of symbols (not used for news collection)
        """
        try:
            async for news in self.get_real_time_news():
                if not self.is_running:
                    break
                
                self.logger.debug(f"Received news: {news.title}")
                
        except Exception as e:
            self.logger.error(f"Error in news collection loop: {e}")


class TwitterCollector(BaseDataCollector):
    """
    Twitter data collector for financial sentiment.
    
    This collector provides access to Twitter data for sentiment analysis
    and market sentiment tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("twitter", config)
        self.api_key = config.get("api_key", settings.TWITTER_API_KEY)
        self.api_secret = config.get("api_secret", settings.TWITTER_API_SECRET)
        self.access_token = config.get("access_token", settings.TWITTER_ACCESS_TOKEN)
        self.access_token_secret = config.get("access_token_secret", settings.TWITTER_ACCESS_TOKEN_SECRET)
        self.bearer_token = config.get("bearer_token")
        self.session = None
        
    async def initialize(self) -> None:
        """Initialize the Twitter collector."""
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            self.logger.warning("Twitter API credentials not fully configured")
            return
        
        self.session = aiohttp.ClientSession()
        self.logger.info("Twitter collector initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.logger.info("Twitter collector cleaned up")
    
    async def search_tweets(
        self,
        query: str,
        count: int = 100,
        result_type: str = "recent"
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets with financial content.
        
        Args:
            query: Search query
            count: Number of tweets to retrieve
            result_type: Type of results (recent, popular, mixed)
            
        Returns:
            List of tweet data
        """
        await self._apply_rate_limit()
        
        try:
            # This is a simplified implementation
            # In practice, you'd use the Twitter API v2 with proper authentication
            
            # For now, return mock data
            mock_tweets = [
                {
                    "id": "123456789",
                    "text": f"Mock tweet about {query}",
                    "created_at": datetime.now().isoformat(),
                    "user": {"screen_name": "mock_user"},
                    "sentiment": 0.5
                }
            ]
            
            self.logger.info(f"Retrieved {len(mock_tweets)} tweets for query: {query}")
            return mock_tweets
            
        except Exception as e:
            self.logger.error(f"Error searching tweets: {e}")
            return []
    
    async def get_symbol_tweets(self, symbol: str, count: int = 50) -> List[Dict[str, Any]]:
        """
        Get tweets about a specific stock symbol.
        
        Args:
            symbol: Stock symbol
            count: Number of tweets to retrieve
            
        Returns:
            List of tweets about the symbol
        """
        query = f"${symbol} OR {symbol} stock"
        return await self.search_tweets(query, count)
    
    async def _start_collection_loop(self, symbols: List[str]) -> None:
        """
        Start the Twitter collection loop.
        
        Args:
            symbols: List of symbols to track
        """
        while self.is_running:
            try:
                for symbol in symbols:
                    if not self.is_running:
                        break
                    
                    tweets = await self.get_symbol_tweets(symbol)
                    
                    for tweet in tweets:
                        # Process tweet sentiment
                        sentiment = self._analyze_sentiment(tweet.get("text", ""))
                        
                        # Create market data for sentiment
                        data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            data_type=DataType.SENTIMENT,
                            frequency=DataFrequency.TICK,
                            data={
                                "sentiment": sentiment,
                                "source": "twitter",
                                "text": tweet.get("text", ""),
                                "user": tweet.get("user", {}).get("screen_name", ""),
                                "tweet_id": tweet.get("id", "")
                            },
                            source="twitter"
                        )
                        
                        self.logger.debug(f"Twitter sentiment for {symbol}: {sentiment}")
                
                # Wait before next update
                await asyncio.sleep(self.config.get("update_interval", 300))  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in Twitter collection loop: {e}")
                await asyncio.sleep(60)
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of tweet text.
        
        Args:
            text: Tweet text
            
        Returns:
            Sentiment score between -1 and 1
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            self.logger.warning(f"Error analyzing tweet sentiment: {e}")
            return 0.0


class NewsDataCollector:
    """
    Main news and sentiment data collector that manages multiple sources.
    
    This class provides a unified interface for collecting news and sentiment
    data from multiple sources with automatic failover.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("news_data_collector")
        self.collectors = {}
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """Initialize all available news data collectors."""
        # NewsAPI collector (if API key is available)
        if settings.NEWS_API_KEY:
            self.collectors["newsapi"] = NewsAPICollector(
                self.config.get("newsapi", {"api_key": settings.NEWS_API_KEY})
            )
        
        # Twitter collector (if credentials are available)
        if all([settings.TWITTER_API_KEY, settings.TWITTER_API_SECRET, 
                settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET]):
            self.collectors["twitter"] = TwitterCollector(
                self.config.get("twitter", {
                    "api_key": settings.TWITTER_API_KEY,
                    "api_secret": settings.TWITTER_API_SECRET,
                    "access_token": settings.TWITTER_ACCESS_TOKEN,
                    "access_token_secret": settings.TWITTER_ACCESS_TOKEN_SECRET
                })
            )
        
        self.logger.info(f"Initialized {len(self.collectors)} news data collectors")
    
    async def initialize(self) -> None:
        """Initialize all collectors."""
        for name, collector in self.collectors.items():
            try:
                await collector.initialize()
                self.logger.info(f"Initialized {name} collector")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name} collector: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all collectors."""
        for name, collector in self.collectors.items():
            try:
                await collector.cleanup()
                self.logger.info(f"Cleaned up {name} collector")
            except Exception as e:
                self.logger.error(f"Failed to cleanup {name} collector: {e}")
    
    async def get_news(
        self,
        query: str = "stocks OR trading OR finance OR market",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        preferred_source: Optional[str] = None
    ) -> List[NewsData]:
        """
        Get news from the best available source.
        
        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            preferred_source: Preferred news source
            
        Returns:
            List of news articles
        """
        # Try preferred source first
        if preferred_source and preferred_source in self.collectors:
            try:
                news = await self.collectors[preferred_source].get_news(
                    query, from_date, to_date
                )
                if news:
                    return news
            except Exception as e:
                self.logger.warning(f"Preferred source {preferred_source} failed: {e}")
        
        # Try other sources
        for name, collector in self.collectors.items():
            if name == preferred_source:
                continue  # Already tried
            
            try:
                if hasattr(collector, 'get_news'):
                    news = await collector.get_news(query, from_date, to_date)
                    if news:
                        self.logger.info(f"Got news from {name}")
                        return news
            except Exception as e:
                self.logger.warning(f"Source {name} failed: {e}")
                continue
        
        return []
    
    async def get_company_news(
        self,
        company_name: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[NewsData]:
        """
        Get news about a specific company.
        
        Args:
            company_name: Company name
            from_date: Start date
            to_date: End date
            
        Returns:
            List of news articles about the company
        """
        all_news = []
        
        for name, collector in self.collectors.items():
            try:
                if hasattr(collector, 'get_company_news'):
                    news = await collector.get_company_news(company_name, from_date, to_date)
                    all_news.extend(news)
            except Exception as e:
                self.logger.warning(f"Error getting company news from {name}: {e}")
        
        # Sort by relevance and timestamp
        all_news.sort(key=lambda x: (x.relevance_score or 0, x.timestamp), reverse=True)
        
        return all_news
    
    async def start_real_time_collection(self, symbols: List[str]) -> None:
        """
        Start real-time news collection.
        
        Args:
            symbols: List of symbols to track
        """
        self.logger.info(f"Starting real-time news collection for {len(symbols)} symbols")
        
        for name, collector in self.collectors.items():
            try:
                await collector.start_collection(symbols)
                self.logger.info(f"Started news collection with {name}")
            except Exception as e:
                self.logger.error(f"Failed to start news collection with {name}: {e}")
    
    async def stop_real_time_collection(self) -> None:
        """Stop real-time news collection."""
        self.logger.info("Stopping real-time news collection")
        
        for collector in self.collectors.values():
            await collector.stop_collection()
    
    async def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get sentiment summary for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Number of hours to look back
            
        Returns:
            Sentiment summary
        """
        # This is a simplified implementation
        # In practice, you'd query your sentiment database
        
        return {
            "symbol": symbol,
            "period_hours": hours,
            "overall_sentiment": 0.1,  # Neutral to slightly positive
            "sentiment_trend": "improving",
            "news_count": 15,
            "positive_count": 8,
            "negative_count": 4,
            "neutral_count": 3,
            "last_updated": datetime.now().isoformat()
        }