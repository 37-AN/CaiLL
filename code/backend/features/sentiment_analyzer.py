"""
Sentiment Analyzer Module - Phase 2.3

This module implements comprehensive sentiment analysis for financial news,
social media, and market commentary. It uses multiple approaches to gauge
market sentiment and情绪 (emotions) that affect trading decisions.

Educational Note:
Sentiment analysis in trading helps quantify the psychological factors
that drive market movements. Fear, greed, optimism, and panic can create
predictable patterns that algorithms can exploit.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import logging
from enum import Enum
import json

# NLP and ML libraries
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    """Types of sentiment analysis"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST = "analyst"
    MARKET_COMMENTARY = "market_commentary"

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: SentimentLabel
    confidence: float
    source: str
    timestamp: datetime
    topics: List[str] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    financial_entities: List[str] = field(default_factory=list)
    market_impact_score: float = 0.0

@dataclass
class MarketSentiment:
    """Aggregated market sentiment"""
    overall_sentiment: float
    sentiment_distribution: Dict[SentimentLabel, float]
    sentiment_trend: List[float]
    volume_weighted_sentiment: float
    source_breakdown: Dict[str, float]
    key_topics: List[Tuple[str, float]]
    emotional_state: Dict[str, float]
    timestamp: datetime

class FinancialSentimentAnalyzer:
    """
    Comprehensive Financial Sentiment Analyzer
    
    Educational Notes:
    - Financial sentiment is different from general sentiment
    - Market-specific vocabulary requires specialized models
    - Context matters: "beat expectations" is positive, "beat down" is negative
    - Time decay: Recent news has more impact than older news
    """
    
    def __init__(self):
        self.nlp = None
        self.vader_analyzer = None
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.financial_keywords = self._load_financial_keywords()
        self.emotion_keywords = self._load_emotion_keywords()
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        logger.info("FinancialSentimentAnalyzer initialized")
    
    def _initialize_nltk(self):
        """Initialize NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize VADER sentiment analyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("NLTK components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            # Continue without NLTK - use fallback methods
    
    def _load_financial_keywords(self) -> Dict[str, float]:
        """
        Load financial keywords with sentiment weights
        
        Educational: Financial terms have specific sentiment meanings
        that general sentiment analyzers might miss.
        """
        return {
            # Positive financial terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.6, 'soar': 0.8,
            'breakout': 0.6, 'momentum': 0.5, 'uptrend': 0.7, 'growth': 0.5,
            'expansion': 0.5, 'recovery': 0.6, 'boom': 0.8, 'prosperity': 0.7,
            'dividend': 0.3, 'profit': 0.6, 'earnings': 0.4, 'revenue': 0.4,
            'beat': 0.5, 'exceed': 0.6, 'outperform': 0.7, 'upgrade': 0.6,
            'buy': 0.4, 'accumulate': 0.5, 'overweight': 0.5, 'strong_buy': 0.8,
            
            # Negative financial terms
            'bearish': -0.8, 'crash': -0.9, 'plunge': -0.8, 'slump': -0.7,
            'downtrend': -0.7, 'recession': -0.8, 'depression': -0.9, 'crisis': -0.8,
            'loss': -0.6, 'deficit': -0.5, 'bankruptcy': -0.9, 'default': -0.8,
            'miss': -0.5, 'underperform': -0.6, 'downgrade': -0.6, 'sell': -0.4,
            'reduce': -0.3, 'underweight': -0.4, 'strong_sell': -0.8,
            'volatility': -0.2, 'uncertainty': -0.3, 'risk': -0.2,
            
            # Neutral/Context-dependent terms
            'stable': 0.1, 'steady': 0.1, 'flat': 0.0, 'neutral': 0.0,
            'mixed': 0.0, 'cautious': -0.1, 'optimistic': 0.3, 'pessimistic': -0.3,
        }
    
    def _load_emotion_keywords(self) -> Dict[str, Dict[str, float]]:
        """
        Load emotion-specific keywords for financial context
        
        Educational: Market emotions drive trading behavior:
        - Fear: Leads to selling, risk aversion
        - Greed: Leads to buying, risk seeking
        - Panic: Extreme fear, rapid selling
        - Euphoria: Extreme greed, irrational buying
        """
        return {
            'fear': {
                'scared': 0.8, 'afraid': 0.7, 'terrified': 0.9, 'panic': 0.9,
                'concern': 0.5, 'worry': 0.6, 'anxious': 0.7, 'nervous': 0.6,
                'flight': 0.8, 'avoid': 0.6, 'escape': 0.7, 'danger': 0.8
            },
            'greed': {
                'greedy': 0.9, 'profit': 0.6, 'gain': 0.5, 'rich': 0.7,
                'wealth': 0.6, 'fortune': 0.8, 'money': 0.4, 'cash': 0.3,
                'opportunity': 0.5, 'windfall': 0.8, 'jackpot': 0.9, 'bonus': 0.6
            },
            'panic': {
                'crash': 0.9, 'collapse': 0.9, 'meltdown': 0.9, 'disaster': 0.8,
                'catastrophe': 0.9, 'emergency': 0.8, 'critical': 0.7, 'urgent': 0.6,
                'run': 0.8, 'stampede': 0.9, 'chaos': 0.8, 'turmoil': 0.7
            },
            'euphoria': {
                'euphoric': 0.9, 'ecstatic': 0.9, 'thrilled': 0.8, 'excited': 0.7,
                'celebration': 0.8, 'triumph': 0.8, 'victory': 0.7, 'success': 0.6,
                'amazing': 0.7, 'incredible': 0.8, 'fantastic': 0.7, 'perfect': 0.8
            },
            'hope': {
                'hope': 0.8, 'optimistic': 0.7, 'confident': 0.6, 'believe': 0.5,
                'expect': 0.4, 'anticipate': 0.5, 'forecast': 0.3, 'predict': 0.3,
                'potential': 0.4, 'promise': 0.5, 'future': 0.3, 'outlook': 0.3
            },
            'disappointment': {
                'disappointed': 0.8, 'sad': 0.6, 'depressed': 0.7, 'frustrated': 0.6,
                'failed': 0.7, 'missed': 0.5, 'letdown': 0.7, 'regret': 0.6,
                'sorry': 0.5, 'unfortunate': 0.4, 'poor': 0.5, 'weak': 0.4
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Educational: Financial text preprocessing requires special handling:
        - Numbers and percentages are important
        - Financial symbols and tickers
        - Company names and financial terms
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep important financial symbols and numbers
        # Remove special characters but keep $, %, ., -
        text = re.sub(r'[^\w\s$%.\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_entities(self, text: str) -> List[str]:
        """
        Extract financial entities from text
        
        Educational: Financial entities include:
        - Stock tickers (AAPL, GOOGL)
        - Company names
        - Financial instruments
        - Market indices
        """
        entities = []
        
        # Stock tickers (uppercase letters, 1-5 characters)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        entities.extend(tickers)
        
        # Common financial terms
        financial_terms = [
            'fed', 'federal reserve', 'ecb', 'bank of japan', 'boe',
            's&p 500', 'nasdaq', 'dow jones', 'ftse', 'dax', 'nikkei',
            'bitcoin', 'ethereum', 'crypto', 'blockchain',
            'treasury', 'bond', 'yield', 'inflation', 'gdp', 'cpi'
        ]
        
        for term in financial_terms:
            if term in text.lower():
                entities.append(term)
        
        return list(set(entities))
    
    def calculate_financial_sentiment(self, text: str) -> float:
        """
        Calculate sentiment using financial-specific keywords
        
        Educational: This method uses domain-specific knowledge
        to improve sentiment accuracy for financial text.
        """
        words = word_tokenize(text.lower())
        sentiment_score = 0.0
        word_count = 0
        
        for word in words:
            if word in self.financial_keywords:
                sentiment_score += self.financial_keywords[word]
                word_count += 1
        
        # Normalize by word count
        if word_count > 0:
            sentiment_score = sentiment_score / word_count
        
        return sentiment_score
    
    def calculate_emotions(self, text: str) -> Dict[str, float]:
        """
        Calculate emotional content of text
        
        Educational: Market emotions are key drivers of short-term price movements.
        Fear and greed are the most powerful emotions in trading.
        """
        words = word_tokenize(text.lower())
        emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            emotion_score = 0.0
            keyword_count = 0
            
            for word in words:
                for keyword, weight in keywords.items():
                    if keyword in word:
                        emotion_score += weight
                        keyword_count += 1
            
            # Normalize
            if keyword_count > 0:
                emotions[emotion] = emotion_score / keyword_count
            else:
                emotions[emotion] = 0.0
        
        return emotions
    
    def analyze_vader_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
        
        Educational: VADER is specifically tuned for social media text and
        works well for financial news and social media sentiment.
        """
        if not self.vader_analyzer:
            return 0.0, 0.0
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            # Compound score ranges from -1 to 1
            compound_score = scores['compound']
            confidence = abs(compound_score)  # Higher absolute score = higher confidence
            return compound_score, confidence
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return 0.0, 0.0
    
    def analyze_textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using TextBlob
        
        Educational: TextBlob provides a simple API for sentiment analysis
        and can be used as a baseline comparison.
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Use subjectivity as confidence measure
            confidence = 1.0 - subjectivity  # More objective = higher confidence
            
            return polarity, confidence
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return 0.0, 0.0
    
    def ensemble_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Combine multiple sentiment analysis methods
        
        Educational: Ensemble methods often provide more robust results
        by combining the strengths of different approaches.
        """
        # Get sentiment from different methods
        financial_score = self.calculate_financial_sentiment(text)
        vader_score, vader_conf = self.analyze_vader_sentiment(text)
        textblob_score, textblob_conf = self.analyze_textblob_sentiment(text)
        
        # Weight the scores (financial keywords get highest weight)
        weights = {
            'financial': 0.5,
            'vader': 0.3,
            'textblob': 0.2
        }
        
        ensemble_score = (
            financial_score * weights['financial'] +
            vader_score * weights['vader'] +
            textblob_score * weights['textblob']
        )
        
        # Calculate confidence based on agreement between methods
        scores = [financial_score, vader_score, textblob_score]
        agreement = 1.0 - np.std(scores)  # Lower std = higher agreement
        confidence = min(agreement, 1.0)
        
        return ensemble_score, confidence
    
    def classify_sentiment(self, score: float) -> SentimentLabel:
        """
        Classify sentiment score into discrete labels
        """
        if score <= -0.6:
            return SentimentLabel.VERY_BEARISH
        elif score <= -0.2:
            return SentimentLabel.BEARISH
        elif score <= 0.2:
            return SentimentLabel.NEUTRAL
        elif score <= 0.6:
            return SentimentLabel.BULLISH
        else:
            return SentimentLabel.VERY_BULLISH
    
    def calculate_market_impact(self, sentiment_score: float, confidence: float, 
                              source: str, volume: int = 1) -> float:
        """
        Calculate potential market impact of sentiment
        
        Educational: Market impact depends on:
        - Sentiment strength
        - Confidence in analysis
        - Source credibility
        - Volume/Reach of the information
        """
        source_weights = {
            'reuters': 0.9,
            'bloomberg': 0.9,
            'wsj': 0.8,
            'cnbc': 0.7,
            'twitter': 0.5,
            'reddit': 0.4,
            'analyst': 0.8,
            'sec_filing': 0.9,
            'press_release': 0.7
        }
        
        source_weight = source_weights.get(source.lower(), 0.5)
        
        # Calculate impact score
        impact = abs(sentiment_score) * confidence * source_weight * min(volume / 1000, 1.0)
        
        return min(impact, 1.0)
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> List[Tuple[str, float]]:
        """
        Extract topics from a collection of texts using LDA
        
        Educational: Topic modeling helps identify what the market is talking about,
        which can be as important as sentiment.
        """
        if len(texts) < num_topics:
            return []
        
        try:
            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Create LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            
            # Fit LDA model
            self.lda_model.fit(tfidf_matrix)
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = topic.sum()
                
                topics.append((" ".join(top_words[:3]), topic_weight))
            
            # Sort by weight
            topics.sort(key=lambda x: x[1], reverse=True)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            return []
    
    def analyze_single_text(self, text: str, source: str = "unknown", 
                           timestamp: Optional[datetime] = None) -> SentimentResult:
        """
        Analyze sentiment of a single text
        """
        if not timestamp:
            timestamp = datetime.now()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Calculate sentiment
        sentiment_score, confidence = self.ensemble_sentiment(processed_text)
        sentiment_label = self.classify_sentiment(sentiment_score)
        
        # Extract additional information
        emotions = self.calculate_emotions(processed_text)
        financial_entities = self.extract_financial_entities(text)
        
        # Calculate market impact
        market_impact = self.calculate_market_impact(
            sentiment_score, confidence, source
        )
        
        return SentimentResult(
            text=text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
            topics=[],
            emotions=emotions,
            financial_entities=financial_entities,
            market_impact_score=market_impact
        )
    
    async def analyze_batch(self, texts: List[Dict[str, Any]]) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts asynchronously
        """
        results = []
        
        for item in texts:
            text = item.get('text', '')
            source = item.get('source', 'unknown')
            timestamp = item.get('timestamp', datetime.now())
            
            result = self.analyze_single_text(text, source, timestamp)
            results.append(result)
        
        return results
    
    def aggregate_market_sentiment(self, results: List[SentimentResult], 
                                 time_window: timedelta = timedelta(hours=24)) -> MarketSentiment:
        """
        Aggregate individual sentiment results into market sentiment
        
        Educational: Market sentiment is the collective emotional state
        of all market participants at a given time.
        """
        if not results:
            return MarketSentiment(
                overall_sentiment=0.0,
                sentiment_distribution={},
                sentiment_trend=[],
                volume_weighted_sentiment=0.0,
                source_breakdown={},
                key_topics=[],
                emotional_state={},
                timestamp=datetime.now()
            )
        
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_results = [r for r in results if r.timestamp >= cutoff_time]
        
        if not recent_results:
            recent_results = results  # Fallback to all results
        
        # Calculate overall sentiment
        sentiment_scores = [r.sentiment_score for r in recent_results]
        overall_sentiment = np.mean(sentiment_scores)
        
        # Calculate sentiment distribution
        sentiment_counts = {}
        for label in SentimentLabel:
            sentiment_counts[label] = sum(1 for r in recent_results if r.sentiment_label == label)
        
        total = len(recent_results)
        sentiment_distribution = {
            label: count / total for label, count in sentiment_counts.items()
        }
        
        # Calculate sentiment trend (last 10 results)
        recent_scores = sentiment_scores[-10:] if len(sentiment_scores) >= 10 else sentiment_scores
        sentiment_trend = recent_scores
        
        # Calculate volume-weighted sentiment
        if recent_results:
            volume_weighted_sentiment = np.average(
                [r.sentiment_score for r in recent_results],
                weights=[r.market_impact_score for r in recent_results]
            )
        else:
            volume_weighted_sentiment = 0.0
        
        # Source breakdown
        source_sentiments = {}
        for result in recent_results:
            if result.source not in source_sentiments:
                source_sentiments[result.source] = []
            source_sentiments[result.source].append(result.sentiment_score)
        
        source_breakdown = {
            source: np.mean(scores) for source, scores in source_sentiments.items()
        }
        
        # Extract topics from all texts
        all_texts = [r.text for r in recent_results]
        key_topics = self.extract_topics(all_texts, num_topics=5)
        
        # Aggregate emotions
        all_emotions = {}
        for result in recent_results:
            for emotion, score in result.emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)
        
        emotional_state = {
            emotion: np.mean(scores) for emotion, scores in all_emotions.items()
        }
        
        return MarketSentiment(
            overall_sentiment=overall_sentiment,
            sentiment_distribution=sentiment_distribution,
            sentiment_trend=sentiment_trend,
            volume_weighted_sentiment=volume_weighted_sentiment,
            source_breakdown=source_breakdown,
            key_topics=key_topics,
            emotional_state=emotional_state,
            timestamp=datetime.now()
        )
    
    def generate_sentiment_signals(self, market_sentiment: MarketSentiment) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment analysis
        
        Educational: Sentiment signals should be used as confirmation
        rather than primary trading signals.
        """
        signals = {
            'primary_signal': 'hold',
            'confidence': 0.0,
            'reasoning': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Primary signal based on overall sentiment
        if market_sentiment.overall_sentiment > 0.3:
            signals['primary_signal'] = 'bullish'
            signals['confidence'] = min(abs(market_sentiment.overall_sentiment), 1.0)
            signals['reasoning'].append(f"Positive market sentiment: {market_sentiment.overall_sentiment:.2f}")
        elif market_sentiment.overall_sentiment < -0.3:
            signals['primary_signal'] = 'bearish'
            signals['confidence'] = min(abs(market_sentiment.overall_sentiment), 1.0)
            signals['reasoning'].append(f"Negative market sentiment: {market_sentiment.overall_sentiment:.2f}")
        
        # Check for extreme emotions
        for emotion, score in market_sentiment.emotional_state.items():
            if score > 0.7:
                if emotion in ['fear', 'panic']:
                    signals['risk_factors'].append(f"High {emotion} detected: {score:.2f}")
                elif emotion in ['euphoria', 'greed']:
                    signals['risk_factors'].append(f"High {emotion} detected: {score:.2f}")
        
        # Check sentiment trend
        if len(market_sentiment.sentiment_trend) >= 3:
            recent_trend = market_sentiment.sentiment_trend[-3:]
            if all(x > y for x, y in zip(recent_trend[1:], recent_trend[:-1])):
                signals['opportunities'].append("Improving sentiment trend")
            elif all(x < y for x, y in zip(recent_trend[1:], recent_trend[:-1])):
                signals['risk_factors'].append("Deteriorating sentiment trend")
        
        # Source credibility check
        credible_sources = ['reuters', 'bloomberg', 'wsj', 'sec_filing']
        credible_sentiment = [
            score for source, score in market_sentiment.source_breakdown.items()
            if source in credible_sources
        ]
        
        if credible_sentiment:
            avg_credible_sentiment = np.mean(credible_sentiment)
            if abs(avg_credible_sentiment) > 0.4:
                signals['confidence'] += 0.2
                signals['reasoning'].append("Strong sentiment from credible sources")
        
        # Cap confidence at 1.0
        signals['confidence'] = min(signals['confidence'], 1.0)
        
        return signals

# Educational: Usage Examples
"""
Educational Usage Examples:

1. Single News Analysis:
   analyzer = FinancialSentimentAnalyzer()
   result = analyzer.analyze_single_text(
       "Apple beats earnings expectations, stock rallies 5%",
       source="reuters"
   )
   print(f"Sentiment: {result.sentiment_label} (Score: {result.sentiment_score:.2f})")

2. Social Media Sentiment:
   tweets = [
       {"text": "Bullish on $TSLA! 🚀🚀🚀", "source": "twitter"},
       {"text": "Worried about market crash", "source": "reddit"}
   ]
   results = await analyzer.analyze_batch(tweets)

3. Market Sentiment Aggregation:
   market_sentiment = analyzer.aggregate_market_sentiment(results)
   signals = analyzer.generate_sentiment_signals(market_sentiment)

4. Real-time Monitoring:
   while True:
       news = get_latest_news()
       results = await analyzer.analyze_batch(news)
       market_sentiment = analyzer.aggregate_market_sentiment(results)
       
       if market_sentiment.overall_sentiment > 0.5:
           print("Strong bullish sentiment detected")
       elif market_sentiment.overall_sentiment < -0.5:
           print("Strong bearish sentiment detected")

5. Emotion Analysis:
   emotions = market_sentiment.emotional_state
   if emotions['fear'] > 0.7:
       print("High fear detected - potential buying opportunity")
   if emotions['greed'] > 0.7:
       print("High greed detected - potential warning sign")

Key Insights:
- Sentiment analysis works best when combined with technical analysis
- Extreme sentiment often signals potential reversals
- Source credibility is crucial for weighing sentiment
- Volume and reach amplify sentiment impact
- Time decay reduces the impact of older sentiment
"""