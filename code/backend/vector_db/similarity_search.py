"""
Similarity Search Module

This module provides advanced similarity search capabilities for finding
similar market patterns, news events, and portfolio states.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from backend.vector_db.vector_store import PineconeVectorStore, InMemoryVectorStore, VectorEntry, VectorMetadata
from backend.vector_db.embeddings import EmbeddingGenerator, PricePattern, NewsPattern, PortfolioState
from backend.core.config import settings
from backend.core.exceptions import ValidationError, DataCollectionError
from backend.core.logging import get_logger

logger = get_logger(__name__)


class SearchType(Enum):
    """Types of similarity search."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    HYBRID = "hybrid"


class SimilarityResult:
    """Result of similarity search with additional metadata."""
    
    def __init__(
        self,
        vector_id: str,
        score: float,
        metadata: Dict[str, Any],
        vector: Optional[List[float]] = None,
        explanation: Optional[str] = None
    ):
        self.vector_id = vector_id
        self.score = score
        self.metadata = metadata
        self.vector = vector
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_id": self.vector_id,
            "score": self.score,
            "metadata": self.metadata,
            "vector": self.vector,
            "explanation": self.explanation
        }


@dataclass
class SearchQuery:
    """Search query with parameters."""
    query_vector: List[float]
    search_type: SearchType = SearchType.APPROXIMATE
    top_k: int = 10
    threshold: float = 0.5
    filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    include_vectors: bool = False
    
    def __post_init__(self):
        """Validate search query."""
        if not self.query_vector:
            raise ValidationError("Query vector cannot be empty", "similarity_search")
        if self.top_k <= 0:
            raise ValidationError("Top_k must be positive", "similarity_search")
        if not 0 <= self.threshold <= 1:
            raise ValidationError("Threshold must be between 0 and 1", "similarity_search")


class PatternMatcher:
    """
    Advanced pattern matching for market data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("pattern_matcher")
        
        # Pattern matching parameters
        self.min_pattern_length = config.get("min_pattern_length", 10)
        self.max_pattern_length = config.get("max_pattern_length", 100)
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        
    async def find_similar_price_patterns(
        self,
        query_pattern: PricePattern,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        embedding_generator: EmbeddingGenerator,
        top_k: int = 10,
        time_window: Optional[timedelta] = None
    ) -> List[SimilarityResult]:
        """
        Find similar price patterns in the vector store.
        
        Args:
            query_pattern: Query price pattern
            vector_store: Vector store to search in
            embedding_generator: Embedding generator
            top_k: Number of results to return
            time_window: Time window for search
            
        Returns:
            List of similar patterns
        """
        try:
            # Generate embedding for query pattern
            query_embedding = await embedding_generator.generate_price_pattern_embedding(query_pattern)
            
            # Prepare filters
            filters = {
                "data_type": "price_pattern",
                "symbol": query_pattern.symbol
            }
            
            # Add time window filter if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                filters["timestamp"] = {"$gte": cutoff_time.isoformat()}
            
            # Search in vector store
            search_results = await vector_store.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filters,
                include_metadata=True
            )
            
            # Convert to SimilarityResult objects
            similarity_results = []
            for result in search_results:
                if result["score"] >= self.similarity_threshold:
                    similarity_result = SimilarityResult(
                        vector_id=result["id"],
                        score=result["score"],
                        metadata=result.get("metadata", {}),
                        vector=result.get("vector"),
                        explanation=f"Price pattern similarity: {result['score']:.3f}"
                    )
                    similarity_results.append(similarity_result)
            
            self.logger.info(f"Found {len(similarity_results)} similar price patterns")
            return similarity_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar price patterns: {e}")
            raise DataCollectionError(f"Failed to find similar price patterns: {e}", "pattern_matcher")
    
    async def find_similar_news_events(
        self,
        query_news: NewsPattern,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        embedding_generator: EmbeddingGenerator,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Find similar news events in the vector store.
        
        Args:
            query_news: Query news pattern
            vector_store: Vector store to search in
            embedding_generator: Embedding generator
            top_k: Number of results to return
            sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')
            
        Returns:
            List of similar news events
        """
        try:
            # Generate embedding for query news
            query_embedding = await embedding_generator.generate_news_pattern_embedding(query_news)
            
            # Prepare filters
            filters = {
                "data_type": "news_sentiment"
            }
            
            # Add symbol filter if specified
            if query_news.symbol:
                filters["symbol"] = query_news.symbol
            
            # Add sentiment filter if specified
            if sentiment_filter:
                if sentiment_filter == "positive":
                    filters["sentiment_score"] = {"$gte": 0.1}
                elif sentiment_filter == "negative":
                    filters["sentiment_score"] = {"$lte": -0.1}
                else:
                    filters["sentiment_score"] = {"$gt": -0.1, "$lt": 0.1}
            
            # Search in vector store
            search_results = await vector_store.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filters,
                include_metadata=True
            )
            
            # Convert to SimilarityResult objects
            similarity_results = []
            for result in search_results:
                if result["score"] >= self.similarity_threshold:
                    similarity_result = SimilarityResult(
                        vector_id=result["id"],
                        score=result["score"],
                        metadata=result.get("metadata", {}),
                        vector=result.get("vector"),
                        explanation=f"News sentiment similarity: {result['score']:.3f}"
                    )
                    similarity_results.append(similarity_result)
            
            self.logger.info(f"Found {len(similarity_results)} similar news events")
            return similarity_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar news events: {e}")
            raise DataCollectionError(f"Failed to find similar news events: {e}", "pattern_matcher")
    
    async def find_similar_portfolio_states(
        self,
        query_portfolio: PortfolioState,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        embedding_generator: EmbeddingGenerator,
        top_k: int = 10,
        risk_level_filter: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Find similar portfolio states in the vector store.
        
        Args:
            query_portfolio: Query portfolio state
            vector_store: Vector store to search in
            embedding_generator: Embedding generator
            top_k: Number of results to return
            risk_level_filter: Filter by risk level
            
        Returns:
            List of similar portfolio states
        """
        try:
            # Generate embedding for query portfolio
            query_embedding = await embedding_generator.generate_portfolio_embedding(query_portfolio)
            
            # Prepare filters
            filters = {
                "data_type": "portfolio_state"
            }
            
            # Add risk level filter if specified
            if risk_level_filter:
                filters["regime"] = risk_level_filter
            
            # Search in vector store
            search_results = await vector_store.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=filters,
                include_metadata=True
            )
            
            # Convert to SimilarityResult objects
            similarity_results = []
            for result in search_results:
                if result["score"] >= self.similarity_threshold:
                    similarity_result = SimilarityResult(
                        vector_id=result["id"],
                        score=result["score"],
                        metadata=result.get("metadata", {}),
                        vector=result.get("vector"),
                        explanation=f"Portfolio state similarity: {result['score']:.3f}"
                    )
                    similarity_results.append(similarity_result)
            
            self.logger.info(f"Found {len(similarity_results)} similar portfolio states")
            return similarity_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar portfolio states: {e}")
            raise DataCollectionError(f"Failed to find similar portfolio states: {e}", "pattern_matcher")


class MarketRegimeDetector:
    """
    Detects market regimes using clustering on historical patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("market_regime_detector")
        
        # Clustering parameters
        self.n_regimes = config.get("n_regimes", 5)
        self.regime_labels = ["bull_market", "bear_market", "sideways", "volatile", "transition"]
        
    async def detect_market_regime(
        self,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        symbol: str,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        Detect current market regime for a symbol.
        
        Args:
            vector_store: Vector store to analyze
            symbol: Symbol to analyze
            time_window: Time window for analysis
            
        Returns:
            Market regime information
        """
        try:
            # Get recent patterns for the symbol
            cutoff_time = datetime.now() - time_window
            
            # This is a simplified implementation
            # In practice, you'd query the vector store for recent patterns
            # and perform clustering to identify regimes
            
            # Mock regime detection
            regime_scores = {
                "bull_market": 0.3,
                "bear_market": 0.1,
                "sideways": 0.4,
                "volatile": 0.15,
                "transition": 0.05
            }
            
            # Determine dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[dominant_regime]
            
            return {
                "symbol": symbol,
                "dominant_regime": dominant_regime,
                "confidence": confidence,
                "regime_scores": regime_scores,
                "analysis_period": time_window.days,
                "detected_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return {
                "symbol": symbol,
                "dominant_regime": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def get_historical_regime_analysis(
        self,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        symbol: str,
        period: timedelta = timedelta(days=365)
    ) -> List[Dict[str, Any]]:
        """
        Get historical regime analysis for a symbol.
        
        Args:
            vector_store: Vector store to analyze
            symbol: Symbol to analyze
            period: Analysis period
            
        Returns:
            List of historical regime data points
        """
        try:
            # This is a simplified implementation
            # In practice, you'd analyze historical patterns and regime changes
            
            historical_data = []
            current_date = datetime.now() - period
            
            while current_date <= datetime.now():
                # Mock historical regime data
                regime = np.random.choice(self.regime_labels)
                confidence = np.random.uniform(0.5, 0.9)
                
                historical_data.append({
                    "date": current_date.isoformat(),
                    "regime": regime,
                    "confidence": confidence
                })
                
                current_date += timedelta(days=1)
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical regime analysis: {e}")
            return []


class SimilaritySearchEngine:
    """
    Main similarity search engine that orchestrates all search functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("similarity_search_engine")
        self.is_initialized = False
        
        # Initialize components
        self.pattern_matcher = PatternMatcher(config.get("pattern_matching", {}))
        self.regime_detector = MarketRegimeDetector(config.get("regime_detection", {}))
        
    async def initialize(self) -> None:
        """Initialize the similarity search engine."""
        try:
            self.is_initialized = True
            self.logger.info("Similarity search engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize similarity search engine: {e}")
            raise DataCollectionError(f"Similarity search engine initialization failed: {e}", "similarity_search_engine")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.is_initialized = False
        self.logger.info("Similarity search engine cleaned up")
    
    async def search_similar_patterns(
        self,
        query: SearchQuery,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore]
    ) -> List[SimilarityResult]:
        """
        Perform similarity search with the given query.
        
        Args:
            query: Search query
            vector_store: Vector store to search in
            
        Returns:
            List of similarity results
        """
        if not self.is_initialized:
            raise ValidationError("Similarity search engine not initialized", "similarity_search_engine")
        
        try:
            # Perform search based on type
            if query.search_type == SearchType.EXACT:
                results = await self._exact_search(query, vector_store)
            elif query.search_type == SearchType.APPROXIMATE:
                results = await self._approximate_search(query, vector_store)
            else:  # HYBRID
                results = await self._hybrid_search(query, vector_store)
            
            # Filter by threshold
            filtered_results = [
                result for result in results
                if result.score >= query.threshold
            ]
            
            # Sort by score (descending)
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top_k results
            return filtered_results[:query.top_k]
            
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {e}")
            raise DataCollectionError(f"Failed to perform similarity search: {e}", "similarity_search_engine")
    
    async def _exact_search(
        self,
        query: SearchQuery,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore]
    ) -> List[SimilarityResult]:
        """Perform exact similarity search."""
        search_results = await vector_store.search_similar(
            query_vector=query.query_vector,
            top_k=query.top_k,
            filter_dict=query.filters,
            include_metadata=query.include_vectors
        )
        
        similarity_results = []
        for result in search_results:
            similarity_result = SimilarityResult(
                vector_id=result["id"],
                score=result["score"],
                metadata=result.get("metadata", {}),
                vector=result.get("vector"),
                explanation="Exact similarity search"
            )
            similarity_results.append(similarity_result)
        
        return similarity_results
    
    async def _approximate_search(
        self,
        query: SearchQuery,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore]
    ) -> List[SimilarityResult]:
        """Perform approximate similarity search."""
        # For now, use the same implementation as exact search
        # In practice, you might use different algorithms or parameters
        return await self._exact_search(query, vector_store)
    
    async def _hybrid_search(
        self,
        query: SearchQuery,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore]
    ) -> List[SimilarityResult]:
        """Perform hybrid similarity search."""
        # Combine exact and approximate search results
        exact_results = await self._exact_search(query, vector_store)
        approximate_results = await self._approximate_search(query, vector_store)
        
        # Merge and deduplicate results
        all_results = exact_results + approximate_results
        seen_ids = set()
        merged_results = []
        
        for result in all_results:
            if result.vector_id not in seen_ids:
                seen_ids.add(result.vector_id)
                merged_results.append(result)
        
        return merged_results
    
    async def find_analogous_patterns(
        self,
        query_vector: List[float],
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore],
        outcome_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[SimilarityResult]:
        """
        Find historical patterns that led to specific outcomes.
        
        Args:
            query_vector: Query vector
            vector_store: Vector store to search in
            outcome_filter: Filter by outcome ('profitable', 'loss', 'breakeven')
            top_k: Number of results to return
            
        Returns:
            List of analogous patterns with outcomes
        """
        try:
            # Prepare filters
            filters = {}
            
            if outcome_filter:
                if outcome_filter == "profitable":
                    filters["outcome"] = {"$gt": 0}
                elif outcome_filter == "loss":
                    filters["outcome"] = {"$lt": 0}
                else:
                    filters["outcome"] = {"$eq": 0}
            
            # Search for similar patterns
            search_results = await vector_store.search_similar(
                query_vector=query_vector,
                top_k=top_k,
                filter_dict=filters,
                include_metadata=True
            )
            
            # Convert to SimilarityResult objects
            similarity_results = []
            for result in search_results:
                metadata = result.get("metadata", {})
                outcome = metadata.get("outcome", 0)
                
                explanation = f"Analogous pattern with outcome: {outcome:.2%}"
                if outcome > 0:
                    explanation += " (profitable)"
                elif outcome < 0:
                    explanation += " (loss)"
                else:
                    explanation += " (breakeven)"
                
                similarity_result = SimilarityResult(
                    vector_id=result["id"],
                    score=result["score"],
                    metadata=metadata,
                    vector=result.get("vector"),
                    explanation=explanation
                )
                similarity_results.append(similarity_result)
            
            self.logger.info(f"Found {len(similarity_results)} analogous patterns")
            return similarity_results
            
        except Exception as e:
            self.logger.error(f"Error finding analogous patterns: {e}")
            raise DataCollectionError(f"Failed to find analogous patterns: {e}", "similarity_search_engine")
    
    async def get_search_statistics(
        self,
        vector_store: Union[PineconeVectorStore, InMemoryVectorStore]
    ) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Args:
            vector_store: Vector store to analyze
            
        Returns:
            Search statistics
        """
        try:
            # Get index stats
            index_stats = await vector_store.get_index_stats()
            
            # Analyze data distribution
            stats = {
                "index_stats": index_stats,
                "data_types": {},
                "symbol_distribution": {},
                "temporal_distribution": {},
                "quality_distribution": {}
            }
            
            # This is a simplified implementation
            # In practice, you'd query the vector store to get detailed statistics
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting search statistics: {e}")
            return {}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get the current status of the similarity search engine."""
        return {
            "initialized": self.is_initialized,
            "pattern_matcher": {
                "similarity_threshold": self.pattern_matcher.similarity_threshold,
                "min_pattern_length": self.pattern_matcher.min_pattern_length,
                "max_pattern_length": self.pattern_matcher.max_pattern_length
            },
            "regime_detector": {
                "n_regimes": self.regime_detector.n_regimes,
                "regime_labels": self.regime_detector.regime_labels
            },
            "last_updated": datetime.now().isoformat()
        }