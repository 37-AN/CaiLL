"""
Vector Database Service

This service orchestrates the vector database functionality including
storage, embeddings generation, and similarity search for pattern recognition.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import uuid

from backend.vector_db.vector_store import PineconeVectorStore, InMemoryVectorStore, VectorEntry, VectorMetadata
from backend.vector_db.embeddings import EmbeddingGenerator, PricePattern, NewsPattern, PortfolioState
from backend.vector_db.similarity_search import SimilaritySearchEngine, SearchQuery, SearchType, SimilarityResult
from backend.data_collectors.base_collector import OHLCVData, NewsData
from backend.core.config import settings
from backend.core.exceptions import DataCollectionError, ValidationError
from backend.core.logging import get_logger, trading_logger

logger = get_logger(__name__)


@dataclass
class PatternStorageRequest:
    """Request to store a pattern in the vector database."""
    pattern_type: str  # 'price', 'news', 'portfolio', 'multimodal'
    symbol: str
    data: Any  # PricePattern, NewsPattern, PortfolioState, or dict
    outcome: Optional[float] = None
    regime: Optional[str] = None
    confidence: Optional[float] = None
    quality_score: float = 1.0
    source: str = "system"


class VectorDatabaseService:
    """
    Main vector database service that orchestrates all vector operations.
    
    This service provides a unified interface for storing market patterns,
    generating embeddings, and performing similarity search for pattern recognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("vector_database_service")
        self.is_initialized = False
        
        # Initialize components
        self.use_pinecone = config.get("use_pinecone", True) and bool(settings.PINECONE_API_KEY)
        
        if self.use_pinecone:
            self.vector_store = PineconeVectorStore(config.get("pinecone", {}))
        else:
            self.vector_store = InMemoryVectorStore(config.get("in_memory", {}).get("dimension", 384))
        
        self.embedding_generator = EmbeddingGenerator(config.get("embeddings", {}))
        self.similarity_engine = SimilaritySearchEngine(config.get("similarity_search", {}))
        
        # Storage statistics
        self.storage_stats = {
            "total_patterns": 0,
            "price_patterns": 0,
            "news_patterns": 0,
            "portfolio_patterns": 0,
            "multimodal_patterns": 0
        }
        
    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            self.logger.info("Initializing Vector Database Service...")
            
            # Initialize components
            await self.vector_store.initialize()
            await self.embedding_generator.initialize()
            await self.similarity_engine.initialize()
            
            # Update statistics
            await self._update_storage_stats()
            
            self.is_initialized = True
            self.logger.info("Vector Database Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vector Database Service: {e}")
            raise DataCollectionError(f"Vector Database Service initialization failed: {e}", "vector_database_service")
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        try:
            self.logger.info("Cleaning up Vector Database Service...")
            
            await self.vector_store.cleanup()
            await self.embedding_generator.cleanup()
            await self.similarity_engine.cleanup()
            
            self.is_initialized = False
            self.logger.info("Vector Database Service cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def store_price_pattern(
        self,
        symbol: str,
        price_data: List[OHLCVData],
        technical_indicators: Dict[str, List[float]],
        outcome: Optional[float] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Store a price pattern in the vector database.
        
        Args:
            symbol: Trading symbol
            price_data: List of OHLCV data
            technical_indicators: Technical indicators for the pattern
            outcome: Pattern outcome (e.g., return after pattern)
            regime: Market regime
            confidence: Pattern confidence
            
        Returns:
            Pattern ID
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Create price pattern
            price_pattern = PricePattern(
                symbol=symbol,
                timestamps=[data.timestamp for data in price_data],
                prices=[data.close for data in price_data],
                volumes=[data.volume for data in price_data],
                technical_indicators=technical_indicators
            )
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_price_pattern_embedding(price_pattern)
            
            # Create vector entry
            vector_id = str(uuid.uuid4())
            metadata = VectorMetadata(
                id=vector_id,
                timestamp=datetime.now(),
                symbol=symbol,
                data_type="price_pattern",
                source="price_collector",
                outcome=outcome,
                regime=regime,
                confidence=confidence
            )
            
            vector_entry = VectorEntry(vector=embedding, metadata=metadata)
            
            # Store in vector database
            stored_id = await self.vector_store.store_single_vector(vector_entry)
            
            # Update statistics
            self.storage_stats["price_patterns"] += 1
            self.storage_stats["total_patterns"] += 1
            
            trading_logger.system_event(
                event_type="pattern_stored",
                message=f"Stored price pattern for {symbol}",
                component="vector_database_service"
            )
            
            self.logger.info(f"Stored price pattern for {symbol} with ID: {stored_id}")
            return stored_id
            
        except Exception as e:
            self.logger.error(f"Error storing price pattern for {symbol}: {e}")
            raise DataCollectionError(f"Failed to store price pattern: {e}", "vector_database_service")
    
    async def store_news_pattern(
        self,
        news_data: List[NewsData],
        outcome: Optional[float] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Store a news pattern in the vector database.
        
        Args:
            news_data: List of news articles
            outcome: Pattern outcome
            regime: Market regime
            confidence: Pattern confidence
            
        Returns:
            Pattern ID
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Create news pattern
            symbol = news_data[0].symbol if news_data and news_data[0].symbol else None
            
            news_pattern = NewsPattern(
                symbol=symbol,
                headlines=[article.title for article in news_data],
                contents=[article.content for article in news_data],
                timestamps=[article.timestamp for article in news_data],
                sentiment_scores=[article.sentiment_score or 0 for article in news_data]
            )
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_news_pattern_embedding(news_pattern)
            
            # Create vector entry
            vector_id = str(uuid.uuid4())
            metadata = VectorMetadata(
                id=vector_id,
                timestamp=datetime.now(),
                symbol=symbol or "general",
                data_type="news_sentiment",
                source="news_collector",
                outcome=outcome,
                regime=regime,
                confidence=confidence
            )
            
            vector_entry = VectorEntry(vector=embedding, metadata=metadata)
            
            # Store in vector database
            stored_id = await self.vector_store.store_single_vector(vector_entry)
            
            # Update statistics
            self.storage_stats["news_patterns"] += 1
            self.storage_stats["total_patterns"] += 1
            
            self.logger.info(f"Stored news pattern with ID: {stored_id}")
            return stored_id
            
        except Exception as e:
            self.logger.error(f"Error storing news pattern: {e}")
            raise DataCollectionError(f"Failed to store news pattern: {e}", "vector_database_service")
    
    async def store_portfolio_pattern(
        self,
        portfolio_state: PortfolioState,
        outcome: Optional[float] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Store a portfolio pattern in the vector database.
        
        Args:
            portfolio_state: Portfolio state data
            outcome: Pattern outcome
            regime: Market regime
            confidence: Pattern confidence
            
        Returns:
            Pattern ID
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Generate embedding
            embedding = await self.embedding_generator.generate_portfolio_embedding(portfolio_state)
            
            # Create vector entry
            vector_id = str(uuid.uuid4())
            metadata = VectorMetadata(
                id=vector_id,
                timestamp=portfolio_state.timestamp,
                symbol="portfolio",
                data_type="portfolio_state",
                source="portfolio_tracker",
                outcome=outcome,
                regime=regime,
                confidence=confidence
            )
            
            vector_entry = VectorEntry(vector=embedding, metadata=metadata)
            
            # Store in vector database
            stored_id = await self.vector_store.store_single_vector(vector_entry)
            
            # Update statistics
            self.storage_stats["portfolio_patterns"] += 1
            self.storage_stats["total_patterns"] += 1
            
            self.logger.info(f"Stored portfolio pattern with ID: {stored_id}")
            return stored_id
            
        except Exception as e:
            self.logger.error(f"Error storing portfolio pattern: {e}")
            raise DataCollectionError(f"Failed to store portfolio pattern: {e}", "vector_database_service")
    
    async def store_multimodal_pattern(
        self,
        request: PatternStorageRequest
    ) -> str:
        """
        Store a multimodal pattern combining multiple data types.
        
        Args:
            request: Pattern storage request
            
        Returns:
            Pattern ID
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Extract pattern data
            price_pattern = None
            news_pattern = None
            portfolio_state = None
            
            if request.pattern_type == "multimodal":
                # Parse multimodal data
                if isinstance(request.data, dict):
                    if "price_pattern" in request.data:
                        price_pattern = request.data["price_pattern"]
                    if "news_pattern" in request.data:
                        news_pattern = request.data["news_pattern"]
                    if "portfolio_state" in request.data:
                        portfolio_state = request.data["portfolio_state"]
            
            # Generate multimodal embedding
            embedding = await self.embedding_generator.generate_multimodal_embedding(
                price_pattern=price_pattern,
                news_pattern=news_pattern,
                portfolio_state=portfolio_state
            )
            
            # Create vector entry
            vector_id = str(uuid.uuid4())
            metadata = VectorMetadata(
                id=vector_id,
                timestamp=datetime.now(),
                symbol=request.symbol,
                data_type="multimodal_pattern",
                source=request.source,
                outcome=request.outcome,
                regime=request.regime,
                confidence=request.confidence,
                quality_score=request.quality_score
            )
            
            vector_entry = VectorEntry(vector=embedding, metadata=metadata)
            
            # Store in vector database
            stored_id = await self.vector_store.store_single_vector(vector_entry)
            
            # Update statistics
            self.storage_stats["multimodal_patterns"] += 1
            self.storage_stats["total_patterns"] += 1
            
            self.logger.info(f"Stored multimodal pattern for {request.symbol} with ID: {stored_id}")
            return stored_id
            
        except Exception as e:
            self.logger.error(f"Error storing multimodal pattern: {e}")
            raise DataCollectionError(f"Failed to store multimodal pattern: {e}", "vector_database_service")
    
    async def find_similar_patterns(
        self,
        pattern_type: str,
        symbol: str,
        query_data: Any,
        top_k: int = 10,
        threshold: float = 0.5,
        time_window: Optional[timedelta] = None,
        outcome_filter: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Find similar patterns in the vector database.
        
        Args:
            pattern_type: Type of pattern to search for
            symbol: Symbol to search for
            query_data: Query data (pattern to match)
            top_k: Number of results to return
            threshold: Similarity threshold
            time_window: Time window for search
            outcome_filter: Filter by outcome
            
        Returns:
            List of similar patterns
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Generate embedding for query
            if pattern_type == "price":
                embedding = await self.embedding_generator.generate_price_pattern_embedding(query_data)
            elif pattern_type == "news":
                embedding = await self.embedding_generator.generate_news_pattern_embedding(query_data)
            elif pattern_type == "portfolio":
                embedding = await self.embedding_generator.generate_portfolio_embedding(query_data)
            else:
                raise ValidationError(f"Unsupported pattern type: {pattern_type}", "vector_database_service")
            
            # Prepare search query
            search_query = SearchQuery(
                query_vector=embedding,
                search_type=SearchType.APPROXIMATE,
                top_k=top_k,
                threshold=threshold,
                filters={
                    "data_type": f"{pattern_type}_pattern" if pattern_type != "portfolio" else "portfolio_state",
                    "symbol": symbol
                }
            )
            
            # Add time window filter
            if time_window:
                cutoff_time = datetime.now() - time_window
                search_query.filters["timestamp"] = {"$gte": cutoff_time.isoformat()}
            
            # Add outcome filter
            if outcome_filter:
                if outcome_filter == "profitable":
                    search_query.filters["outcome"] = {"$gt": 0}
                elif outcome_filter == "loss":
                    search_query.filters["outcome"] = {"$lt": 0}
                else:
                    search_query.filters["outcome"] = {"$eq": 0}
            
            # Perform search
            results = await self.similarity_engine.search_similar_patterns(
                query=search_query,
                vector_store=self.vector_store
            )
            
            self.logger.info(f"Found {len(results)} similar {pattern_type} patterns for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            raise DataCollectionError(f"Failed to find similar patterns: {e}", "vector_database_service")
    
    async def find_analogous_scenarios(
        self,
        symbol: str,
        current_conditions: Dict[str, Any],
        top_k: int = 10,
        look_for_outcome: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Find historical scenarios analogous to current conditions.
        
        Args:
            symbol: Symbol to analyze
            current_conditions: Current market conditions
            top_k: Number of results to return
            look_for_outcome: Look for specific outcomes
            
        Returns:
            List of analogous scenarios
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Create multimodal query from current conditions
            query_embedding = await self.embedding_generator.generate_multimodal_embedding(
                price_pattern=current_conditions.get("price_pattern"),
                news_pattern=current_conditions.get("news_pattern"),
                portfolio_state=current_conditions.get("portfolio_state")
            )
            
            # Find analogous patterns
            results = await self.similarity_engine.find_analogous_patterns(
                query_vector=query_embedding,
                vector_store=self.vector_store,
                outcome_filter=look_for_outcome,
                top_k=top_k
            )
            
            self.logger.info(f"Found {len(results)} analogous scenarios for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding analogous scenarios: {e}")
            raise DataCollectionError(f"Failed to find analogous scenarios: {e}", "vector_database_service")
    
    async def detect_market_regime(
        self,
        symbol: str,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        Detect current market regime for a symbol.
        
        Args:
            symbol: Symbol to analyze
            time_window: Time window for analysis
            
        Returns:
            Market regime information
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Detect market regime
            regime_info = await self.similarity_engine.regime_detector.detect_market_regime(
                vector_store=self.vector_store,
                symbol=symbol,
                time_window=time_window
            )
            
            self.logger.info(f"Detected market regime for {symbol}: {regime_info.get('dominant_regime', 'unknown')}")
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return {
                "symbol": symbol,
                "dominant_regime": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def update_pattern_outcome(
        self,
        pattern_id: str,
        outcome: float
    ) -> bool:
        """
        Update the outcome of a stored pattern.
        
        Args:
            pattern_id: Pattern ID
            outcome: Pattern outcome
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Update metadata
            success = await self.vector_store.update_vector_metadata(
                vector_id=pattern_id,
                metadata_updates={"outcome": outcome}
            )
            
            if success:
                self.logger.info(f"Updated outcome for pattern {pattern_id}: {outcome}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating pattern outcome: {e}")
            return False
    
    async def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern data or None if not found
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            return await self.vector_store.get_vector_by_id(pattern_id)
            
        except Exception as e:
            self.logger.error(f"Error retrieving pattern {pattern_id}: {e}")
            return None
    
    async def delete_patterns(
        self,
        pattern_ids: List[str]
    ) -> bool:
        """
        Delete patterns by IDs.
        
        Args:
            pattern_ids: List of pattern IDs to delete
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            success = await self.vector_store.delete_vectors(pattern_ids)
            
            if success:
                # Update statistics
                self.storage_stats["total_patterns"] -= len(pattern_ids)
                await self._update_storage_stats()
                
                self.logger.info(f"Deleted {len(pattern_ids)} patterns")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting patterns: {e}")
            return False
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Database statistics
        """
        if not self.is_initialized:
            raise ValidationError("Vector Database Service not initialized", "vector_database_service")
        
        try:
            # Get index statistics
            index_stats = await self.vector_store.get_index_stats()
            
            # Get search statistics
            search_stats = await self.similarity_engine.get_search_statistics(self.vector_store)
            
            # Combine all statistics
            database_stats = {
                "storage_stats": self.storage_stats,
                "index_stats": index_stats,
                "search_stats": search_stats,
                "service_status": self.get_service_status()
            }
            
            return database_stats
            
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def _update_storage_stats(self) -> None:
        """Update storage statistics from the vector store."""
        try:
            index_stats = await self.vector_store.get_index_stats()
            self.storage_stats["total_patterns"] = index_stats.get("vector_count", 0)
            
        except Exception as e:
            self.logger.error(f"Error updating storage stats: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the vector database service.
        
        Returns:
            Service status information
        """
        return {
            "initialized": self.is_initialized,
            "vector_store": {
                "type": "pinecone" if self.use_pinecone else "in_memory",
                "status": self.vector_store.get_store_status()
            },
            "embedding_generator": self.embedding_generator.get_generator_status(),
            "similarity_engine": self.similarity_engine.get_engine_status(),
            "storage_stats": self.storage_stats,
            "last_updated": datetime.now().isoformat()
        }