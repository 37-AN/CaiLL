"""
Vector Store Implementation

This module provides the interface for storing and retrieving high-dimensional
vectors in Pinecone for pattern recognition and similarity search.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import uuid

import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import settings
from backend.core.exceptions import DataCollectionError, ValidationError
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorMetadata:
    """Metadata for vector entries."""
    id: str
    timestamp: datetime
    symbol: str
    data_type: str  # 'price_pattern', 'news_sentiment', 'portfolio_state', etc.
    source: str
    quality_score: float = 1.0
    outcome: Optional[float] = None  # For trade outcomes
    regime: Optional[str] = None  # Market regime
    confidence: Optional[float] = None  # Model confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Pinecone."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "data_type": self.data_type,
            "source": self.source,
            "quality_score": self.quality_score,
            "outcome": self.outcome,
            "regime": self.regime,
            "confidence": self.confidence
        }


@dataclass
class VectorEntry:
    """Complete vector entry with data and metadata."""
    vector: List[float]
    metadata: VectorMetadata
    
    def __post_init__(self):
        """Validate vector entry."""
        if not self.vector:
            raise ValidationError("Vector cannot be empty", "vector_store")
        if len(self.vector) == 0:
            raise ValidationError("Vector dimension cannot be zero", "vector_store")
        if not self.metadata.symbol:
            raise ValidationError("Symbol is required", "vector_store")


class PineconeVectorStore:
    """
    Pinecone vector store implementation.
    
    This class provides methods to store, retrieve, and search vectors
    in Pinecone for pattern recognition and similarity search.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("pinecone_vector_store")
        self.index = None
        self.is_initialized = False
        
        # Pinecone configuration
        self.api_key = config.get("api_key", settings.PINECONE_API_KEY)
        self.environment = config.get("environment", settings.PINECONE_ENVIRONMENT)
        self.index_name = config.get("index_name", "trading-patterns")
        self.dimension = config.get("dimension", 384)  # Default for sentence transformers
        self.metric = config.get("metric", "cosine")
        
    async def initialize(self) -> None:
        """Initialize Pinecone connection and create index if needed."""
        try:
            if not self.api_key:
                raise ValidationError("Pinecone API key is required", "pinecone_vector_store")
            
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                self.logger.info(f"Creating new index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            
            # Wait for index to be ready
            while not pinecone.describe_index(self.index_name).status['ready']:
                await asyncio.sleep(1)
            
            self.is_initialized = True
            self.logger.info(f"Pinecone vector store initialized: {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {e}")
            raise DataCollectionError(f"Pinecone initialization failed: {e}", "pinecone_vector_store")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Pinecone doesn't require explicit cleanup
        self.is_initialized = False
        self.logger.info("Pinecone vector store cleaned up")
    
    async def store_vectors(
        self,
        vectors: List[VectorEntry],
        batch_size: int = 100
    ) -> List[str]:
        """
        Store multiple vectors in Pinecone.
        
        Args:
            vectors: List of vector entries to store
            batch_size: Number of vectors to store per batch
            
        Returns:
            List of vector IDs
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        if not vectors:
            return []
        
        try:
            stored_ids = []
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Prepare batch data
                batch_data = []
                for entry in batch:
                    batch_data.append({
                        "id": entry.metadata.id,
                        "values": entry.vector,
                        "metadata": entry.metadata.to_dict()
                    })
                
                # Upsert batch
                self.index.upsert(vectors=batch_data)
                stored_ids.extend([entry.metadata.id for entry in batch])
                
                self.logger.debug(f"Stored batch of {len(batch)} vectors")
            
            self.logger.info(f"Successfully stored {len(stored_ids)} vectors")
            return stored_ids
            
        except Exception as e:
            self.logger.error(f"Error storing vectors: {e}")
            raise DataCollectionError(f"Failed to store vectors: {e}", "pinecone_vector_store")
    
    async def store_single_vector(self, vector_entry: VectorEntry) -> str:
        """
        Store a single vector in Pinecone.
        
        Args:
            vector_entry: Vector entry to store
            
        Returns:
            Vector ID
        """
        return (await self.store_vectors([vector_entry]))[0]
    
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar vectors with scores
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            # Prepare query
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Execute search
            results = self.index.query(**query_params)
            
            # Process results
            processed_results = []
            for match in results['matches']:
                processed_result = {
                    "id": match['id'],
                    "score": match['score'],
                    "vector": match.get('values', [])
                }
                
                if include_metadata and 'metadata' in match:
                    processed_result["metadata"] = match['metadata']
                
                processed_results.append(processed_result)
            
            self.logger.info(f"Found {len(processed_results)} similar vectors")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error searching similar vectors: {e}")
            raise DataCollectionError(f"Failed to search vectors: {e}", "pinecone_vector_store")
    
    async def get_vector_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: Vector ID to retrieve
            
        Returns:
            Vector data or None if not found
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            result = self.index.fetch(ids=[vector_id])
            
            if vector_id in result['vectors']:
                vector_data = result['vectors'][vector_id]
                return {
                    "id": vector_data['id'],
                    "vector": vector_data['values'],
                    "metadata": vector_data.get('metadata', {})
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching vector {vector_id}: {e}")
            return None
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            self.index.delete(ids=vector_ids)
            self.logger.info(f"Deleted {len(vector_ids)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting vectors: {e}")
            return False
    
    async def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete vectors by metadata filter.
        
        Args:
            filter_dict: Metadata filter
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            self.index.delete(filter=filter_dict)
            self.logger.info(f"Deleted vectors matching filter: {filter_dict}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting vectors by filter: {e}")
            return False
    
    async def update_vector_metadata(
        self,
        vector_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            vector_id: Vector ID
            metadata_updates: Metadata updates
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            # Get current vector
            current_vector = await self.get_vector_by_id(vector_id)
            if not current_vector:
                return False
            
            # Update metadata
            updated_metadata = current_vector["metadata"].copy()
            updated_metadata.update(metadata_updates)
            
            # Re-store with updated metadata
            updated_entry = VectorEntry(
                vector=current_vector["vector"],
                metadata=VectorMetadata(**updated_metadata)
            )
            
            await self.store_single_vector(updated_entry)
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating vector metadata: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "dimension": self.dimension,
                "metric": self.metric,
                "vector_count": stats.get('totalVectorCount', 0),
                "index_fullness": stats.get('indexFullness', 0),
                "namespaces": stats.get('namespaces', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {}
    
    async def create_namespace(self, namespace: str) -> bool:
        """
        Create a new namespace for organizing vectors.
        
        Args:
            namespace: Namespace name
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            # Namespaces are created automatically when vectors are added
            self.logger.info(f"Namespace {namespace} ready for use")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating namespace: {e}")
            return False
    
    async def search_in_namespace(
        self,
        query_vector: List[float],
        namespace: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a specific namespace.
        
        Args:
            query_vector: Query vector
            namespace: Namespace to search in
            top_k: Number of results to return
            filter_dict: Metadata filter
            
        Returns:
            List of similar vectors with scores
        """
        if not self.is_initialized:
            raise ValidationError("Vector store not initialized", "pinecone_vector_store")
        
        try:
            # Prepare query with namespace
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Execute search
            results = self.index.query(**query_params)
            
            # Process results
            processed_results = []
            for match in results['matches']:
                processed_result = {
                    "id": match['id'],
                    "score": match['score'],
                    "vector": match.get('values', []),
                    "namespace": namespace
                }
                
                if 'metadata' in match:
                    processed_result["metadata"] = match['metadata']
                
                processed_results.append(processed_result)
            
            self.logger.info(f"Found {len(processed_results)} similar vectors in namespace {namespace}")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error searching in namespace {namespace}: {e}")
            raise DataCollectionError(f"Failed to search in namespace: {e}", "pinecone_vector_store")
    
    async def backup_index(self, backup_name: str) -> bool:
        """
        Create a backup of the index (if supported by plan).
        
        Args:
            backup_name: Name for the backup
            
        Returns:
            True if successful
        """
        # This is a placeholder - actual backup implementation depends on Pinecone plan
        self.logger.info(f"Backup functionality not available in current plan")
        return False
    
    def get_store_status(self) -> Dict[str, Any]:
        """
        Get the current status of the vector store.
        
        Returns:
            Store status information
        """
        return {
            "initialized": self.is_initialized,
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "api_key_configured": bool(self.api_key),
            "environment_configured": bool(self.environment),
            "last_updated": datetime.now().isoformat()
        }


class InMemoryVectorStore:
    """
    In-memory vector store for testing and development.
    
    This class provides a simple in-memory implementation of the vector store
    interface for testing purposes.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: Dict[str, VectorEntry] = {}
        self.logger = get_logger("in_memory_vector_store")
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the in-memory store."""
        self.is_initialized = True
        self.logger.info("In-memory vector store initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.vectors.clear()
        self.is_initialized = False
        self.logger.info("In-memory vector store cleaned up")
    
    async def store_vectors(self, vectors: List[VectorEntry], batch_size: int = 100) -> List[str]:
        """Store vectors in memory."""
        stored_ids = []
        for entry in vectors:
            self.vectors[entry.metadata.id] = entry
            stored_ids.append(entry.metadata.id)
        
        self.logger.info(f"Stored {len(stored_ids)} vectors in memory")
        return stored_ids
    
    async def store_single_vector(self, vector_entry: VectorEntry) -> str:
        """Store a single vector in memory."""
        self.vectors[vector_entry.metadata.id] = vector_entry
        return vector_entry.metadata.id
    
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity."""
        if len(query_vector) != self.dimension:
            raise ValidationError(f"Query vector dimension {len(query_vector)} doesn't match store dimension {self.dimension}")
        
        # Calculate cosine similarity
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for vector_id, entry in self.vectors.items():
            # Apply filter if provided
            if filter_dict:
                metadata_dict = entry.metadata.to_dict()
                if not all(metadata_dict.get(k) == v for k, v in filter_dict.items()):
                    continue
            
            # Calculate cosine similarity
            entry_norm = np.linalg.norm(entry.vector)
            if entry_norm == 0 or query_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, entry.vector) / (entry_norm * query_norm)
            
            similarities.append({
                "id": vector_id,
                "score": float(similarity),
                "vector": entry.vector if include_metadata else [],
                "metadata": entry.metadata.to_dict() if include_metadata else {}
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
    
    async def get_vector_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID."""
        if vector_id in self.vectors:
            entry = self.vectors[vector_id]
            return {
                "id": entry.metadata.id,
                "vector": entry.vector,
                "metadata": entry.metadata.to_dict()
            }
        return None
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        for vector_id in vector_ids:
            self.vectors.pop(vector_id, None)
        return True
    
    async def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Delete vectors by filter."""
        to_delete = []
        for vector_id, entry in self.vectors.items():
            metadata_dict = entry.metadata.to_dict()
            if all(metadata_dict.get(k) == v for k, v in filter_dict.items()):
                to_delete.append(vector_id)
        
        for vector_id in to_delete:
            self.vectors.pop(vector_id, None)
        
        return True
    
    async def update_vector_metadata(self, vector_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update vector metadata."""
        if vector_id in self.vectors:
            entry = self.vectors[vector_id]
            # Update metadata
            for key, value in metadata_updates.items():
                if hasattr(entry.metadata, key):
                    setattr(entry.metadata, key, value)
            return True
        return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "dimension": self.dimension,
            "metric": "cosine",
            "vector_count": len(self.vectors),
            "index_fullness": 0.0,
            "namespaces": {}
        }
    
    def get_store_status(self) -> Dict[str, Any]:
        """Get store status."""
        return {
            "initialized": self.is_initialized,
            "index_name": "in_memory",
            "dimension": self.dimension,
            "metric": "cosine",
            "vector_count": len(self.vectors),
            "last_updated": datetime.now().isoformat()
        }