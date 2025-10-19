"""
Embeddings Generation Module

This module provides functionality to generate embeddings for different types of
market data including price patterns, news sentiment, and portfolio states.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import re

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from backend.data_collectors.base_collector import OHLCVData, NewsData
from backend.core.config import settings
from backend.core.exceptions import ValidationError, DataCollectionError
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PricePattern:
    """Price pattern data for embedding generation."""
    symbol: str
    timestamps: List[datetime]
    prices: List[float]
    volumes: List[int]
    technical_indicators: Dict[str, List[float]]
    
    def __post_init__(self):
        """Validate price pattern data."""
        if not self.prices or len(self.prices) == 0:
            raise ValidationError("Price pattern cannot be empty", "embeddings")
        if len(self.prices) != len(self.volumes):
            raise ValidationError("Prices and volumes must have same length", "embeddings")


@dataclass
class NewsPattern:
    """News pattern data for embedding generation."""
    symbol: Optional[str]
    headlines: List[str]
    contents: List[str]
    timestamps: List[datetime]
    sentiment_scores: List[float]
    
    def __post_init__(self):
        """Validate news pattern data."""
        if not self.headlines or len(self.headlines) == 0:
            raise ValidationError("News pattern cannot be empty", "embeddings")


@dataclass
class PortfolioState:
    """Portfolio state data for embedding generation."""
    positions: Dict[str, Dict[str, Any]]  # symbol -> {quantity, avg_price, current_price}
    cash: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    risk_metrics: Dict[str, float]
    timestamp: datetime
    
    def __post_init__(self):
        """Validate portfolio state data."""
        if self.total_value <= 0:
            raise ValidationError("Total portfolio value must be positive", "embeddings")


class TextEmbeddingGenerator:
    """
    Generates embeddings for text data using sentence transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.logger = get_logger("text_embedding_generator")
        self.model = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the text embedding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.is_initialized = True
            self.logger.info(f"Text embedding model initialized: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text embedding model: {e}")
            raise DataCollectionError(f"Text embedding initialization failed: {e}", "text_embedding_generator")
    
    async def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        if not self.is_initialized:
            raise ValidationError("Text embedding generator not initialized", "text_embedding_generator")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {e}")
            raise DataCollectionError(f"Failed to generate text embedding: {e}", "text_embedding_generator")
    
    async def generate_batch_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.is_initialized:
            raise ValidationError("Text embedding generator not initialized", "text_embedding_generator")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"Error generating batch text embeddings: {e}")
            raise DataCollectionError(f"Failed to generate batch text embeddings: {e}", "text_embedding_generator")
    
    async def generate_news_embedding(self, news_pattern: NewsPattern) -> List[float]:
        """
        Generate embedding for news pattern.
        
        Args:
            news_pattern: News pattern data
            
        Returns:
            News embedding vector
        """
        try:
            # Combine headlines and contents
            combined_texts = []
            
            for i, (headline, content) in enumerate(zip(news_pattern.headlines, news_pattern.contents)):
                # Create weighted text (headline is more important)
                weighted_text = f"HEADLINE: {headline} CONTENT: {content}"
                combined_texts.append(weighted_text)
            
            # Generate embeddings for all texts
            embeddings = await self.generate_batch_text_embeddings(combined_texts)
            
            # Aggregate embeddings (weighted average)
            if embeddings:
                # Weight by recency and sentiment
                weights = []
                now = datetime.now()
                
                for i, (timestamp, sentiment) in enumerate(zip(news_pattern.timestamps, news_pattern.sentiment_scores)):
                    # Recency weight (more recent = higher weight)
                    hours_ago = (now - timestamp).total_seconds() / 3600
                    recency_weight = np.exp(-hours_ago / 24)  # Decay over 24 hours
                    
                    # Sentiment weight (stronger sentiment = higher weight)
                    sentiment_weight = abs(sentiment)
                    
                    # Combined weight
                    combined_weight = recency_weight * (1 + sentiment_weight)
                    weights.append(combined_weight)
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Weighted average of embeddings
                weighted_embedding = np.average(embeddings, axis=0, weights=weights)
                return weighted_embedding.tolist()
            
            # Fallback to simple average
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                return avg_embedding.tolist()
            
            # Empty fallback
            return [0.0] * self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            self.logger.error(f"Error generating news embedding: {e}")
            raise DataCollectionError(f"Failed to generate news embedding: {e}", "text_embedding_generator")


class PriceEmbeddingGenerator:
    """
    Generates embeddings for price patterns using neural networks.
    """
    
    def __init__(self, sequence_length: int = 50, embedding_dim: int = 128):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.logger = get_logger("price_embedding_generator")
        self.model = None
        self.scaler = StandardScaler()
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the price embedding model."""
        try:
            # Create a simple LSTM-based embedding model
            self.model = PriceEmbeddingNet(
                input_size=6,  # OHLCV + 1 technical indicator
                hidden_size=64,
                embedding_dim=self.embedding_dim,
                sequence_length=self.sequence_length
            )
            
            self.is_initialized = True
            self.logger.info("Price embedding model initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize price embedding model: {e}")
            raise DataCollectionError(f"Price embedding initialization failed: {e}", "price_embedding_generator")
    
    async def generate_price_embedding(self, price_pattern: PricePattern) -> List[float]:
        """
        Generate embedding for price pattern.
        
        Args:
            price_pattern: Price pattern data
            
        Returns:
            Price embedding vector
        """
        if not self.is_initialized:
            raise ValidationError("Price embedding generator not initialized", "price_embedding_generator")
        
        try:
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data(price_pattern)
            
            if len(sequence_data) == 0:
                return [0.0] * self.embedding_dim
            
            # Pad or truncate to sequence length
            if len(sequence_data) < self.sequence_length:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - len(sequence_data), sequence_data.shape[1]))
                sequence_data = np.vstack([padding, sequence_data])
            else:
                # Take the most recent sequence
                sequence_data = sequence_data[-self.sequence_length:]
            
            # Generate embedding
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)  # Add batch dimension
                embedding = self.model(sequence_tensor)
                embedding = embedding.squeeze().numpy()
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating price embedding: {e}")
            raise DataCollectionError(f"Failed to generate price embedding: {e}", "price_embedding_generator")
    
    def _prepare_sequence_data(self, price_pattern: PricePattern) -> np.ndarray:
        """
        Prepare sequence data from price pattern.
        
        Args:
            price_pattern: Price pattern data
            
        Returns:
            Normalized sequence data
        """
        try:
            # Calculate returns
            prices = np.array(price_pattern.prices)
            volumes = np.array(price_pattern.volumes, dtype=float)
            
            # Calculate basic features
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Add 0 for first element
            
            # Normalize volumes
            volumes_norm = volumes / np.max(volumes) if np.max(volumes) > 0 else volumes
            
            # Get technical indicators
            rsi_values = price_pattern.technical_indicators.get('rsi', [50] * len(prices))
            sma_values = price_pattern.technical_indicators.get('sma', list(prices))
            
            # Combine features
            features = np.column_stack([
                prices,
                returns,
                volumes_norm,
                rsi_values[:len(prices)],
                sma_values[:len(prices)],
                np.ones(len(prices))  # Bias term
            ])
            
            # Normalize features
            if len(features) > 1:
                features = self.scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {e}")
            return np.array([])
    
    async def generate_batch_price_embeddings(self, price_patterns: List[PricePattern]) -> List[List[float]]:
        """
        Generate embeddings for multiple price patterns.
        
        Args:
            price_patterns: List of price patterns
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for pattern in price_patterns:
            try:
                embedding = await self.generate_price_embedding(pattern)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error generating embedding for pattern: {e}")
                embeddings.append([0.0] * self.embedding_dim)
        
        return embeddings


class PortfolioEmbeddingGenerator:
    """
    Generates embeddings for portfolio states.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.logger = get_logger("portfolio_embedding_generator")
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the portfolio embedding generator."""
        self.is_initialized = True
        self.logger.info("Portfolio embedding generator initialized")
    
    async def generate_portfolio_embedding(self, portfolio_state: PortfolioState) -> List[float]:
        """
        Generate embedding for portfolio state.
        
        Args:
            portfolio_state: Portfolio state data
            
        Returns:
            Portfolio embedding vector
        """
        if not self.is_initialized:
            raise ValidationError("Portfolio embedding generator not initialized", "portfolio_embedding_generator")
        
        try:
            # Extract portfolio features
            features = []
            
            # Basic portfolio metrics
            features.append(portfolio_state.cash / portfolio_state.total_value)  # Cash ratio
            features.append(portfolio_state.unrealized_pnl / portfolio_state.total_value)  # Unrealized PnL ratio
            features.append(portfolio_state.realized_pnl / portfolio_state.total_value)  # Realized PnL ratio
            
            # Position concentration
            position_values = []
            for symbol, position in portfolio_state.positions.items():
                position_value = position['quantity'] * position['current_price']
                position_values.append(position_value)
            
            if position_values:
                # Concentration metrics
                max_position_ratio = max(position_values) / portfolio_state.total_value
                features.append(max_position_ratio)
                
                # Number of positions
                features.append(len(position_values) / 20)  # Normalized by max expected positions
                
                # Position variance (diversification)
                position_ratios = np.array(position_values) / portfolio_state.total_value
                position_variance = np.var(position_ratios)
                features.append(position_variance)
            else:
                features.extend([0, 0, 0])  # No positions
            
            # Risk metrics
            for metric_name, value in portfolio_state.risk_metrics.items():
                if metric_name in ['var', 'beta', 'sharpe_ratio', 'max_drawdown']:
                    features.append(value)
            
            # Pad or truncate to embedding dimension
            while len(features) < self.embedding_dim:
                features.append(0.0)
            
            features = features[:self.embedding_dim]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio embedding: {e}")
            raise DataCollectionError(f"Failed to generate portfolio embedding: {e}", "portfolio_embedding_generator")


class PriceEmbeddingNet(nn.Module):
    """
    Neural network for generating price pattern embeddings.
    """
    
    def __init__(self, input_size: int, hidden_size: int, embedding_dim: int, sequence_length: int):
        super(PriceEmbeddingNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, embedding_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Final embedding
        embedding = self.fc(pooled)
        
        return embedding


class EmbeddingGenerator:
    """
    Main embedding generator that orchestrates all embedding types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("embedding_generator")
        self.is_initialized = False
        
        # Initialize generators
        self.text_generator = TextEmbeddingGenerator(
            model_name=self.config.get("text_model", "all-MiniLM-L6-v2")
        )
        self.price_generator = PriceEmbeddingGenerator(
            sequence_length=self.config.get("price_sequence_length", 50),
            embedding_dim=self.config.get("price_embedding_dim", 128)
        )
        self.portfolio_generator = PortfolioEmbeddingGenerator(
            embedding_dim=self.config.get("portfolio_embedding_dim", 64)
        )
        
    async def initialize(self) -> None:
        """Initialize all embedding generators."""
        try:
            await self.text_generator.initialize()
            await self.price_generator.initialize()
            await self.portfolio_generator.initialize()
            
            self.is_initialized = True
            self.logger.info("All embedding generators initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding generators: {e}")
            raise DataCollectionError(f"Embedding generators initialization failed: {e}", "embedding_generator")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.is_initialized = False
        self.logger.info("Embedding generators cleaned up")
    
    async def generate_price_pattern_embedding(self, price_pattern: PricePattern) -> List[float]:
        """Generate embedding for price pattern."""
        return await self.price_generator.generate_price_embedding(price_pattern)
    
    async def generate_news_pattern_embedding(self, news_pattern: NewsPattern) -> List[float]:
        """Generate embedding for news pattern."""
        return await self.text_generator.generate_news_embedding(news_pattern)
    
    async def generate_portfolio_embedding(self, portfolio_state: PortfolioState) -> List[float]:
        """Generate embedding for portfolio state."""
        return await self.portfolio_generator.generate_portfolio_embedding(portfolio_state)
    
    async def generate_multimodal_embedding(
        self,
        price_pattern: Optional[PricePattern] = None,
        news_pattern: Optional[NewsPattern] = None,
        portfolio_state: Optional[PortfolioState] = None
    ) -> List[float]:
        """
        Generate a combined embedding from multiple data types.
        
        Args:
            price_pattern: Price pattern data
            news_pattern: News pattern data
            portfolio_state: Portfolio state data
            
        Returns:
            Combined embedding vector
        """
        embeddings = []
        
        try:
            # Generate individual embeddings
            if price_pattern:
                price_emb = await self.generate_price_pattern_embedding(price_pattern)
                embeddings.append(price_emb)
            
            if news_pattern:
                news_emb = await self.generate_news_pattern_embedding(news_pattern)
                embeddings.append(news_emb)
            
            if portfolio_state:
                portfolio_emb = await self.generate_portfolio_embedding(portfolio_state)
                embeddings.append(portfolio_emb)
            
            if not embeddings:
                raise ValidationError("At least one data type must be provided", "embedding_generator")
            
            # Combine embeddings (concatenate and reduce)
            if len(embeddings) == 1:
                return embeddings[0]
            
            # Concatenate embeddings
            combined = np.concatenate(embeddings)
            
            # Reduce dimensionality using PCA if needed
            if len(combined) > 384:  # Target dimension
                pca = PCA(n_components=384)
                combined = pca.fit_transform(combined.reshape(1, -1)).flatten()
            
            return combined.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal embedding: {e}")
            raise DataCollectionError(f"Failed to generate multimodal embedding: {e}", "embedding_generator")
    
    def get_generator_status(self) -> Dict[str, Any]:
        """Get status of all generators."""
        return {
            "initialized": self.is_initialized,
            "text_generator": {
                "model_name": self.text_generator.model_name,
                "initialized": self.text_generator.is_initialized
            },
            "price_generator": {
                "sequence_length": self.price_generator.sequence_length,
                "embedding_dim": self.price_generator.embedding_dim,
                "initialized": self.price_generator.is_initialized
            },
            "portfolio_generator": {
                "embedding_dim": self.portfolio_generator.embedding_dim,
                "initialized": self.portfolio_generator.is_initialized
            },
            "last_updated": datetime.now().isoformat()
        }