"""
Embedding service using sentence-transformers for RAG.
Handles generation of embeddings for transactions and policy documents.
"""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from config.settings import settings


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    _instance: Optional['EmbeddingService'] = None
    
    def __new__(cls) -> 'EmbeddingService':
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
            
        print(f"Loading embedding model: {settings.embedding_model}...")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = settings.embedding_dimension
        self._initialized = True
        print(f"[OK] Embedding model loaded (dimension: {self.dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def embed_transaction(self, transaction: dict) -> List[float]:
        """
        Generate embedding for a transaction.
        Converts transaction to natural language description first.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Embedding vector
        """
        text = self.transaction_to_text(transaction)
        return self.embed_text(text)
    
    def transaction_to_text(self, transaction: dict) -> str:
        """
        Convert transaction to natural language for embedding.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Natural language description of transaction
        """
        user_id = transaction.get('user_id', 'unknown')
        amt = transaction.get('amt', 0)
        merchant = transaction.get('merchant', 'unknown merchant')
        category = transaction.get('category', 'unknown category')
        trans_time = transaction.get('trans_date_trans_time', 'unknown time')
        city = transaction.get('city', 'unknown city')
        state = transaction.get('state', '')
        country = transaction.get('country', 'US')
        
        # Extract hour if datetime available
        hour_str = ""
        if hasattr(trans_time, 'hour'):
            hour = trans_time.hour
            if hour < 6:
                hour_str = "early morning"
            elif hour < 12:
                hour_str = "morning"
            elif hour < 18:
                hour_str = "afternoon"
            elif hour < 22:
                hour_str = "evening"
            else:
                hour_str = "late night"
        
        location = f"{city}"
        if state:
            location += f", {state}"
        if country != "US":
            location += f", {country}"
        
        text = (
            f"User {user_id} made a ${amt:.2f} purchase at {merchant} "
            f"in {category} category on {trans_time} "
            f"in {location}"
        )
        
        if hour_str:
            text += f" during {hour_str}"
        
        return text
    
    def embed_policy_chunk(self, chunk: str, metadata: dict = None) -> List[float]:
        """
        Generate embedding for a policy document chunk.
        
        Args:
            chunk: Text chunk from policy document
            metadata: Optional metadata to include in context
            
        Returns:
            Embedding vector
        """
        # Optionally prepend metadata context
        if metadata:
            policy_type = metadata.get('policy_type', '')
            source = metadata.get('source', '')
            if policy_type or source:
                chunk = f"[{policy_type} policy from {source}] {chunk}"
        
        return self.embed_text(chunk)
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()


# Convenience singleton
embedding_service = EmbeddingService()
