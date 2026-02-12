"""
Vector store service using NumPy-based RAG (fallback for Windows compatibility).
Manages separate collections for behavioral data and policy documents.

Note: ChromaDB with chroma-hnswlib has access violation issues on Windows/Python 3.12.
This module re-exports the numpy-based implementation as a drop-in replacement.
"""

# Import from numpy-based implementation
from services.vector_store_numpy import vector_store, NumpyVectorStore

# Re-export for compatibility
VectorStoreManager = NumpyVectorStore

__all__ = ['vector_store', 'VectorStoreManager']
