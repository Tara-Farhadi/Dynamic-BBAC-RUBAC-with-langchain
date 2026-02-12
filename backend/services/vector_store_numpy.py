"""
Simple NumPy-based vector store as fallback when ChromaDB has compatibility issues.
This is a lightweight alternative that stores vectors in memory with JSON persistence.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import threading

from config.settings import settings
from services.embedding import embedding_service


class NumpyVectorStore:
    """Simple NumPy-based vector store for behavioral and policy RAG."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.persist_dir = Path(settings.chroma_persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-user behavioral collections
        self._user_data: Dict[str, Dict] = {}
        
        # Policy collection (shared)
        self._policy_data: Dict = {
            'ids': [],
            'embeddings': [],
            'documents': [],
            'metadatas': []
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing data
        self._load_from_disk()
        
        print(f"[OK] NumPy vector store initialized at {self.persist_dir}")
    
    def _get_user_data(self, user_id: str) -> Dict:
        """Get or create user data structure."""
        if user_id not in self._user_data:
            self._user_data[user_id] = {
                'ids': [],
                'embeddings': [],
                'documents': [],
                'metadatas': []
            }
        return self._user_data[user_id]
    
    def _save_to_disk(self):
        """Save all data to disk."""
        try:
            # Save user data
            user_file = self.persist_dir / "user_vectors.json"
            user_data_serializable = {}
            for user_id, data in self._user_data.items():
                user_data_serializable[user_id] = {
                    'ids': data['ids'],
                    'embeddings': [e.tolist() if isinstance(e, np.ndarray) else e for e in data['embeddings']],
                    'documents': data['documents'],
                    'metadatas': data['metadatas']
                }
            with open(user_file, 'w') as f:
                json.dump(user_data_serializable, f)
            
            # Save policy data
            policy_file = self.persist_dir / "policy_vectors.json"
            policy_serializable = {
                'ids': self._policy_data['ids'],
                'embeddings': [e.tolist() if isinstance(e, np.ndarray) else e for e in self._policy_data['embeddings']],
                'documents': self._policy_data['documents'],
                'metadatas': self._policy_data['metadatas']
            }
            with open(policy_file, 'w') as f:
                json.dump(policy_serializable, f)
        except Exception as e:
            print(f"Warning: Failed to save vector store: {e}")
    
    def _load_from_disk(self):
        """Load data from disk if exists."""
        try:
            # Load user data
            user_file = self.persist_dir / "user_vectors.json"
            if user_file.exists():
                with open(user_file, 'r') as f:
                    loaded = json.load(f)
                for user_id, data in loaded.items():
                    self._user_data[user_id] = {
                        'ids': data['ids'],
                        'embeddings': [np.array(e) for e in data['embeddings']],
                        'documents': data['documents'],
                        'metadatas': data['metadatas']
                    }
            
            # Load policy data
            policy_file = self.persist_dir / "policy_vectors.json"
            if policy_file.exists():
                with open(policy_file, 'r') as f:
                    loaded = json.load(f)
                self._policy_data = {
                    'ids': loaded['ids'],
                    'embeddings': [np.array(e) for e in loaded['embeddings']],
                    'documents': loaded['documents'],
                    'metadatas': loaded['metadatas']
                }
        except Exception as e:
            print(f"Warning: Failed to load vector store: {e}")
    
    def _cosine_similarity(self, query: np.ndarray, vectors: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarity between query and vectors."""
        if not vectors:
            return np.array([])
        
        vectors_matrix = np.array(vectors)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        vectors_norm = vectors_matrix / (np.linalg.norm(vectors_matrix, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(vectors_norm, query_norm)
        return similarities
    
    # ========================================
    # Behavioral Collections (Per-User)
    # ========================================
    
    def add_transaction(
        self,
        user_id: str,
        transaction_id: str,
        transaction: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add a transaction to user's behavioral collection."""
        with self._lock:
            data = self._get_user_data(user_id)
            
            # Generate embedding if not provided
            if embedding is None:
                embedding = embedding_service.embed_transaction(transaction)
            
            # Create document text
            doc_text = embedding_service.transaction_to_text(transaction)
            
            # Prepare metadata
            metadata = {
                "user_id": user_id,
                "amt": float(transaction.get('amt', 0)),
                "merchant": str(transaction.get('merchant', '')),
                "category": str(transaction.get('category', '')),
                "city": str(transaction.get('city', '')),
                "state": str(transaction.get('state', '')),
                "country": str(transaction.get('country', 'US')),
            }
            
            trans_time = transaction.get('trans_date_trans_time')
            if trans_time:
                if hasattr(trans_time, 'isoformat'):
                    metadata['timestamp'] = trans_time.isoformat()
                else:
                    metadata['timestamp'] = str(trans_time)
            
            # Check for duplicate
            if transaction_id in data['ids']:
                idx = data['ids'].index(transaction_id)
                data['embeddings'][idx] = np.array(embedding)
                data['documents'][idx] = doc_text
                data['metadatas'][idx] = metadata
            else:
                data['ids'].append(transaction_id)
                data['embeddings'].append(np.array(embedding))
                data['documents'].append(doc_text)
                data['metadatas'].append(metadata)
            
            self._save_to_disk()
    
    def add_transactions_batch(
        self,
        user_id: str,
        transactions: List[Dict[str, Any]],
        transaction_ids: List[str]
    ) -> int:
        """Add multiple transactions in batch."""
        with self._lock:
            data = self._get_user_data(user_id)
            
            # Generate embeddings in batch
            texts = [embedding_service.transaction_to_text(t) for t in transactions]
            embeddings = embedding_service.embed_texts(texts)
            
            for i, (txn, txn_id, emb, text) in enumerate(zip(transactions, transaction_ids, embeddings, texts)):
                metadata = {
                    "user_id": user_id,
                    "amt": float(txn.get('amt', 0)),
                    "merchant": str(txn.get('merchant', '')),
                    "category": str(txn.get('category', '')),
                    "city": str(txn.get('city', '')),
                    "state": str(txn.get('state', '')),
                    "country": str(txn.get('country', 'US')),
                }
                
                trans_time = txn.get('trans_date_trans_time')
                if trans_time:
                    if hasattr(trans_time, 'isoformat'):
                        metadata['timestamp'] = trans_time.isoformat()
                    else:
                        metadata['timestamp'] = str(trans_time)
                
                # Upsert
                if txn_id in data['ids']:
                    idx = data['ids'].index(txn_id)
                    data['embeddings'][idx] = np.array(emb)
                    data['documents'][idx] = text
                    data['metadatas'][idx] = metadata
                else:
                    data['ids'].append(txn_id)
                    data['embeddings'].append(np.array(emb))
                    data['documents'].append(text)
                    data['metadatas'].append(metadata)
            
            self._save_to_disk()
            return len(transactions)
    
    def search_similar_transactions(
        self,
        user_id: str,
        query_embedding: List[float],
        n_results: int = None
    ) -> Dict[str, Any]:
        """Search for similar transactions in user's history."""
        if n_results is None:
            n_results = settings.behavioral_k_results
        
        data = self._get_user_data(user_id)
        
        if not data['embeddings']:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        query = np.array(query_embedding)
        similarities = self._cosine_similarity(query, data['embeddings'])
        
        # Get top-k indices
        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Convert similarity to distance (1 - similarity)
        distances = [1 - similarities[i] for i in top_indices]
        documents = [data['documents'][i] for i in top_indices]
        metadatas = [data['metadatas'][i] for i in top_indices]
        
        return {
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances]
        }
    
    def get_user_transaction_count(self, user_id: str) -> int:
        """Get the number of transactions for a user."""
        data = self._get_user_data(user_id)
        return len(data['ids'])
    
    def delete_user_collection(self, user_id: str) -> bool:
        """Delete a user's behavioral collection."""
        with self._lock:
            if user_id in self._user_data:
                del self._user_data[user_id]
                self._save_to_disk()
                return True
            return False
    
    # ========================================
    # Policy Collection (Shared)
    # ========================================
    
    def add_policy_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add a policy document to the collection."""
        with self._lock:
            if embedding is None:
                embedding = embedding_service.embed_text(text)
            
            # Upsert
            if doc_id in self._policy_data['ids']:
                idx = self._policy_data['ids'].index(doc_id)
                self._policy_data['embeddings'][idx] = np.array(embedding)
                self._policy_data['documents'][idx] = text
                self._policy_data['metadatas'][idx] = metadata
            else:
                self._policy_data['ids'].append(doc_id)
                self._policy_data['embeddings'].append(np.array(embedding))
                self._policy_data['documents'].append(text)
                self._policy_data['metadatas'].append(metadata)
            
            self._save_to_disk()
    
    def search_policies(
        self,
        query_text: str,
        policy_type: Optional[str] = None,
        n_results: int = None
    ) -> Dict[str, Any]:
        """Search for relevant policy documents."""
        if n_results is None:
            n_results = settings.policy_k_results
        
        if not self._policy_data['embeddings']:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Filter by policy_type if specified
        if policy_type:
            filtered_indices = [
                i for i, meta in enumerate(self._policy_data['metadatas'])
                if meta.get('policy_type') == policy_type
            ]
            
            if not filtered_indices:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            filtered_embeddings = [self._policy_data['embeddings'][i] for i in filtered_indices]
            filtered_documents = [self._policy_data['documents'][i] for i in filtered_indices]
            filtered_metadatas = [self._policy_data['metadatas'][i] for i in filtered_indices]
        else:
            filtered_indices = list(range(len(self._policy_data['embeddings'])))
            filtered_embeddings = self._policy_data['embeddings']
            filtered_documents = self._policy_data['documents']
            filtered_metadatas = self._policy_data['metadatas']
        
        query_embedding = embedding_service.embed_text(query_text)
        query = np.array(query_embedding)
        similarities = self._cosine_similarity(query, filtered_embeddings)
        
        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        distances = [1 - similarities[i] for i in top_indices]
        documents = [filtered_documents[i] for i in top_indices]
        metadatas = [filtered_metadatas[i] for i in top_indices]
        
        return {
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances]
        }
    
    def load_policy_pdf(
        self,
        pdf_path: str,
        policy_type: str = "organizational"
    ) -> int:
        """
        Load a PDF policy document, extract text, chunk it, and index.
        
        Args:
            pdf_path: Path to the PDF file
            policy_type: Type of policy (organizational or regulatory)
            
        Returns:
            Number of chunks created
        """
        try:
            import PyPDF2
        except ImportError:
            print("Warning: PyPDF2 not installed. Cannot extract PDF text.")
            return 0
        
        # Extract text from PDF
        text_content = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return 0
        
        if not text_content:
            return 0
        
        full_text = "\n".join(text_content)
        
        # Chunk the text (~400 tokens per chunk)
        chunks = self._chunk_text(full_text, chunk_size=400)
        
        # Index each chunk
        chunks_created = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"{Path(pdf_path).stem}_{policy_type}_{i}"
            metadata = {
                "source": Path(pdf_path).name,
                "policy_type": policy_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            self.add_policy_document(
                doc_id=doc_id,
                text=chunk,
                metadata=metadata
            )
            chunks_created += 1
        
        return chunks_created
    
    def _chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        Uses sentence boundaries where possible.
        """
        # Simple word-based chunking (approximates tokens)
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += 1
            
            # Check if we've reached chunk size and hit a sentence boundary
            if current_size >= chunk_size:
                # Look for sentence endings
                if word.endswith(('.', '!', '?', ';')):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                elif current_size >= chunk_size + 100:  # Force split if too long
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
        
        # Add remaining text
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_policy_count(self) -> int:
        """Get the number of policy documents."""
        return len(self._policy_data['ids'])
    
    def list_users(self) -> List[str]:
        """List all users with behavioral data."""
        return list(self._user_data.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        total_transactions = sum(
            len(data['ids']) for data in self._user_data.values()
        )
        return {
            'policy_documents': len(self._policy_data['ids']),
            'user_collections': len(self._user_data),
            'total_transactions': total_transactions,
            'persist_directory': str(self.persist_dir)
        }


# Singleton instance
vector_store = NumpyVectorStore()
