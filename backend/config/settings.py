"""
GUARDIAN System Configuration Settings
Centralized configuration management using Pydantic Settings
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Settings
    app_name: str = "GUARDIAN Transaction Monitor"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = "/api/v1"
    
    # Database Settings
    database_url: str = Field(
        default="sqlite:///./data/transactions.db",
        description="SQLAlchemy database URL"
    )
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # Vector Database Settings (ChromaDB)
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    behavioral_collection_prefix: str = "user_behavioral"
    policy_collection_name: str = "policy_documents"
    
    # LLM Settings (OpenAI)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="LLM model to use"
    )
    llm_max_tokens: int = Field(default=2048, description="Max tokens for LLM response")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_timeout: int = Field(default=30, description="LLM API timeout in seconds")
    
    # Embedding Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    
    # RAG Settings
    behavioral_k_results: int = Field(
        default=5,
        description="Number of similar transactions to retrieve for behavioral RAG"
    )
    policy_k_results: int = Field(
        default=5,
        description="Number of policy chunks to retrieve for policy RAG"
    )
    policy_chunk_size: int = Field(
        default=400,
        description="Token size for policy document chunks"
    )
    policy_chunk_overlap: int = Field(
        default=50,
        description="Overlap between policy chunks"
    )
    
    # Decision Thresholds
    threshold_low: float = Field(
        default=0.3,
        description="Below this threshold: ALLOW"
    )
    threshold_high: float = Field(
        default=0.7,
        description="Above this threshold: DENY"
    )
    
    # Fusion Weights (initial values, updated by learning)
    behavioral_weight: float = Field(
        default=0.5,
        description="Weight for behavioral score in fusion"
    )
    policy_weight: float = Field(
        default=0.5,
        description="Weight for policy score in fusion"
    )
    
    # Learning Settings
    learning_rate: float = Field(default=0.01, description="Learning rate for weight updates")
    reward_correct: int = Field(default=10, description="Reward for correct decisions")
    penalty_false_positive: int = Field(default=-2, description="Penalty for false positives")
    penalty_false_negative: int = Field(default=-50, description="Penalty for false negatives")
    
    # Data Paths
    uploads_directory: str = Field(
        default="./data/uploads",
        description="Directory for uploaded files (CSV, PDF)"
    )
    users_data_path: str = Field(
        default="./data/users",
        description="Directory for user CSV files"
    )
    policies_data_path: str = Field(
        default="./data/policies",
        description="Directory for policy PDF files"
    )
    logs_path: str = Field(
        default="./logs",
        description="Directory for application logs"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=10, description="Max requests per second")
    
    # Context Window Settings
    transaction_history_days: int = Field(
        default=30,
        description="Days of transaction history to consider"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields in .env file


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
