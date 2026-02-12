"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os

from config.settings import settings
from models.database import Base


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        """Initialize database connection."""
        # Ensure data directory exists
        db_path = settings.database_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def init_db(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        print("[OK] Database schema initialized")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close the database engine."""
        self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
