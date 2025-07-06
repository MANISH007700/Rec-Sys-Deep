"""
Database connection management for the recommendation system.
Supports both PostgreSQL and MongoDB connections.
"""

import logging
import os
from typing import Optional

import pymongo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager for PostgreSQL."""
    
    def __init__(self, database_type: str = "postgresql"):
        """
        Initialize database manager.
        
        Args:
            database_type: Type of database ('postgresql' or 'mongodb')
        """
        self.database_type = database_type
        self.engine = None
        self.SessionLocal = None
        self.client = None
        self.db = None
        
        if database_type == "postgresql":
            self._setup_postgresql()
        elif database_type == "mongodb":
            self._setup_mongodb()
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    def _setup_postgresql(self):
        """Setup PostgreSQL connection."""
        # Get database configuration from environment variables
        host = os.getenv("DATABASE_HOST", "localhost")
        port = os.getenv("DATABASE_PORT", "5432")
        name = os.getenv("DATABASE_NAME", "recsys")
        user = os.getenv("DATABASE_USER", "postgres")
        password = os.getenv("DATABASE_PASSWORD", "password")
        
        # Create database URL
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"PostgreSQL connection established to {host}:{port}/{name}")
    
    def _setup_mongodb(self):
        """Setup MongoDB connection."""
        # Get database configuration from environment variables
        host = os.getenv("DATABASE_HOST", "localhost")
        port = int(os.getenv("DATABASE_PORT", "27017"))
        name = os.getenv("DATABASE_NAME", "recsys")
        user = os.getenv("DATABASE_USER", "")
        password = os.getenv("DATABASE_PASSWORD", "")
        
        # Create connection string
        if user and password:
            connection_string = f"mongodb://{user}:{password}@{host}:{port}/{name}"
        else:
            connection_string = f"mongodb://{host}:{port}/{name}"
        
        # Create client
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[name]
        
        logger.info(f"MongoDB connection established to {host}:{port}/{name}")
    
    def create_tables(self):
        """Create all database tables (PostgreSQL only)."""
        if self.database_type == "postgresql":
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        else:
            logger.warning("create_tables() is only supported for PostgreSQL")
    
    def drop_tables(self):
        """Drop all database tables (PostgreSQL only)."""
        if self.database_type == "postgresql":
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        else:
            logger.warning("drop_tables() is only supported for PostgreSQL")
    
    def get_session(self) -> Session:
        """Get database session (PostgreSQL only)."""
        if self.database_type == "postgresql":
            return self.SessionLocal()
        else:
            raise ValueError("get_session() is only supported for PostgreSQL")
    
    def get_collection(self, collection_name: str):
        """Get MongoDB collection."""
        if self.database_type == "mongodb":
            return self.db[collection_name]
        else:
            raise ValueError("get_collection() is only supported for MongoDB")
    
    def close(self):
        """Close database connections."""
        if self.database_type == "postgresql" and self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connection closed")
        elif self.database_type == "mongodb" and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        database_type = os.getenv("DATABASE_TYPE", "postgresql")
        db_manager = DatabaseManager(database_type)
    return db_manager


def get_db_session() -> Session:
    """Get database session for dependency injection."""
    db_manager = get_database_manager()
    if db_manager.database_type == "postgresql":
        db = db_manager.get_session()
        try:
            yield db
        finally:
            db.close()
    else:
        raise ValueError("Session-based operations are only supported for PostgreSQL")


def init_database():
    """Initialize database tables."""
    db_manager = get_database_manager()
    db_manager.create_tables()


def close_database():
    """Close database connections."""
    global db_manager
    if db_manager:
        db_manager.close()
        db_manager = None 