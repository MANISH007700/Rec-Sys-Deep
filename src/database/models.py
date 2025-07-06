"""
Database models for the recommendation system.
Defines SQLAlchemy models for users, movies, ratings, and recommendations.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User model for storing user information."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    occupation = Column(String(50))
    zipcode = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = relationship("Rating", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")


class Movie(Base):
    """Movie model for storing movie information."""
    
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    genres = Column(Text)
    year = Column(Integer)
    imdb_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = relationship("Rating", back_populates="movie")
    recommendations = relationship("Recommendation", back_populates="movie")


class Rating(Base):
    """Rating model for storing user-movie ratings."""
    
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")


class Recommendation(Base):
    """Recommendation model for storing generated recommendations."""
    
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    score = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    rank = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    movie = relationship("Movie", back_populates="recommendations")


class ModelPerformance(Base):
    """Model performance tracking."""
    
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_split = Column(String(20))  # train, test, validation
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    class Meta:
        unique_together = ('model_name', 'metric_name', 'dataset_split', 'timestamp')


class UserFeedback(Base):
    """User feedback on recommendations."""
    
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    feedback_type = Column(String(20), nullable=False)  # like, dislike, click, view
    feedback_value = Column(Float)  # rating if applicable
    session_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    movie = relationship("Movie") 