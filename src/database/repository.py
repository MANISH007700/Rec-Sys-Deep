"""
Database repository layer for data access operations.
Provides clean interface for database operations with caching and error handling.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from .models import ModelPerformance, Movie, Rating, Recommendation, User, UserFeedback

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_user(self, user_id: int, age: Optional[int] = None, 
                   gender: Optional[str] = None, occupation: Optional[str] = None,
                   zipcode: Optional[str] = None) -> User:
        """Create a new user."""
        try:
            user = User(
                user_id=user_id,
                age=age,
                gender=gender,
                occupation=occupation,
                zipcode=zipcode
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            logger.info(f"Created user with ID: {user_id}")
            return user
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user {user_id}: {e}")
            raise
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by user_id."""
        return self.db.query(User).filter(User.user_id == user_id).first()
    
    def get_user_by_db_id(self, db_id: int) -> Optional[User]:
        """Get user by database ID."""
        return self.db.query(User).filter(User.id == db_id).first()
    
    def get_all_users(self, limit: Optional[int] = None) -> List[User]:
        """Get all users with optional limit."""
        query = self.db.query(User)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user information."""
        try:
            user = self.get_user_by_id(user_id)
            if user:
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                user.updated_at = datetime.utcnow()
                self.db.commit()
                logger.info(f"Updated user {user_id}")
                return user
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user by user_id."""
        try:
            user = self.get_user_by_id(user_id)
            if user:
                self.db.delete(user)
                self.db.commit()
                logger.info(f"Deleted user {user_id}")
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            raise


class MovieRepository:
    """Repository for movie-related database operations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_movie(self, movie_id: int, title: str, genres: Optional[str] = None,
                    year: Optional[int] = None, imdb_url: Optional[str] = None) -> Movie:
        """Create a new movie."""
        try:
            movie = Movie(
                movie_id=movie_id,
                title=title,
                genres=genres,
                year=year,
                imdb_url=imdb_url
            )
            self.db.add(movie)
            self.db.commit()
            self.db.refresh(movie)
            logger.info(f"Created movie with ID: {movie_id}")
            return movie
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating movie {movie_id}: {e}")
            raise
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        """Get movie by movie_id."""
        return self.db.query(Movie).filter(Movie.movie_id == movie_id).first()
    
    def get_movie_by_db_id(self, db_id: int) -> Optional[Movie]:
        """Get movie by database ID."""
        return self.db.query(Movie).filter(Movie.id == db_id).first()
    
    def get_movies_by_genre(self, genre: str, limit: Optional[int] = None) -> List[Movie]:
        """Get movies by genre."""
        query = self.db.query(Movie).filter(Movie.genres.contains(genre))
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def search_movies(self, search_term: str, limit: Optional[int] = None) -> List[Movie]:
        """Search movies by title."""
        query = self.db.query(Movie).filter(Movie.title.ilike(f"%{search_term}%"))
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_popular_movies(self, limit: int = 10) -> List[Tuple[Movie, int]]:
        """Get most popular movies by rating count."""
        result = self.db.query(
            Movie, func.count(Rating.id).label('rating_count')
        ).join(Rating).group_by(Movie.id).order_by(
            desc('rating_count')
        ).limit(limit).all()
        
        return [(movie, count) for movie, count in result]
    
    def get_all_movies(self, limit: Optional[int] = None) -> List[Movie]:
        """Get all movies with optional limit."""
        query = self.db.query(Movie)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def update_movie(self, movie_id: int, **kwargs) -> Optional[Movie]:
        """Update movie information."""
        try:
            movie = self.get_movie_by_id(movie_id)
            if movie:
                for key, value in kwargs.items():
                    if hasattr(movie, key):
                        setattr(movie, key, value)
                movie.updated_at = datetime.utcnow()
                self.db.commit()
                logger.info(f"Updated movie {movie_id}")
                return movie
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating movie {movie_id}: {e}")
            raise


class RatingRepository:
    """Repository for rating-related database operations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_rating(self, user_id: int, movie_id: int, rating: float,
                     timestamp: Optional[datetime] = None) -> Rating:
        """Create a new rating."""
        try:
            # Get user and movie from database
            user = self.db.query(User).filter(User.user_id == user_id).first()
            movie = self.db.query(Movie).filter(Movie.movie_id == movie_id).first()
            
            if not user or not movie:
                raise ValueError(f"User {user_id} or Movie {movie_id} not found")
            
            rating_obj = Rating(
                user_id=user.id,
                movie_id=movie.id,
                rating=rating,
                timestamp=timestamp or datetime.utcnow()
            )
            self.db.add(rating_obj)
            self.db.commit()
            self.db.refresh(rating_obj)
            logger.info(f"Created rating: User {user_id} -> Movie {movie_id} = {rating}")
            return rating_obj
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating rating: {e}")
            raise
    
    def get_user_ratings(self, user_id: int) -> List[Rating]:
        """Get all ratings for a user."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user:
            return self.db.query(Rating).filter(Rating.user_id == user.id).all()
        return []
    
    def get_movie_ratings(self, movie_id: int) -> List[Rating]:
        """Get all ratings for a movie."""
        movie = self.db.query(Movie).filter(Movie.movie_id == movie_id).first()
        if movie:
            return self.db.query(Rating).filter(Rating.movie_id == movie.id).all()
        return []
    
    def get_rating(self, user_id: int, movie_id: int) -> Optional[Rating]:
        """Get specific rating by user and movie."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        movie = self.db.query(Movie).filter(Movie.movie_id == movie_id).first()
        
        if user and movie:
            return self.db.query(Rating).filter(
                and_(Rating.user_id == user.id, Rating.movie_id == movie.id)
            ).first()
        return None
    
    def update_rating(self, user_id: int, movie_id: int, new_rating: float) -> Optional[Rating]:
        """Update an existing rating."""
        try:
            rating = self.get_rating(user_id, movie_id)
            if rating:
                rating.rating = new_rating
                rating.timestamp = datetime.utcnow()
                self.db.commit()
                logger.info(f"Updated rating: User {user_id} -> Movie {movie_id} = {new_rating}")
                return rating
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating rating: {e}")
            raise
    
    def delete_rating(self, user_id: int, movie_id: int) -> bool:
        """Delete a rating."""
        try:
            rating = self.get_rating(user_id, movie_id)
            if rating:
                self.db.delete(rating)
                self.db.commit()
                logger.info(f"Deleted rating: User {user_id} -> Movie {movie_id}")
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting rating: {e}")
            raise
    
    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Get all ratings as a pandas DataFrame."""
        ratings = self.db.query(Rating).all()
        data = []
        for rating in ratings:
            data.append({
                'user_id': rating.user.user_id,
                'movie_id': rating.movie.movie_id,
                'rating': rating.rating,
                'timestamp': rating.timestamp
            })
        return pd.DataFrame(data)


class RecommendationRepository:
    """Repository for recommendation-related database operations."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_recommendation(self, user_id: int, movie_id: int, score: float,
                            model_name: str, rank: Optional[int] = None) -> Recommendation:
        """Create a new recommendation."""
        try:
            # Get user and movie from database
            user = self.db.query(User).filter(User.user_id == user_id).first()
            movie = self.db.query(Movie).filter(Movie.movie_id == movie_id).first()
            
            if not user or not movie:
                raise ValueError(f"User {user_id} or Movie {movie_id} not found")
            
            recommendation = Recommendation(
                user_id=user.id,
                movie_id=movie.id,
                score=score,
                model_name=model_name,
                rank=rank
            )
            self.db.add(recommendation)
            self.db.commit()
            self.db.refresh(recommendation)
            logger.info(f"Created recommendation: User {user_id} -> Movie {movie_id} (score: {score})")
            return recommendation
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating recommendation: {e}")
            raise
    
    def get_user_recommendations(self, user_id: int, model_name: Optional[str] = None,
                               limit: Optional[int] = None) -> List[Recommendation]:
        """Get recommendations for a user."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return []
        
        query = self.db.query(Recommendation).filter(Recommendation.user_id == user.id)
        
        if model_name:
            query = query.filter(Recommendation.model_name == model_name)
        
        query = query.order_by(desc(Recommendation.score))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def delete_user_recommendations(self, user_id: int, model_name: Optional[str] = None) -> int:
        """Delete recommendations for a user."""
        try:
            user = self.db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return 0
            
            query = self.db.query(Recommendation).filter(Recommendation.user_id == user.id)
            
            if model_name:
                query = query.filter(Recommendation.model_name == model_name)
            
            count = query.count()
            query.delete()
            self.db.commit()
            
            logger.info(f"Deleted {count} recommendations for user {user_id}")
            return count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting recommendations: {e}")
            raise
    
    def get_recommendation_stats(self) -> Dict:
        """Get recommendation statistics."""
        total_recommendations = self.db.query(Recommendation).count()
        unique_users = self.db.query(Recommendation.user_id).distinct().count()
        unique_movies = self.db.query(Recommendation.movie_id).distinct().count()
        
        # Get model distribution
        model_counts = self.db.query(
            Recommendation.model_name,
            func.count(Recommendation.id).label('count')
        ).group_by(Recommendation.model_name).all()
        
        return {
            'total_recommendations': total_recommendations,
            'unique_users': unique_users,
            'unique_movies': unique_movies,
            'model_distribution': {model: count for model, count in model_counts}
        }


class ModelPerformanceRepository:
    """Repository for model performance tracking."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def save_performance(self, model_name: str, metric_name: str, metric_value: float,
                        dataset_split: str = "test") -> ModelPerformance:
        """Save model performance metric."""
        try:
            performance = ModelPerformance(
                model_name=model_name,
                metric_name=metric_name,
                metric_value=metric_value,
                dataset_split=dataset_split
            )
            self.db.add(performance)
            self.db.commit()
            self.db.refresh(performance)
            logger.info(f"Saved performance: {model_name} - {metric_name} = {metric_value}")
            return performance
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving performance: {e}")
            raise
    
    def get_model_performance(self, model_name: str, metric_name: Optional[str] = None,
                            dataset_split: Optional[str] = None) -> List[ModelPerformance]:
        """Get performance metrics for a model."""
        query = self.db.query(ModelPerformance).filter(
            ModelPerformance.model_name == model_name
        )
        
        if metric_name:
            query = query.filter(ModelPerformance.metric_name == metric_name)
        
        if dataset_split:
            query = query.filter(ModelPerformance.dataset_split == dataset_split)
        
        return query.order_by(desc(ModelPerformance.timestamp)).all()
    
    def get_latest_performance(self, model_name: str, metric_name: str,
                             dataset_split: str = "test") -> Optional[ModelPerformance]:
        """Get the latest performance metric for a model."""
        return self.db.query(ModelPerformance).filter(
            and_(
                ModelPerformance.model_name == model_name,
                ModelPerformance.metric_name == metric_name,
                ModelPerformance.dataset_split == dataset_split
            )
        ).order_by(desc(ModelPerformance.timestamp)).first() 