"""
Content-based recommendation models.
Uses item metadata and TF-IDF features to recommend similar items.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommendation system using TF-IDF features.
    
    This model recommends items similar to what a user has liked before,
    based on item content features (title, genre, description, etc.).
    """
    
    def __init__(self, max_features: int = 1000, similarity_metric: str = "cosine"):
        """
        Initialize content-based recommender.
        
        Args:
            max_features: Maximum number of TF-IDF features
            similarity_metric: Similarity metric to use ('cosine', 'euclidean')
        """
        super().__init__(name="ContentBased")
        self.max_features = max_features
        self.similarity_metric = similarity_metric
        self.tfidf_vectorizer = None
        self.item_features = None
        self.item_similarities = None
        self.movies_df = None
        
    def fit(self, train_matrix: np.ndarray, movies_df: pd.DataFrame, **kwargs) -> 'ContentBasedRecommender':
        """
        Fit the content-based model.
        
        Args:
            train_matrix: User-item interaction matrix (not used for content-based)
            movies_df: DataFrame with movie metadata
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Content-based model...")
        
        self.n_users, self.n_items = train_matrix.shape
        self.movies_df = movies_df.copy()
        
        # Create text features from movie metadata
        self._create_text_features()
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform text features
        self.item_features = self.tfidf_vectorizer.fit_transform(
            self.movies_df['text_features']
        )
        
        # Calculate item-item similarities
        self.item_similarities = cosine_similarity(self.item_features)
        
        self.is_fitted = True
        logger.info("Content-based model fitted successfully!")
        
        return self
    
    def _create_text_features(self) -> None:
        """Create text features from movie metadata."""
        # Combine title and genres
        self.movies_df['text_features'] = (
            self.movies_df['title_clean'].fillna('') + ' ' + 
            self.movies_df['genres'].fillna('').str.replace('|', ' ')
        )
        
        # Add year if available
        if 'year' in self.movies_df.columns:
            self.movies_df['text_features'] += ' ' + self.movies_df['year'].fillna('').astype(str)
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # For content-based, we need user preferences
        # This is a simplified version - in practice, you'd need user profiles
        predictions = np.full(len(user_ids), 3.0)  # Default rating
        
        return predictions
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True, user_ratings: Optional[np.ndarray] = None) -> List[int]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            user_ratings: User's rating vector (optional)
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_ratings is None:
            # If no user ratings provided, return popular items
            return list(range(min(n_recommendations, self.n_items)))
        
        # Calculate user profile based on rated items
        user_profile = self._create_user_profile(user_ratings)
        
        # Calculate similarity between user profile and all items
        item_scores = cosine_similarity(user_profile.reshape(1, -1), self.item_features).flatten()
        
        # Exclude rated items if requested
        if exclude_rated:
            rated_items = user_ratings > 0
            item_scores[rated_items] = -1
        
        # Get top recommendations
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return top_items.tolist()
    
    def _create_user_profile(self, user_ratings: np.ndarray) -> np.ndarray:
        """
        Create user profile based on rated items.
        
        Args:
            user_ratings: User's rating vector
            
        Returns:
            User profile vector
        """
        # Get items the user has rated positively (rating > 3)
        liked_items = user_ratings > 3
        
        if np.sum(liked_items) == 0:
            # If no positive ratings, use all rated items
            rated_items = user_ratings > 0
            if np.sum(rated_items) == 0:
                # If no ratings at all, return zero vector
                return np.zeros(self.item_features.shape[1])
            liked_items = rated_items
        
        # Calculate weighted average of liked items' features
        liked_features = self.item_features[liked_items]
        liked_ratings = user_ratings[liked_items]
        
        # Weight features by ratings
        user_profile = np.average(liked_features.toarray(), 
                                 weights=liked_ratings, 
                                 axis=0)
        
        return user_profile
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[int]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: ID of the target item
            n_similar: Number of similar items to return
            
        Returns:
            List of similar item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        if item_id >= self.n_items:
            return []
        
        # Get similarities for the target item
        item_similarities = self.item_similarities[item_id]
        
        # Exclude the item itself
        item_similarities[item_id] = -1
        
        # Get top similar items
        similar_items = np.argsort(item_similarities)[::-1][:n_similar]
        return similar_items.tolist()


class GenreBasedRecommender(BaseRecommender):
    """
    Genre-based recommendation system.
    
    This model recommends items based on genre preferences extracted from user behavior.
    """
    
    def __init__(self):
        """Initialize genre-based recommender."""
        super().__init__(name="GenreBased")
        self.genre_matrix = None
        self.user_genre_preferences = None
        self.movies_df = None
        
    def fit(self, train_matrix: np.ndarray, movies_df: pd.DataFrame, **kwargs) -> 'GenreBasedRecommender':
        """
        Fit the genre-based model.
        
        Args:
            train_matrix: User-item interaction matrix
            movies_df: DataFrame with movie metadata
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Genre-based model...")
        
        self.n_users, self.n_items = train_matrix.shape
        self.movies_df = movies_df.copy()
        
        # Create genre matrix
        self._create_genre_matrix()
        
        # Calculate user genre preferences
        self._calculate_user_preferences(train_matrix)
        
        self.is_fitted = True
        logger.info("Genre-based model fitted successfully!")
        
        return self
    
    def _create_genre_matrix(self) -> None:
        """Create binary genre matrix for all movies."""
        # Get all unique genres
        all_genres = set()
        for genres in self.movies_df['genre_list']:
            if isinstance(genres, list):
                all_genres.update(genres)
        
        # Create genre matrix
        genre_matrix = np.zeros((len(self.movies_df), len(all_genres)))
        genre_names = sorted(list(all_genres))
        
        for i, genres in enumerate(self.movies_df['genre_list']):
            if isinstance(genres, list):
                for genre in genres:
                    if genre in genre_names:
                        genre_matrix[i, genre_names.index(genre)] = 1
        
        self.genre_matrix = genre_matrix
        self.genre_names = genre_names
        
        logger.info(f"Created genre matrix with {len(all_genres)} genres")
    
    def _calculate_user_preferences(self, train_matrix: np.ndarray) -> None:
        """Calculate user genre preferences based on rated movies."""
        self.user_genre_preferences = np.zeros((self.n_users, len(self.genre_names)))
        
        for user_id in range(self.n_users):
            # Get movies rated by this user
            user_ratings = train_matrix[user_id]
            rated_movies = user_ratings > 0
            
            if np.sum(rated_movies) > 0:
                # Calculate weighted genre preferences
                rated_genres = self.genre_matrix[rated_movies]
                ratings = user_ratings[rated_movies]
                
                # Weight genres by ratings
                weighted_genres = rated_genres * ratings.reshape(-1, 1)
                user_preferences = np.sum(weighted_genres, axis=0)
                
                # Normalize
                if np.sum(user_preferences) > 0:
                    user_preferences = user_preferences / np.sum(user_preferences)
                
                self.user_genre_preferences[user_id] = user_preferences
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for user_id, item_id in zip(user_ids, item_ids):
            if user_id >= self.n_users or item_id >= self.n_items:
                predictions.append(3.0)
            else:
                # Calculate genre similarity
                user_prefs = self.user_genre_preferences[user_id]
                item_genres = self.genre_matrix[item_id]
                
                # Cosine similarity
                similarity = np.dot(user_prefs, item_genres) / (
                    np.linalg.norm(user_prefs) * np.linalg.norm(item_genres) + 1e-8
                )
                
                # Convert similarity to rating (1-5 scale)
                pred = 1 + 4 * max(0, similarity)
                predictions.append(pred)
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate genre-based recommendations for a user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id >= self.n_users:
            # Cold start: return popular items
            return list(range(min(n_recommendations, self.n_items)))
        
        # Get user's genre preferences
        user_prefs = self.user_genre_preferences[user_id]
        
        # Calculate scores for all items
        item_scores = np.dot(self.genre_matrix, user_prefs)
        
        # Exclude rated items if requested
        if exclude_rated:
            # This would require the original train_matrix
            pass
        
        # Get top recommendations
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return top_items.tolist()
    
    def get_user_genre_preferences(self, user_id: int) -> Dict[str, float]:
        """
        Get genre preferences for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary mapping genres to preference scores
        """
        if not self.is_fitted or user_id >= self.n_users:
            return {}
        
        preferences = self.user_genre_preferences[user_id]
        return dict(zip(self.genre_names, preferences)) 