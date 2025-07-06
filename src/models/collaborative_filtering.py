"""
Collaborative Filtering models for recommendation systems.
Implements SVD and ALS matrix factorization algorithms.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """
    Singular Value Decomposition (SVD) based collaborative filtering.
    
    This model uses SVD to decompose the user-item interaction matrix into
    lower-dimensional user and item latent factors.
    """
    
    def __init__(self, n_factors: int = 100, random_state: int = 42):
        """
        Initialize SVD recommender.
        
        Args:
            n_factors: Number of latent factors
            random_state: Random seed for reproducibility
        """
        super().__init__(name="SVD")
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'SVDRecommender':
        """
        Fit the SVD model to the training data.
        
        Args:
            train_matrix: User-item interaction matrix (n_users x n_items)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting SVD model with {self.n_factors} factors...")
        
        self.n_users, self.n_items = train_matrix.shape
        self.global_mean = np.mean(train_matrix[train_matrix > 0])
        
        # Center the data
        centered_matrix = train_matrix.copy()
        centered_matrix[centered_matrix == 0] = self.global_mean
        centered_matrix = centered_matrix - self.global_mean
        
        # Apply SVD
        self.svd = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        
        # Fit SVD on the centered matrix
        self.svd.fit(centered_matrix.T)  # Transpose for item-based decomposition
        
        # Get user and item factors
        self.item_factors = self.svd.components_.T  # (n_items, n_factors)
        self.user_factors = self.svd.transform(centered_matrix)  # (n_users, n_factors)
        
        # Calculate biases
        self.user_biases = np.mean(centered_matrix, axis=1)  # User biases
        self.item_biases = np.mean(centered_matrix, axis=0)  # Item biases
        
        self.is_fitted = True
        logger.info("SVD model fitted successfully!")
        
        return self
    
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
                predictions.append(self.global_mean)
            else:
                pred = (self.global_mean + 
                       self.user_biases[user_id] + 
                       self.item_biases[item_id] +
                       np.dot(self.user_factors[user_id], self.item_factors[item_id]))
                predictions.append(max(1, min(5, pred)))  # Clip to rating range
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations for a specific user.
        
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
        
        # Calculate predicted ratings for all items
        user_factors = self.user_factors[user_id]
        item_scores = (self.global_mean + 
                      self.user_biases[user_id] + 
                      self.item_biases + 
                      np.dot(self.item_factors, user_factors))
        
        # Get top recommendations
        if exclude_rated:
            # This would require the original train_matrix to exclude rated items
            # For now, we'll return top items
            pass
        
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return top_items.tolist()


class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) matrix factorization.
    
    This model uses ALS optimization to find user and item latent factors
    by alternating between fixing user factors and item factors.
    """
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, 
                 reg: float = 0.1, random_state: int = 42):
        """
        Initialize ALS recommender.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            reg: Regularization parameter
            random_state: Random seed for reproducibility
        """
        super().__init__(name="ALS")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'ALSRecommender':
        """
        Fit the ALS model to the training data.
        
        Args:
            train_matrix: User-item interaction matrix (n_users x n_items)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ALS model with {self.n_factors} factors for {self.n_epochs} epochs...")
        
        self.n_users, self.n_items = train_matrix.shape
        self.global_mean = np.mean(train_matrix[train_matrix > 0])
        
        # Initialize factors randomly
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        # Center the data
        centered_matrix = train_matrix.copy()
        centered_matrix[centered_matrix == 0] = self.global_mean
        centered_matrix = centered_matrix - self.global_mean
        
        # Create sparse matrix for efficiency
        sparse_matrix = csr_matrix(centered_matrix)
        
        # ALS optimization
        for epoch in range(self.n_epochs):
            # Update user factors
            for i in range(self.n_users):
                items_rated = sparse_matrix[i].nonzero()[1]
                if len(items_rated) > 0:
                    item_factors_subset = self.item_factors[items_rated]
                    ratings_subset = sparse_matrix[i, items_rated].toarray().flatten()
                    
                    # Solve least squares problem
                    A = item_factors_subset.T @ item_factors_subset + self.reg * np.eye(self.n_factors)
                    b = item_factors_subset.T @ ratings_subset
                    self.user_factors[i] = np.linalg.solve(A, b)
            
            # Update item factors
            for j in range(self.n_items):
                users_rated = sparse_matrix[:, j].nonzero()[0]
                if len(users_rated) > 0:
                    user_factors_subset = self.user_factors[users_rated]
                    ratings_subset = sparse_matrix[users_rated, j].toarray().flatten()
                    
                    # Solve least squares problem
                    A = user_factors_subset.T @ user_factors_subset + self.reg * np.eye(self.n_factors)
                    b = user_factors_subset.T @ ratings_subset
                    self.item_factors[j] = np.linalg.solve(A, b)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"ALS epoch {epoch + 1}/{self.n_epochs} completed")
        
        self.is_fitted = True
        logger.info("ALS model fitted successfully!")
        
        return self
    
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
                predictions.append(self.global_mean)
            else:
                pred = (self.global_mean + 
                       np.dot(self.user_factors[user_id], self.item_factors[item_id]))
                predictions.append(max(1, min(5, pred)))  # Clip to rating range
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations for a specific user.
        
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
        
        # Calculate predicted ratings for all items
        user_factors = self.user_factors[user_id]
        item_scores = (self.global_mean + 
                      np.dot(self.item_factors, user_factors))
        
        # Get top recommendations
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return top_items.tolist()


class UserBasedCF(BaseRecommender):
    """
    User-based Collaborative Filtering using cosine similarity.
    
    This model finds similar users and recommends items that similar users have rated highly.
    """
    
    def __init__(self, n_neighbors: int = 50, min_similarity: float = 0.1):
        """
        Initialize user-based CF.
        
        Args:
            n_neighbors: Number of similar users to consider
            min_similarity: Minimum similarity threshold
        """
        super().__init__(name="UserBasedCF")
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.train_matrix = None
        self.user_similarities = None
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'UserBasedCF':
        """
        Fit the user-based CF model.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting User-based CF model...")
        
        self.n_users, self.n_items = train_matrix.shape
        self.train_matrix = train_matrix.copy()
        
        # Calculate user similarities
        # Use cosine similarity on non-zero ratings
        user_similarities = cosine_similarity(train_matrix)
        
        # Set diagonal to 0 (user not similar to itself)
        np.fill_diagonal(user_similarities, 0)
        
        # Store similarities
        self.user_similarities = user_similarities
        
        self.is_fitted = True
        logger.info("User-based CF model fitted successfully!")
        
        return self
    
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
                predictions.append(3.0)  # Default rating
                continue
            
            # Find similar users who rated this item
            similar_users = np.argsort(self.user_similarities[user_id])[::-1][:self.n_neighbors]
            item_ratings = self.train_matrix[similar_users, item_id]
            similarities = self.user_similarities[user_id][similar_users]
            
            # Filter out users who haven't rated the item or have low similarity
            valid_mask = (item_ratings > 0) & (similarities > self.min_similarity)
            
            if np.sum(valid_mask) == 0:
                predictions.append(3.0)  # Default rating
            else:
                valid_ratings = item_ratings[valid_mask]
                valid_similarities = similarities[valid_mask]
                
                # Weighted average
                pred = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
                predictions.append(max(1, min(5, pred)))
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations for a specific user.
        
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
        
        # Find similar users
        similar_users = np.argsort(self.user_similarities[user_id])[::-1][:self.n_neighbors]
        
        # Calculate item scores based on similar users' ratings
        item_scores = np.zeros(self.n_items)
        item_counts = np.zeros(self.n_items)
        
        for similar_user in similar_users:
            similarity = self.user_similarities[user_id, similar_user]
            if similarity > self.min_similarity:
                user_ratings = self.train_matrix[similar_user]
                item_scores += similarity * user_ratings
                item_counts += similarity * (user_ratings > 0)
        
        # Normalize scores
        item_scores = np.where(item_counts > 0, item_scores / item_counts, 0)
        
        # Exclude rated items if requested
        if exclude_rated:
            rated_items = self.train_matrix[user_id] > 0
            item_scores[rated_items] = -1
        
        # Get top recommendations
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return top_items.tolist() 