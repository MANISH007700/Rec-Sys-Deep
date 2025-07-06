"""
Hybrid recommendation models.
Combines multiple recommendation approaches for better performance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_model import BaseRecommender
from .collaborative_filtering import ALSRecommender, SVDRecommender
from .content_based import ContentBasedRecommender, GenreBasedRecommender
from .neural_cf import NeuralCFRecommender

logger = logging.getLogger(__name__)


class WeightedEnsembleRecommender(BaseRecommender):
    """
    Weighted ensemble of multiple recommendation models.
    
    This model combines predictions from different recommendation algorithms
    using weighted averaging for improved performance.
    """
    
    def __init__(self, models: List[BaseRecommender], weights: Optional[List[float]] = None):
        """
        Initialize weighted ensemble recommender.
        
        Args:
            models: List of recommendation models
            weights: Weights for each model (if None, equal weights are used)
        """
        super().__init__(name="WeightedEnsemble")
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'WeightedEnsembleRecommender':
        """
        Fit all models in the ensemble.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments passed to each model
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ensemble with {len(self.models)} models...")
        
        self.n_users, self.n_items = train_matrix.shape
        
        # Fit each model
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {model.name}")
            
            # Pass additional arguments if available
            model_kwargs = {}
            if 'movies_df' in kwargs and hasattr(model, 'fit'):
                model_kwargs['movies_df'] = kwargs['movies_df']
            
            model.fit(train_matrix, **model_kwargs)
        
        self.is_fitted = True
        logger.info("Ensemble fitted successfully!")
        
        return self
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings using weighted ensemble.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(user_ids, item_ids)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations using weighted ensemble.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from all models
        all_recommendations = []
        for model in self.models:
            recs = model.recommend(user_id, n_recommendations, exclude_rated)
            all_recommendations.extend(recs)
        
        # Count occurrences of each item
        item_counts = {}
        for item_id in all_recommendations:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        # Sort by count (descending) and return top items
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, count in sorted_items[:n_recommendations]]
        
        return recommendations
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the weights assigned to each model."""
        return {model.name: weight for model, weight in zip(self.models, self.weights)}


class CascadeRecommender(BaseRecommender):
    """
    Cascade recommendation system.
    
    This model uses a cascade approach where one model filters items
    and another model ranks the filtered items.
    """
    
    def __init__(self, filter_model: BaseRecommender, rank_model: BaseRecommender,
                 filter_ratio: float = 0.1):
        """
        Initialize cascade recommender.
        
        Args:
            filter_model: Model used for initial filtering
            rank_model: Model used for final ranking
            filter_ratio: Ratio of items to keep after filtering
        """
        super().__init__(name="Cascade")
        self.filter_model = filter_model
        self.rank_model = rank_model
        self.filter_ratio = filter_ratio
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'CascadeRecommender':
        """
        Fit both models in the cascade.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting cascade model...")
        
        self.n_users, self.n_items = train_matrix.shape
        
        # Fit filter model
        logger.info(f"Fitting filter model: {self.filter_model.name}")
        self.filter_model.fit(train_matrix, **kwargs)
        
        # Fit rank model
        logger.info(f"Fitting rank model: {self.rank_model.name}")
        self.rank_model.fit(train_matrix, **kwargs)
        
        self.is_fitted = True
        logger.info("Cascade model fitted successfully!")
        
        return self
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings using cascade approach.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use rank model for predictions
        return self.rank_model.predict(user_ids, item_ids)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations using cascade approach.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Step 1: Filter items using filter model
        n_filter = max(1, int(self.n_items * self.filter_ratio))
        filtered_items = self.filter_model.recommend(user_id, n_filter, exclude_rated)
        
        # Step 2: Rank filtered items using rank model
        if len(filtered_items) == 0:
            return []
        
        # Create a temporary user-item matrix for ranking
        temp_matrix = np.zeros((1, self.n_items))
        temp_matrix[0, filtered_items] = 1  # Mark filtered items as "rated"
        
        # Get scores for filtered items
        item_scores = []
        for item_id in filtered_items:
            score = self.rank_model.predict(np.array([user_id]), np.array([item_id]))[0]
            item_scores.append((item_id, score))
        
        # Sort by score and return top recommendations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in item_scores[:n_recommendations]]
        
        return recommendations


class FeatureFusionRecommender(BaseRecommender):
    """
    Feature fusion recommendation system.
    
    This model combines collaborative filtering features with content-based features
    in a unified model.
    """
    
    def __init__(self, cf_model: BaseRecommender, content_model: BaseRecommender,
                 fusion_weight: float = 0.5):
        """
        Initialize feature fusion recommender.
        
        Args:
            cf_model: Collaborative filtering model
            content_model: Content-based model
            fusion_weight: Weight for collaborative filtering (content weight = 1 - fusion_weight)
        """
        super().__init__(name="FeatureFusion")
        self.cf_model = cf_model
        self.content_model = content_model
        self.fusion_weight = fusion_weight
        self.content_weight = 1 - fusion_weight
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'FeatureFusionRecommender':
        """
        Fit both models for feature fusion.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature fusion model...")
        
        self.n_users, self.n_items = train_matrix.shape
        
        # Fit collaborative filtering model
        logger.info(f"Fitting CF model: {self.cf_model.name}")
        self.cf_model.fit(train_matrix, **kwargs)
        
        # Fit content-based model
        logger.info(f"Fitting content model: {self.content_model.name}")
        self.content_model.fit(train_matrix, **kwargs)
        
        self.is_fitted = True
        logger.info("Feature fusion model fitted successfully!")
        
        return self
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings using feature fusion.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from both models
        cf_predictions = self.cf_model.predict(user_ids, item_ids)
        content_predictions = self.content_model.predict(user_ids, item_ids)
        
        # Weighted fusion
        fused_predictions = (self.fusion_weight * cf_predictions + 
                           self.content_weight * content_predictions)
        
        return fused_predictions
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations using feature fusion.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        cf_recommendations = self.cf_model.recommend(user_id, n_recommendations, exclude_rated)
        content_recommendations = self.content_model.recommend(user_id, n_recommendations, exclude_rated)
        
        # Combine and rank recommendations
        all_items = set(cf_recommendations + content_recommendations)
        
        # Calculate scores for all items
        item_scores = []
        for item_id in all_items:
            cf_score = self.cf_model.predict(np.array([user_id]), np.array([item_id]))[0]
            content_score = self.content_model.predict(np.array([user_id]), np.array([item_id]))[0]
            
            fused_score = (self.fusion_weight * cf_score + 
                          self.content_weight * content_score)
            item_scores.append((item_id, fused_score))
        
        # Sort by score and return top recommendations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in item_scores[:n_recommendations]]
        
        return recommendations


def create_hybrid_ensemble(train_matrix: np.ndarray, movies_df: pd.DataFrame,
                          ensemble_type: str = "weighted") -> BaseRecommender:
    """
    Create a hybrid ensemble with multiple recommendation models.
    
    Args:
        train_matrix: Training interaction matrix
        movies_df: Movie metadata DataFrame
        ensemble_type: Type of ensemble ('weighted', 'cascade', 'fusion')
        
    Returns:
        Configured hybrid recommender
    """
    # Create base models
    svd_model = SVDRecommender(n_factors=50)
    als_model = ALSRecommender(n_factors=50, n_epochs=10)
    content_model = ContentBasedRecommender(max_features=500)
    genre_model = GenreBasedRecommender()
    
    if ensemble_type == "weighted":
        # Weighted ensemble
        models = [svd_model, als_model, content_model, genre_model]
        weights = [0.4, 0.3, 0.2, 0.1]  # Give more weight to CF models
        return WeightedEnsembleRecommender(models, weights)
    
    elif ensemble_type == "cascade":
        # Cascade: content-based filter + CF ranker
        return CascadeRecommender(content_model, svd_model, filter_ratio=0.2)
    
    elif ensemble_type == "fusion":
        # Feature fusion: CF + content
        return FeatureFusionRecommender(svd_model, content_model, fusion_weight=0.7)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}") 