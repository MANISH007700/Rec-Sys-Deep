"""
Base model class for recommendation systems.
Defines the interface that all recommendation models must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.
    
    This class defines the interface that all recommendation models must implement,
    ensuring consistency across different algorithms.
    """
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize the base recommender.
        
        Args:
            name: Name of the model for identification
        """
        self.name = name
        self.is_fitted = False
        self.n_users = None
        self.n_items = None
        
    @abstractmethod
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'BaseRecommender':
        """
        Fit the model to the training data.
        
        Args:
            train_matrix: User-item interaction matrix (n_users x n_items)
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Array of predicted ratings
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            List of recommended item IDs
        """
        pass
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10,
                                exclude_rated: bool = True) -> Dict[str, Any]:
        """
        Get recommendations for a user with additional metadata.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude rated items
            
        Returns:
            Dictionary with recommendations and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = self.recommend(user_id, n_recommendations, exclude_rated)
        
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'n_recommendations': len(recommendations),
            'model_name': self.name
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseRecommender':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model 