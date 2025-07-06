#!/usr/bin/env python3
"""
Simple test script to verify the recommendation system components.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

import numpy as np
import pandas as pd

# Import our modules
from src.data.data_loader import load_movielens_dataset
from src.evaluation.metrics import Evaluator
from src.models.collaborative_filtering import SVDRecommender
from src.models.content_based import ContentBasedRecommender

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        # Load a small subset for testing
        train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata = load_movielens_dataset(
            data_path="data/",
            dataset_size="100k",
            min_interactions=3,  # Lower threshold for testing
            test_size=0.1  # Smaller test set
        )
        
        logger.info(f"Data loaded successfully!")
        logger.info(f"Train matrix shape: {train_matrix.shape}")
        logger.info(f"Test matrix shape: {test_matrix.shape}")
        logger.info(f"Movies DataFrame shape: {movies_df.shape}")
        logger.info(f"Metadata: {metadata}")
        
        return train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise


def test_model_training(train_matrix, movies_df):
    """Test model training functionality."""
    logger.info("Testing model training...")
    
    try:
        # Test SVD model
        logger.info("Training SVD model...")
        svd_model = SVDRecommender(n_factors=20)  # Smaller for testing
        svd_model.fit(train_matrix)
        
        # Test content-based model
        logger.info("Training Content-based model...")
        content_model = ContentBasedRecommender(max_features=100)  # Smaller for testing
        content_model.fit(train_matrix, movies_df=movies_df)
        
        logger.info("Model training completed successfully!")
        
        return svd_model, content_model
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def test_recommendations(svd_model, content_model, test_matrix):
    """Test recommendation generation."""
    logger.info("Testing recommendation generation...")
    
    try:
        # Test recommendations for a few users
        test_users = [0, 1, 2]
        
        for user_id in test_users:
            if user_id < svd_model.n_users:
                # SVD recommendations
                svd_recs = svd_model.recommend(user_id, n_recommendations=5)
                logger.info(f"User {user_id} - SVD recommendations: {svd_recs}")
                
                # Content-based recommendations (using user's ratings)
                user_ratings = test_matrix[user_id]
                content_recs = content_model.recommend(user_id, n_recommendations=5, user_ratings=user_ratings)
                logger.info(f"User {user_id} - Content recommendations: {content_recs}")
        
        logger.info("Recommendation generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in recommendation generation: {e}")
        raise


def test_evaluation(test_matrix, test_ratings, movies_df, svd_model):
    """Test evaluation functionality."""
    logger.info("Testing evaluation...")
    
    try:
        evaluator = Evaluator(k_values=[5, 10])
        
        # Evaluate SVD model
        results = evaluator.evaluate_model(svd_model, test_matrix, test_ratings, movies_df)
        
        logger.info("Evaluation results:")
        for metric_type, metrics in results.items():
            logger.info(f"  {metric_type}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise


def main():
    """Main test function."""
    logger.info("Starting recommendation system test...")
    
    try:
        # Test data loading
        train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata = test_data_loading()
        
        # Test model training
        svd_model, content_model = test_model_training(train_matrix, movies_df)
        
        # Test recommendations
        test_recommendations(svd_model, content_model, test_matrix)
        
        # Test evaluation
        test_evaluation(test_matrix, test_ratings, movies_df, svd_model)
        
        logger.info("All tests completed successfully!")
        logger.info("The recommendation system is working correctly!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 