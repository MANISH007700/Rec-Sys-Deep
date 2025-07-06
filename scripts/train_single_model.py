#!/usr/bin/env python3
"""
Simplified training script that trains only the SVD model for quick testing.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import yaml

from src.data.data_loader import MovieLensDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train only the SVD model for quick testing."""
    logger.info("Starting simplified model training (SVD only)...")
    
    try:
        # Load configuration
        config_path = "config/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        # Load dataset directly without downloading
        logger.info("Loading MovieLens dataset...")
        loader = MovieLensDataLoader(
            data_path=config['data']['data_path'],
            min_interactions=config['data']['min_interactions']
        )
        
        # Load existing data without downloading
        loader.load_data("100k")
        loader.preprocess_data()
        
        # Get train/test split
        train_ratings, test_ratings, train_matrix, test_matrix = loader.get_train_test_split(
            test_size=config['data']['test_size']
        )
        
        # Get metadata
        metadata = loader.get_data_summary()
        logger.info(f"Dataset loaded: {metadata['n_users']} users, {metadata['n_items']} items, {metadata['n_ratings']} ratings")
        
        # Create and train SVD model
        logger.info("Training SVD model...")
        from src.models.collaborative_filtering import SVDRecommender
        
        svd_model = SVDRecommender(
            n_factors=50,  # Reduced for faster training
            random_state=42
        )
        
        svd_model.fit(train_matrix)
        
        # Save model
        model_path = "models/svd_model.pkl"
        os.makedirs("models", exist_ok=True)
        svd_model.save_model(model_path)
        logger.info(f"SVD model saved to {model_path}")
        
        # Test the model
        logger.info("Testing SVD model...")
        
        # Test predictions
        test_users = test_ratings['user_id_encoded'].values[:10]  # Test with first 10 users
        test_items = test_ratings['item_id_encoded'].values[:10]
        predictions = svd_model.predict(test_users, test_items)
        
        logger.info(f"Sample predictions: {predictions[:5]}")
        
        # Test recommendations
        user_id = 1
        recommendations = svd_model.recommend(user_id, n_recommendations=5)
        logger.info(f"Sample recommendations for user {user_id}: {recommendations}")
        
        logger.info("SVD model training and testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 