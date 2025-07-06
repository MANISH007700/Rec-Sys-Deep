#!/usr/bin/env python3
"""
Script to download and prepare MovieLens dataset for the recommendation system.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.data.data_loader import load_movielens_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to download and prepare the dataset."""
    logger.info("Starting dataset preparation...")
    
    try:
        # Load MovieLens 100k dataset
        logger.info("Loading MovieLens 100k dataset...")
        train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata = load_movielens_dataset(
            data_path="data/",
            dataset_size="100k",
            min_interactions=5,
            test_size=0.2
        )
        
        # Print dataset summary
        logger.info("Dataset loaded successfully!")
        logger.info(f"Number of users: {metadata['n_users']}")
        logger.info(f"Number of movies: {metadata['n_items']}")
        logger.info(f"Number of ratings: {metadata['n_ratings']}")
        logger.info(f"Average rating: {metadata['avg_rating']:.2f}")
        logger.info(f"Sparsity: {metadata['sparsity']:.4f}")
        logger.info(f"Train set size: {len(train_ratings)}")
        logger.info(f"Test set size: {len(test_ratings)}")
        
        # Save processed data
        logger.info("Saving processed data...")
        os.makedirs("data/processed", exist_ok=True)
        
        train_ratings.to_csv("data/processed/train_ratings.csv", index=False)
        test_ratings.to_csv("data/processed/test_ratings.csv", index=False)
        movies_df.to_csv("data/processed/movies_processed.csv", index=False)
        
        # Save matrices as numpy arrays
        import numpy as np
        np.save("data/processed/train_matrix.npy", train_matrix)
        np.save("data/processed/test_matrix.npy", test_matrix)
        np.save("data/processed/content_features.npy", content_features.toarray())
        
        logger.info("Data preparation completed successfully!")
        logger.info("Files saved in data/processed/")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main() 