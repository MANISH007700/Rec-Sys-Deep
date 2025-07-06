#!/usr/bin/env python3
"""
Script to train and evaluate multiple recommendation models.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Import our modules
from src.data.data_loader import load_movielens_dataset
from src.evaluation.metrics import Evaluator
from src.models.collaborative_filtering import (
    ALSRecommender,
    SVDRecommender,
    UserBasedCF,
)
from src.models.content_based import ContentBasedRecommender, GenreBasedRecommender
from src.models.hybrid import create_hybrid_ensemble
from src.models.neural_cf import NeuralCFRecommender

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create and configure all recommendation models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of model name -> model instance
    """
    models = {}
    
    # Collaborative Filtering models
    cf_config = config['models']['collaborative_filtering']
    models['SVD'] = SVDRecommender(
        n_factors=cf_config['n_factors'],
        random_state=42
    )
    
    models['ALS'] = ALSRecommender(
        n_factors=cf_config['n_factors'],
        n_epochs=cf_config['n_epochs'],
        reg=cf_config['reg'],
        random_state=42
    )
    
    models['UserBasedCF'] = UserBasedCF(
        n_neighbors=50,
        min_similarity=0.1
    )
    
    # Content-based models
    content_config = config['models']['content_based']
    models['ContentBased'] = ContentBasedRecommender(
        max_features=content_config['tfidf_max_features'],
        similarity_metric=content_config['similarity_metric']
    )
    
    models['GenreBased'] = GenreBasedRecommender()
    
    # Neural models
    neural_config = config['models']['neural_cf']
    models['NeuralCF'] = NeuralCFRecommender(
        embedding_dim=neural_config['embedding_dim'],
        hidden_layers=neural_config['hidden_layers'],
        dropout=neural_config['dropout'],
        lr=neural_config['lr'],
        batch_size=neural_config['batch_size'],
        epochs=neural_config['epochs'],
        device='cpu'  # Use CPU for compatibility
    )
    
    return models


def train_and_evaluate_models(models: Dict[str, Any], train_matrix: np.ndarray,
                             test_matrix: np.ndarray, test_ratings: pd.DataFrame,
                             movies_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Train and evaluate all models.
    
    Args:
        models: Dictionary of models to train
        train_matrix: Training interaction matrix
        test_matrix: Test interaction matrix
        test_ratings: Test ratings DataFrame
        movies_df: Movie metadata DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with evaluation results
    """
    evaluator = Evaluator(k_values=config['evaluation']['k_values'])
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Training and evaluating model: {model_name}")
        
        try:
            # Train model
            if model_name in ['ContentBased', 'GenreBased']:
                model.fit(train_matrix, movies_df=movies_df)
            else:
                model.fit(train_matrix)
            
            # Evaluate model
            model_results = evaluator.evaluate_model(
                model, test_matrix, test_ratings, movies_df
            )
            
            # Flatten results
            flat_results = {'model': model_name}
            for metric_type, metrics in model_results.items():
                for metric_name, value in metrics.items():
                    flat_results[f"{metric_type}_{metric_name}"] = value
            
            results.append(flat_results)
            
            # Save model
            model_path = f"models/{model_name.lower()}_model.pkl"
            os.makedirs("models", exist_ok=True)
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training/evaluating {model_name}: {e}")
            continue
    
    return pd.DataFrame(results)


def train_hybrid_models(train_matrix: np.ndarray, test_matrix: np.ndarray,
                       test_ratings: pd.DataFrame, movies_df: pd.DataFrame,
                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Train and evaluate hybrid models.
    
    Args:
        train_matrix: Training interaction matrix
        test_matrix: Test interaction matrix
        test_ratings: Test ratings DataFrame
        movies_df: Movie metadata DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with evaluation results
    """
    evaluator = Evaluator(k_values=config['evaluation']['k_values'])
    results = []
    
    # Create hybrid models
    hybrid_types = ['weighted', 'cascade', 'fusion']
    
    for ensemble_type in hybrid_types:
        logger.info(f"Training hybrid ensemble: {ensemble_type}")
        
        try:
            # Create hybrid model
            hybrid_model = create_hybrid_ensemble(
                train_matrix, movies_df, ensemble_type
            )
            
            # Train model
            hybrid_model.fit(train_matrix, movies_df=movies_df)
            
            # Evaluate model
            model_results = evaluator.evaluate_model(
                hybrid_model, test_matrix, test_ratings, movies_df
            )
            
            # Flatten results
            flat_results = {'model': f'Hybrid_{ensemble_type}'}
            for metric_type, metrics in model_results.items():
                for metric_name, value in metrics.items():
                    flat_results[f"{metric_type}_{metric_name}"] = value
            
            results.append(flat_results)
            
            # Save model
            model_path = f"models/hybrid_{ensemble_type}_model.pkl"
            os.makedirs("models", exist_ok=True)
            hybrid_model.save_model(model_path)
            logger.info(f"Hybrid model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error training hybrid model {ensemble_type}: {e}")
            continue
    
    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Save evaluation results and generate summary.
    
    Args:
        results_df: DataFrame with evaluation results
        config: Configuration dictionary
    """
    # Save results
    results_path = "models/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Generate summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    # Find best model for each metric type
    metric_types = ['rating_metrics', 'ranking_metrics']
    
    for metric_type in metric_types:
        if metric_type in results_df.columns:
            # Get columns for this metric type
            metric_cols = [col for col in results_df.columns if col.startswith(metric_type)]
            
            if metric_cols:
                logger.info(f"\n{metric_type.upper()}:")
                for col in metric_cols:
                    metric_name = col.replace(f"{metric_type}_", "")
                    
                    if 'rmse' in metric_name or 'mae' in metric_name or 'mape' in metric_name:
                        # Lower is better for these metrics
                        best_idx = results_df[col].idxmin()
                        best_value = results_df.loc[best_idx, col]
                        best_model = results_df.loc[best_idx, 'model']
                        logger.info(f"  Best {metric_name}: {best_model} ({best_value:.4f})")
                    else:
                        # Higher is better for other metrics
                        best_idx = results_df[col].idxmax()
                        best_value = results_df.loc[best_idx, col]
                        best_model = results_df.loc[best_idx, 'model']
                        logger.info(f"  Best {metric_name}: {best_model} ({best_value:.4f})")
    
    # Overall best model (using RMSE as primary metric)
    if 'rating_metrics_rmse' in results_df.columns:
        best_overall_idx = results_df['rating_metrics_rmse'].idxmin()
        best_overall_model = results_df.loc[best_overall_idx, 'model']
        best_overall_rmse = results_df.loc[best_overall_idx, 'rating_metrics_rmse']
        logger.info(f"\nOVERALL BEST MODEL: {best_overall_model} (RMSE: {best_overall_rmse:.4f})")
    
    logger.info("="*50)


def main():
    """Main training and evaluation function."""
    logger.info("Starting model training and evaluation...")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Load dataset
        logger.info("Loading dataset...")
        train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata = load_movielens_dataset(
            data_path=config['data']['data_path'],
            dataset_size=config['data']['dataset_name'],
            min_interactions=config['data']['min_interactions'],
            test_size=config['data']['test_size']
        )
        
        logger.info(f"Dataset loaded: {metadata['n_users']} users, {metadata['n_items']} items, {metadata['n_ratings']} ratings")
        
        # Create models
        models = create_models(config)
        logger.info(f"Created {len(models)} models")
        
        # Train and evaluate individual models
        logger.info("Training and evaluating individual models...")
        individual_results = train_and_evaluate_models(
            models, train_matrix, test_matrix, test_ratings, movies_df, config
        )
        
        # Train and evaluate hybrid models
        logger.info("Training and evaluating hybrid models...")
        hybrid_results = train_hybrid_models(
            train_matrix, test_matrix, test_ratings, movies_df, config
        )
        
        # Combine results
        all_results = pd.concat([individual_results, hybrid_results], ignore_index=True)
        
        # Save results and generate summary
        save_results(all_results, config)
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        raise


if __name__ == "__main__":
    main() 