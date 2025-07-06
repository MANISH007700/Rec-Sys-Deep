"""
Evaluation metrics for recommendation systems.
Implements both rating and ranking metrics for comprehensive evaluation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class RatingMetrics:
    """
    Rating-based evaluation metrics.
    
    These metrics evaluate how well the model predicts explicit ratings.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all rating metrics.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            Dictionary with all rating metrics
        """
        return {
            'rmse': RatingMetrics.rmse(y_true, y_pred),
            'mae': RatingMetrics.mae(y_true, y_pred),
            'mape': RatingMetrics.mape(y_true, y_pred)
        }


class RankingMetrics:
    """
    Ranking-based evaluation metrics.
    
    These metrics evaluate how well the model ranks items for users.
    """
    
    @staticmethod
    def precision_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Precision@k.
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            Precision@k value
        """
        precisions = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(pred_items) == 0:
                precisions.append(0.0)
                continue
            
            # Get top-k predictions
            top_k_pred = pred_items[:k]
            
            # Calculate precision
            relevant_in_top_k = len(set(top_k_pred) & set(true_items))
            precision = relevant_in_top_k / len(top_k_pred)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    @staticmethod
    def recall_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Recall@k.
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            Recall@k value
        """
        recalls = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                recalls.append(0.0)
                continue
            
            # Get top-k predictions
            top_k_pred = pred_items[:k]
            
            # Calculate recall
            relevant_in_top_k = len(set(top_k_pred) & set(true_items))
            recall = relevant_in_top_k / len(true_items)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    @staticmethod
    def f1_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate F1@k.
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            F1@k value
        """
        precision = RankingMetrics.precision_at_k(y_true, y_pred, k)
        recall = RankingMetrics.recall_at_k(y_true, y_pred, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            NDCG@k value
        """
        ndcgs = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(pred_items) == 0:
                ndcgs.append(0.0)
                continue
            
            # Get top-k predictions
            top_k_pred = pred_items[:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
            
            # Calculate IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(true_items), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Calculate NDCG
            if idcg == 0:
                ndcg = 0.0
            else:
                ndcg = dcg / idcg
            
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    @staticmethod
    def map_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Mean Average Precision (MAP@k).
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            MAP@k value
        """
        aps = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                aps.append(0.0)
                continue
            
            # Get top-k predictions
            top_k_pred = pred_items[:k]
            
            # Calculate average precision
            relevant_count = 0
            precision_sum = 0.0
            
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precision_sum += precision_at_i
            
            if relevant_count == 0:
                ap = 0.0
            else:
                ap = precision_sum / len(true_items)
            
            aps.append(ap)
        
        return np.mean(aps)
    
    @staticmethod
    def mrr_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR@k).
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k: Number of top items to consider
            
        Returns:
            MRR@k value
        """
        reciprocal_ranks = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                reciprocal_ranks.append(0.0)
                continue
            
            # Get top-k predictions
            top_k_pred = pred_items[:k]
            
            # Find first relevant item
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def calculate_all(y_true: List[List[int]], y_pred: List[List[int]], 
                     k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Calculate all ranking metrics for multiple k values.
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with all ranking metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = RankingMetrics.precision_at_k(y_true, y_pred, k)
            metrics[f'recall@{k}'] = RankingMetrics.recall_at_k(y_true, y_pred, k)
            metrics[f'f1@{k}'] = RankingMetrics.f1_at_k(y_true, y_pred, k)
            metrics[f'ndcg@{k}'] = RankingMetrics.ndcg_at_k(y_true, y_pred, k)
            metrics[f'map@{k}'] = RankingMetrics.map_at_k(y_true, y_pred, k)
            metrics[f'mrr@{k}'] = RankingMetrics.mrr_at_k(y_true, y_pred, k)
        
        return metrics


class DiversityMetrics:
    """
    Diversity and novelty metrics for recommendation systems.
    """
    
    @staticmethod
    def intra_list_similarity(recommendations: List[List[int]], 
                             item_features: np.ndarray) -> float:
        """
        Calculate intra-list similarity (lower is better for diversity).
        
        Args:
            recommendations: List of recommendation lists for each user
            item_features: Item feature matrix
            
        Returns:
            Average intra-list similarity
        """
        similarities = []
        
        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue
            
            # Calculate pairwise similarities within the list
            list_similarities = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item1, item2 = rec_list[i], rec_list[j]
                    if item1 < item_features.shape[0] and item2 < item_features.shape[0]:
                        sim = np.dot(item_features[item1], item_features[item2]) / (
                            np.linalg.norm(item_features[item1]) * np.linalg.norm(item_features[item2]) + 1e-8
                        )
                        list_similarities.append(sim)
            
            if list_similarities:
                similarities.append(np.mean(list_similarities))
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def coverage(recommendations: List[List[int]], n_items: int) -> float:
        """
        Calculate recommendation coverage.
        
        Args:
            recommendations: List of recommendation lists for each user
            n_items: Total number of items
            
        Returns:
            Coverage percentage
        """
        all_recommended = set()
        for rec_list in recommendations:
            all_recommended.update(rec_list)
        
        return len(all_recommended) / n_items * 100
    
    @staticmethod
    def novelty(recommendations: List[List[int]], item_popularity: Dict[int, int]) -> float:
        """
        Calculate recommendation novelty (lower popularity = higher novelty).
        
        Args:
            recommendations: List of recommendation lists for each user
            item_popularity: Dictionary mapping item IDs to popularity counts
            
        Returns:
            Average novelty score
        """
        novelty_scores = []
        
        for rec_list in recommendations:
            list_novelty = []
            for item in rec_list:
                popularity = item_popularity.get(item, 0)
                # Convert to novelty (inverse of popularity)
                novelty = 1.0 / (popularity + 1)  # +1 to avoid division by zero
                list_novelty.append(novelty)
            
            if list_novelty:
                novelty_scores.append(np.mean(list_novelty))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0


class Evaluator:
    """
    Comprehensive evaluator for recommendation systems.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of k values for ranking metrics
        """
        self.k_values = k_values
    
    def evaluate_rating_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate rating predictions.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            Dictionary with rating metrics
        """
        return RatingMetrics.calculate_all(y_true, y_pred)
    
    def evaluate_ranking_predictions(self, y_true: List[List[int]], 
                                   y_pred: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate ranking predictions.
        
        Args:
            y_true: List of true relevant items for each user
            y_pred: List of predicted items for each user
            
        Returns:
            Dictionary with ranking metrics
        """
        return RankingMetrics.calculate_all(y_true, y_pred, self.k_values)
    
    def evaluate_diversity(self, recommendations: List[List[int]], 
                          item_features: np.ndarray, 
                          item_popularity: Dict[int, int]) -> Dict[str, float]:
        """
        Evaluate diversity and novelty.
        
        Args:
            recommendations: List of recommendation lists for each user
            item_features: Item feature matrix
            item_popularity: Dictionary mapping item IDs to popularity counts
            
        Returns:
            Dictionary with diversity metrics
        """
        return {
            'intra_list_similarity': DiversityMetrics.intra_list_similarity(
                recommendations, item_features
            ),
            'coverage': DiversityMetrics.coverage(recommendations, item_features.shape[0]),
            'novelty': DiversityMetrics.novelty(recommendations, item_popularity)
        }
    
    def evaluate_model(self, model, test_matrix: np.ndarray, 
                      test_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained recommendation model
            test_matrix: Test interaction matrix
            test_ratings: Test ratings DataFrame
            movies_df: Movie metadata DataFrame
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Rating evaluation
        if hasattr(model, 'predict'):
            # Get test user-item pairs
            test_users = test_ratings['user_id_encoded'].values
            test_items = test_ratings['item_id_encoded'].values
            test_ratings_values = test_ratings['rating'].values
            
            # Get predictions
            predictions = model.predict(test_users, test_items)
            
            # Calculate rating metrics
            results['rating_metrics'] = self.evaluate_rating_predictions(
                test_ratings_values, predictions
            )
        
        # Ranking evaluation
        if hasattr(model, 'recommend'):
            # Generate recommendations for test users
            test_user_ids = np.unique(test_users)
            recommendations = []
            true_relevant = []
            
            for user_id in test_user_ids:
                # Get recommendations
                recs = model.recommend(user_id, n_recommendations=20, exclude_rated=False)
                recommendations.append(recs)
                
                # Get true relevant items (ratings >= 4)
                user_ratings = test_ratings[test_ratings['user_id_encoded'] == user_id]
                relevant_items = user_ratings[user_ratings['rating'] >= 4]['item_id_encoded'].tolist()
                true_relevant.append(relevant_items)
            
            # Calculate ranking metrics
            results['ranking_metrics'] = self.evaluate_ranking_predictions(
                true_relevant, recommendations
            )
            
            # Calculate diversity metrics
            if 'content_features' in movies_df.columns:
                # Create item features matrix (simplified)
                item_features = np.random.rand(len(movies_df), 100)  # Placeholder
                
                # Create item popularity
                item_popularity = test_ratings['item_id_encoded'].value_counts().to_dict()
                
                results['diversity_metrics'] = self.evaluate_diversity(
                    recommendations, item_features, item_popularity
                )
        
        return results
    
    def compare_models(self, models: Dict[str, Any], test_matrix: np.ndarray,
                      test_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model name -> model instance
            test_matrix: Test interaction matrix
            test_ratings: Test ratings DataFrame
            movies_df: Movie metadata DataFrame
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            model_results = self.evaluate_model(model, test_matrix, test_ratings, movies_df)
            
            # Flatten results
            flat_results = {'model': model_name}
            
            for metric_type, metrics in model_results.items():
                for metric_name, value in metrics.items():
                    flat_results[f"{metric_type}_{metric_name}"] = value
            
            results.append(flat_results)
        
        return pd.DataFrame(results) 