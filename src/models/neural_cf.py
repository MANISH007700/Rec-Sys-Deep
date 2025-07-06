"""
Neural Collaborative Filtering (NCF) model using PyTorch.
Implements a neural network-based recommendation system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering model architecture.
    
    This model combines user and item embeddings with a multi-layer neural network
    to predict user-item interactions.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_layers: List[int] = [128, 64, 32], dropout: float = 0.2):
        """
        Initialize NCF model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of user and item embeddings
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(NCFModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Neural network layers
        input_dim = embedding_dim * 2
        layers = []
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            user_ids: User ID tensor
            item_ids: Item ID tensor
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_embeds)
        
        return output.squeeze()


class NeuralCFRecommender(BaseRecommender):
    """
    Neural Collaborative Filtering recommender.
    
    This model uses a neural network to learn user-item interactions
    and predict ratings or generate recommendations.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_layers: List[int] = [128, 64, 32],
                 dropout: float = 0.2, lr: float = 0.001, batch_size: int = 256,
                 epochs: int = 50, device: str = "cpu"):
        """
        Initialize Neural CF recommender.
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(name="NeuralCF")
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'NeuralCFRecommender':
        """
        Fit the Neural CF model.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Neural CF model...")
        
        self.n_users, self.n_items = train_matrix.shape
        
        # Create model
        self.model = NCFModel(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Prepare training data
        train_data = self._prepare_training_data(train_matrix)
        
        # Train the model
        self._train_model(train_data)
        
        self.is_fitted = True
        logger.info("Neural CF model fitted successfully!")
        
        return self
    
    def _prepare_training_data(self, train_matrix: np.ndarray) -> DataLoader:
        """
        Prepare training data for the neural network.
        
        Args:
            train_matrix: User-item interaction matrix
            
        Returns:
            DataLoader for training
        """
        # Get positive interactions (ratings > 0)
        user_ids, item_ids = np.where(train_matrix > 0)
        ratings = train_matrix[user_ids, item_ids]
        
        # Create negative samples (ratings = 0)
        n_positive = len(user_ids)
        n_negative = n_positive  # Same number of negative samples
        
        # Sample random negative interactions
        negative_user_ids = np.random.randint(0, self.n_users, n_negative)
        negative_item_ids = np.random.randint(0, self.n_items, n_negative)
        
        # Ensure negative samples are actually negative
        negative_mask = train_matrix[negative_user_ids, negative_item_ids] == 0
        negative_user_ids = negative_user_ids[negative_mask]
        negative_item_ids = negative_item_ids[negative_mask]
        
        # Combine positive and negative samples
        all_user_ids = np.concatenate([user_ids, negative_user_ids[:n_positive]])
        all_item_ids = np.concatenate([item_ids, negative_item_ids[:n_positive]])
        all_ratings = np.concatenate([ratings, np.zeros(n_positive)])
        
        # Convert to tensors
        user_tensor = torch.LongTensor(all_user_ids)
        item_tensor = torch.LongTensor(all_item_ids)
        rating_tensor = torch.FloatTensor(all_ratings)
        
        # Create dataset and dataloader
        dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def _train_model(self, train_data: DataLoader) -> None:
        """
        Train the neural network model.
        
        Args:
            train_data: Training data loader
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_user_ids, batch_item_ids, batch_ratings in train_data:
                # Move to device
                batch_user_ids = batch_user_ids.to(self.device)
                batch_item_ids = batch_item_ids.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_user_ids, batch_item_ids)
                loss = self.criterion(predictions, batch_ratings)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
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
        
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for user_id, item_id in zip(user_ids, item_ids):
                if user_id >= self.n_users or item_id >= self.n_items:
                    predictions.append(3.0)  # Default rating
                else:
                    # Convert to tensors
                    user_tensor = torch.LongTensor([user_id]).to(self.device)
                    item_tensor = torch.LongTensor([item_id]).to(self.device)
                    
                    # Get prediction
                    pred = self.model(user_tensor, item_tensor).cpu().numpy()
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
        
        self.model.eval()
        
        # Calculate scores for all items
        user_tensor = torch.LongTensor([user_id] * self.n_items).to(self.device)
        item_tensor = torch.LongTensor(range(self.n_items)).to(self.device)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        return top_items.tolist()
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get user and item embeddings from the trained model.
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting embeddings")
        
        self.model.eval()
        
        with torch.no_grad():
            user_embeddings = self.model.user_embedding.weight.cpu().numpy()
            item_embeddings = self.model.item_embedding.weight.cpu().numpy()
        
        return user_embeddings, item_embeddings


class AutoencoderRecommender(BaseRecommender):
    """
    Autoencoder-based recommendation system.
    
    This model uses an autoencoder to learn a compressed representation
    of the user-item interaction matrix and reconstruct it for recommendations.
    """
    
    def __init__(self, hidden_dim: int = 64, lr: float = 0.001, 
                 batch_size: int = 256, epochs: int = 50, device: str = "cpu"):
        """
        Initialize Autoencoder recommender.
        
        Args:
            hidden_dim: Dimension of hidden layer
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use
        """
        super().__init__(name="Autoencoder")
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.criterion = None
        
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'AutoencoderRecommender':
        """
        Fit the Autoencoder model.
        
        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Autoencoder model...")
        
        self.n_users, self.n_items = train_matrix.shape
        
        # Create encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_items),
            nn.Sigmoid()  # Output between 0 and 1
        ).to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )
        self.criterion = nn.MSELoss()
        
        # Prepare training data
        train_data = self._prepare_training_data(train_matrix)
        
        # Train the model
        self._train_model(train_data)
        
        self.is_fitted = True
        logger.info("Autoencoder model fitted successfully!")
        
        return self
    
    def _prepare_training_data(self, train_matrix: np.ndarray) -> DataLoader:
        """Prepare training data for the autoencoder."""
        # Normalize ratings to [0, 1]
        normalized_matrix = train_matrix / 5.0
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(normalized_matrix)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def _train_model(self, train_data: DataLoader) -> None:
        """Train the autoencoder model."""
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for (batch_data,) in train_data:
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                encoded = self.encoder(batch_data)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, batch_data)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.encoder.eval()
        self.decoder.eval()
        
        predictions = []
        
        with torch.no_grad():
            for user_id in user_ids:
                if user_id >= self.n_users:
                    predictions.extend([3.0] * len(item_ids))
                else:
                    # Get user's full rating vector
                    user_ratings = torch.zeros(self.n_items).to(self.device)
                    
                    # Encode and decode
                    encoded = self.encoder(user_ratings.unsqueeze(0))
                    decoded = self.decoder(encoded).squeeze()
                    
                    # Get predictions for requested items
                    for item_id in item_ids:
                        if item_id >= self.n_items:
                            pred = 3.0
                        else:
                            pred = decoded[item_id].cpu().numpy() * 5.0  # Scale back to [0, 5]
                        predictions.append(max(1, min(5, pred)))
        
        return np.array(predictions)
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True) -> List[int]:
        """Generate recommendations for a specific user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id >= self.n_users:
            return list(range(min(n_recommendations, self.n_items)))
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Create user vector (all zeros for new user)
            user_vector = torch.zeros(self.n_items).to(self.device)
            
            # Encode and decode
            encoded = self.encoder(user_vector.unsqueeze(0))
            decoded = self.decoder(encoded).squeeze()
            
            # Get scores
            scores = decoded.cpu().numpy() * 5.0  # Scale back to [0, 5]
            
            # Get top recommendations
            top_items = np.argsort(scores)[::-1][:n_recommendations]
            return top_items.tolist() 