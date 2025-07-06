"""
Unit tests for the data loader module.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.data_loader import MovieLensDataLoader, load_movielens_dataset


class TestMovieLensDataLoader(unittest.TestCase):
    """Test cases for MovieLensDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = MovieLensDataLoader(data_path=self.temp_dir, min_interactions=3)
        
        # Create sample MovieLens 100k data
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_sample_data(self):
        """Create sample MovieLens 100k data files."""
        dataset_dir = os.path.join(self.temp_dir, "ml-100k")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create u.data (ratings)
        ratings_data = [
            "1\t1\t5\t874965758",
            "1\t2\t3\t876893171",
            "1\t3\t4\t878542960",
            "2\t1\t3\t888551432",
            "2\t2\t4\t888551432",
            "2\t3\t5\t888551432",
            "3\t1\t4\t888551432",
            "3\t2\t2\t888551432",
            "3\t3\t5\t888551432"
        ]
        
        with open(os.path.join(dataset_dir, "u.data"), "w") as f:
            f.write("\n".join(ratings_data))
        
        # Create u.item (movies)
        movies_data = [
            "1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0",
            "2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0",
            "3|Four Rooms (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0"
        ]
        
        with open(os.path.join(dataset_dir, "u.item"), "w", encoding='latin-1') as f:
            f.write("\n".join(movies_data))
        
        # Create u.user (users)
        users_data = [
            "1|24|M|technician|85711",
            "2|53|F|other|94043",
            "3|23|M|writer|32067"
        ]
        
        with open(os.path.join(dataset_dir, "u.user"), "w") as f:
            f.write("\n".join(users_data))
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.loader.data_path, self.temp_dir)
        self.assertEqual(self.loader.min_interactions, 3)
        self.assertIsNone(self.loader.ratings_df)
        self.assertIsNone(self.loader.movies_df)
        self.assertIsNone(self.loader.users_df)
    
    def test_load_data_movielens_100k(self):
        """Test loading MovieLens 100k data."""
        self.loader.load_data("100k")
        
        # Check ratings
        self.assertIsNotNone(self.loader.ratings_df)
        self.assertEqual(len(self.loader.ratings_df), 9)
        self.assertListEqual(list(self.loader.ratings_df.columns), 
                           ['userId', 'movieId', 'rating', 'timestamp'])
        
        # Check movies
        self.assertIsNotNone(self.loader.movies_df)
        self.assertEqual(len(self.loader.movies_df), 3)
        self.assertIn('movieId', self.loader.movies_df.columns)
        self.assertIn('title', self.loader.movies_df.columns)
        self.assertIn('genres', self.loader.movies_df.columns)
        
        # Check users
        self.assertIsNotNone(self.loader.users_df)
        self.assertEqual(len(self.loader.users_df), 3)
        self.assertIn('userId', self.loader.users_df.columns)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.loader.load_data("100k")
        self.loader.preprocess_data()
        
        # Check that preprocessing was applied
        self.assertIn('user_id_encoded', self.loader.ratings_df.columns)
        self.assertIn('item_id_encoded', self.loader.ratings_df.columns)
        self.assertIn('title_clean', self.loader.movies_df.columns)
        self.assertIn('year', self.loader.movies_df.columns)
        self.assertIn('genre_list', self.loader.movies_df.columns)
    
    def test_create_interaction_matrix(self):
        """Test interaction matrix creation."""
        self.loader.load_data("100k")
        self.loader.preprocess_data()
        
        matrix, metadata = self.loader.create_interaction_matrix()
        
        # Check matrix shape
        self.assertEqual(matrix.shape, (3, 3))  # 3 users, 3 movies
        
        # Check metadata
        self.assertIn('n_users', metadata)
        self.assertIn('n_items', metadata)
        self.assertIn('n_ratings', metadata)
        self.assertEqual(metadata['n_users'], 3)
        self.assertEqual(metadata['n_items'], 3)
        self.assertEqual(metadata['n_ratings'], 9)
    
    def test_get_train_test_split(self):
        """Test train/test split."""
        self.loader.load_data("100k")
        self.loader.preprocess_data()
        
        train_ratings, test_ratings, train_matrix, test_matrix = self.loader.get_train_test_split(
            test_size=0.3, random_state=42
        )
        
        # Check that split was created
        self.assertIsNotNone(train_ratings)
        self.assertIsNotNone(test_ratings)
        self.assertIsNotNone(train_matrix)
        self.assertIsNotNone(test_matrix)
        
        # Check that train and test are disjoint
        train_set = set(zip(train_ratings['user_id_encoded'], train_ratings['item_id_encoded']))
        test_set = set(zip(test_ratings['user_id_encoded'], test_ratings['item_id_encoded']))
        self.assertEqual(len(train_set.intersection(test_set)), 0)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        self.loader.load_data("100k")
        self.loader.preprocess_data()
        
        summary = self.loader.get_data_summary()
        
        # Check summary keys
        expected_keys = ['n_users', 'n_items', 'n_ratings', 'avg_rating', 'sparsity']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check summary values
        self.assertEqual(summary['n_users'], 3)
        self.assertEqual(summary['n_items'], 3)
        self.assertEqual(summary['n_ratings'], 9)


class TestLoadMovieLensDataset(unittest.TestCase):
    """Test cases for load_movielens_dataset function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.data.data_loader.MovieLensDataLoader')
    def test_load_movielens_dataset(self, mock_loader_class):
        """Test the convenience function."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock the return values
        mock_loader.get_train_test_split.return_value = (
            pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([])
        )
        mock_loader.create_interaction_matrix.return_value = (
            np.array([]), {'n_users': 3, 'n_items': 3, 'n_ratings': 9}
        )
        mock_loader.movies_df = pd.DataFrame()
        mock_loader.get_data_summary.return_value = {
            'n_users': 3, 'n_items': 3, 'n_ratings': 9
        }
        
        # Call the function
        result = load_movielens_dataset(
            data_path=self.temp_dir,
            dataset_size="100k",
            min_interactions=3,
            test_size=0.2
        )
        
        # Check that loader was called correctly
        mock_loader_class.assert_called_once_with(
            data_path=self.temp_dir, min_interactions=3
        )
        mock_loader.download_movielens.assert_called_once_with("100k")
        mock_loader.load_data.assert_called_once_with("100k")
        mock_loader.preprocess_data.assert_called_once()
        mock_loader.get_train_test_split.assert_called_once_with(
            test_size=0.2, random_state=42
        )
        
        # Check return value
        self.assertEqual(len(result), 7)  # Should return 7 values


if __name__ == '__main__':
    unittest.main() 