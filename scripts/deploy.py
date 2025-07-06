#!/usr/bin/env python3
"""
Deployment script for the recommendation system.
Sets up database, loads data, trains models, and starts services.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'torch', 'fastapi', 
        'streamlit', 'sqlalchemy', 'psycopg2-binary'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed!")
    return True


def setup_database():
    """Set up the database."""
    logger.info("Setting up database...")
    
    try:
        # Import database modules
        from src.database.connection import init_database

        # Initialize database tables
        init_database()
        logger.info("Database setup completed!")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def download_and_prepare_data():
    """Download and prepare the dataset."""
    logger.info("Downloading and preparing data...")
    
    try:
        # Run data download script
        result = subprocess.run([
            sys.executable, "scripts/download_data.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Data preparation completed!")
            return True
        else:
            logger.error(f"Data preparation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return False


def load_data_to_database():
    """Load processed data into the database."""
    logger.info("Loading data into database...")
    
    try:
        # Import required modules
        from src.data.data_loader import load_movielens_dataset
        from src.database.connection import get_database_manager
        from src.database.repository import (
            MovieRepository,
            RatingRepository,
            UserRepository,
        )

        # Load dataset
        train_ratings, test_ratings, train_matrix, test_matrix, content_features, movies_df, metadata = load_movielens_dataset(
            data_path="data/",
            dataset_size="100k",
            min_interactions=5,
            test_size=0.2
        )
        
        # Get database session
        db_manager = get_database_manager()
        db = db_manager.get_session()
        
        # Load users
        user_repo = UserRepository(db)
        logger.info("Loading users...")
        
        # Create users from the dataset
        unique_users = train_ratings['userId'].unique()
        for user_id in unique_users:
            try:
                user_repo.create_user(user_id=int(user_id))
            except Exception:
                # User might already exist
                pass
        
        # Load movies
        movie_repo = MovieRepository(db)
        logger.info("Loading movies...")
        
        for _, movie in movies_df.iterrows():
            try:
                movie_repo.create_movie(
                    movie_id=int(movie['movieId']),
                    title=movie['title_clean'],
                    genres=movie['genres'],
                    year=movie['year'] if pd.notna(movie['year']) else None
                )
            except Exception:
                # Movie might already exist
                pass
        
        # Load ratings
        rating_repo = RatingRepository(db)
        logger.info("Loading ratings...")
        
        # Load train ratings
        for _, rating in train_ratings.iterrows():
            try:
                rating_repo.create_rating(
                    user_id=int(rating['userId']),
                    movie_id=int(rating['movieId']),
                    rating=float(rating['rating']),
                    timestamp=pd.to_datetime(rating['timestamp'], unit='s')
                )
            except Exception:
                # Rating might already exist
                pass
        
        db.close()
        logger.info("Data loading completed!")
        return True
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False


def train_models():
    """Train recommendation models."""
    logger.info("Training models...")
    
    try:
        # Run model training script
        result = subprocess.run([
            sys.executable, "scripts/train_models.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Model training completed!")
            return True
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def start_api_server():
    """Start the FastAPI server."""
    logger.info("Starting API server...")
    
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if api_process.poll() is None:
            logger.info("API server started successfully!")
            return api_process
        else:
            logger.error("API server failed to start")
            return None
            
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return None


def start_frontend():
    """Start the Streamlit frontend."""
    logger.info("Starting frontend...")
    
    try:
        # Start Streamlit in background
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "src/frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if frontend_process.poll() is None:
            logger.info("Frontend started successfully!")
            return frontend_process
        else:
            logger.error("Frontend failed to start")
            return None
            
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return None


def main():
    """Main deployment function."""
    logger.info("Starting recommendation system deployment...")
    
    # Load configuration
    config = load_config()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        logger.error("Database setup failed. Exiting.")
        sys.exit(1)
    
    # Download and prepare data
    if not download_and_prepare_data():
        logger.error("Data preparation failed. Exiting.")
        sys.exit(1)
    
    # Load data to database
    if not load_data_to_database():
        logger.error("Data loading failed. Exiting.")
        sys.exit(1)
    
    # Train models
    if not train_models():
        logger.error("Model training failed. Exiting.")
        sys.exit(1)
    
    # Start services
    logger.info("Starting services...")
    
    api_process = start_api_server()
    if not api_process:
        logger.error("Failed to start API server. Exiting.")
        sys.exit(1)
    
    frontend_process = start_frontend()
    if not frontend_process:
        logger.error("Failed to start frontend. Exiting.")
        api_process.terminate()
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info("Services running:")
    logger.info(f"  - API Server: http://{config['api']['host']}:{config['api']['port']}")
    logger.info("  - Frontend: http://localhost:8501")
    logger.info("  - API Docs: http://localhost:8000/docs")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to stop all services")
    
    try:
        # Keep services running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        
        if api_process:
            api_process.terminate()
            logger.info("API server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            logger.info("Frontend stopped")
        
        logger.info("All services stopped. Goodbye!")


if __name__ == "__main__":
    main() 