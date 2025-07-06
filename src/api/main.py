"""
FastAPI main application for the recommendation system.
Provides REST API endpoints for recommendations, search, and monitoring.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database.connection import get_db_session, init_database
from ..database.repository import (
    ModelPerformanceRepository,
    MovieRepository,
    RatingRepository,
    RecommendationRepository,
    UserRepository,
)
from ..models.base_model import BaseRecommender
from ..models.collaborative_filtering import ALSRecommender, SVDRecommender
from ..models.content_based import ContentBasedRecommender
from ..models.hybrid import create_hybrid_ensemble

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = "config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="A comprehensive recommendation system with multiple algorithms",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache: Dict[str, BaseRecommender] = {}


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10
    model_name: Optional[str] = None


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    feedback_type: str  # like, dislike, click, view
    feedback_value: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict]
    model_name: str
    timestamp: datetime


class MovieInfo(BaseModel):
    movie_id: int
    title: str
    genres: Optional[str]
    year: Optional[int]
    score: float


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup."""
    logger.info("Starting recommendation system API...")
    init_database()
    await load_models()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down recommendation system API...")


async def load_models():
    """Load trained models into memory."""
    logger.info("Loading recommendation models...")
    
    try:
        # Load individual models
        model_files = {
            'SVD': 'models/svd_model.pkl',
            'ALS': 'models/als_model.pkl',
            'ContentBased': 'models/contentbased_model.pkl',
            'Hybrid_weighted': 'models/hybrid_weighted_model.pkl'
        }
        
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                model = BaseRecommender.load_model(model_path)
                model_cache[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        logger.info(f"Loaded {len(model_cache)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")


def get_model(model_name: str) -> BaseRecommender:
    """Get model from cache."""
    if model_name not in model_cache:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return model_cache[model_name]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": len(model_cache)
    }


# Recommendation endpoints
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db_session)
):
    """Get personalized recommendations for a user."""
    try:
        user_id = request.user_id
        n_recommendations = request.n_recommendations
        model_name = request.model_name or "SVD"  # Default to SVD
        
        # Get model
        model = get_model(model_name)
        
        # Generate recommendations
        recommendations = model.recommend(user_id, n_recommendations)
        
        # Get movie information
        movie_repo = MovieRepository(db)
        recommendation_data = []
        
        for i, movie_id in enumerate(recommendations):
            movie = movie_repo.get_movie_by_id(movie_id)
            if movie:
                recommendation_data.append({
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "genres": movie.genres,
                    "year": movie.year,
                    "rank": i + 1
                })
        
        # Store recommendations in database
        rec_repo = RecommendationRepository(db)
        for i, movie_id in enumerate(recommendations):
            rec_repo.create_recommendation(
                user_id=user_id,
                movie_id=movie_id,
                score=1.0 - (i * 0.1),  # Simple scoring
                model_name=model_name,
                rank=i + 1
            )
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendation_data,
            model_name=model_name,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{user_id}")
async def get_recommendations_simple(
    user_id: int,
    n_recommendations: int = Query(10, ge=1, le=50),
    model_name: str = Query("SVD"),
    db: Session = Depends(get_db_session)
):
    """Simple recommendation endpoint."""
    request = RecommendationRequest(
        user_id=user_id,
        n_recommendations=n_recommendations,
        model_name=model_name
    )
    return await get_recommendations(request, db)


# Rating endpoints
@app.post("/ratings")
async def create_rating(
    request: RatingRequest,
    db: Session = Depends(get_db_session)
):
    """Create or update a user rating."""
    try:
        rating_repo = RatingRepository(db)
        
        # Check if rating already exists
        existing_rating = rating_repo.get_rating(request.user_id, request.movie_id)
        
        if existing_rating:
            # Update existing rating
            rating = rating_repo.update_rating(
                request.user_id, request.movie_id, request.rating
            )
            message = "Rating updated successfully"
        else:
            # Create new rating
            rating = rating_repo.create_rating(
                request.user_id, request.movie_id, request.rating
            )
            message = "Rating created successfully"
        
        return {
            "message": message,
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "rating": request.rating,
            "timestamp": rating.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error creating rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ratings/{user_id}")
async def get_user_ratings(
    user_id: int,
    db: Session = Depends(get_db_session)
):
    """Get all ratings for a user."""
    try:
        rating_repo = RatingRepository(db)
        ratings = rating_repo.get_user_ratings(user_id)
        
        rating_data = []
        for rating in ratings:
            rating_data.append({
                "movie_id": rating.movie.movie_id,
                "title": rating.movie.title,
                "rating": rating.rating,
                "timestamp": rating.timestamp
            })
        
        return {
            "user_id": user_id,
            "ratings": rating_data,
            "count": len(rating_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting user ratings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Movie endpoints
@app.get("/movies")
async def get_movies(
    limit: int = Query(10, ge=1, le=100),
    genre: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db_session)
):
    """Get movies with optional filtering."""
    try:
        movie_repo = MovieRepository(db)
        
        if search:
            movies = movie_repo.search_movies(search, limit)
        elif genre:
            movies = movie_repo.get_movies_by_genre(genre, limit)
        else:
            movies = movie_repo.get_all_movies(limit)
        
        movie_data = []
        for movie in movies:
            movie_data.append({
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres,
                "year": movie.year,
                "imdb_url": movie.imdb_url
            })
        
        return {
            "movies": movie_data,
            "count": len(movie_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/popular")
async def get_popular_movies(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db_session)
):
    """Get most popular movies."""
    try:
        movie_repo = MovieRepository(db)
        popular_movies = movie_repo.get_popular_movies(limit)
        
        movie_data = []
        for movie, rating_count in popular_movies:
            movie_data.append({
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres,
                "year": movie.year,
                "rating_count": rating_count
            })
        
        return {
            "movies": movie_data,
            "count": len(movie_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}")
async def get_movie(
    movie_id: int,
    db: Session = Depends(get_db_session)
):
    """Get specific movie information."""
    try:
        movie_repo = MovieRepository(db)
        movie = movie_repo.get_movie_by_id(movie_id)
        
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "genres": movie.genres,
            "year": movie.year,
            "imdb_url": movie.imdb_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie {movie_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoint
@app.get("/search")
async def search_movies(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db_session)
):
    """Search movies by title."""
    try:
        movie_repo = MovieRepository(db)
        movies = movie_repo.search_movies(q, limit)
        
        movie_data = []
        for movie in movies:
            movie_data.append({
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres,
                "year": movie.year
            })
        
        return {
            "query": q,
            "movies": movie_data,
            "count": len(movie_data)
        }
        
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feedback endpoint
@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db_session)
):
    """Submit user feedback on recommendations."""
    try:
        # For now, we'll just log the feedback
        # In a full implementation, you'd store this in the UserFeedback table
        logger.info(f"User feedback: {request.user_id} -> {request.movie_id} "
                   f"({request.feedback_type}: {request.feedback_value})")
        
        return {
            "message": "Feedback received successfully",
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "feedback_type": request.feedback_type,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model endpoints
@app.get("/models")
async def get_models():
    """Get available models."""
    return {
        "models": list(model_cache.keys()),
        "count": len(model_cache)
    }


@app.get("/models/{model_name}/performance")
async def get_model_performance(
    model_name: str,
    db: Session = Depends(get_db_session)
):
    """Get performance metrics for a model."""
    try:
        perf_repo = ModelPerformanceRepository(db)
        performance = perf_repo.get_model_performance(model_name)
        
        metrics = {}
        for perf in performance:
            if perf.metric_name not in metrics:
                metrics[perf.metric_name] = []
            metrics[perf.metric_name].append({
                "value": perf.metric_value,
                "dataset_split": perf.dataset_split,
                "timestamp": perf.timestamp
            })
        
        return {
            "model_name": model_name,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoints
@app.get("/stats")
async def get_system_stats(db: Session = Depends(get_db_session)):
    """Get system statistics."""
    try:
        # Get basic counts
        user_repo = UserRepository(db)
        movie_repo = MovieRepository(db)
        rating_repo = RatingRepository(db)
        rec_repo = RecommendationRepository(db)
        
        users = user_repo.get_all_users()
        movies = movie_repo.get_all_movies()
        ratings_df = rating_repo.get_ratings_dataframe()
        
        stats = {
            "total_users": len(users),
            "total_movies": len(movies),
            "total_ratings": len(ratings_df) if not ratings_df.empty else 0,
            "models_loaded": len(model_cache),
            "timestamp": datetime.utcnow()
        }
        
        # Add rating statistics if available
        if not ratings_df.empty:
            stats.update({
                "avg_rating": float(ratings_df['rating'].mean()),
                "rating_std": float(ratings_df['rating'].std()),
                "sparsity": 1 - (len(ratings_df) / (len(users) * len(movies)))
            })
        
        # Add recommendation statistics
        rec_stats = rec_repo.get_recommendation_stats()
        stats.update(rec_stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug']
    ) 