"""
Streamlit frontend for the recommendation system.
Provides an interactive web interface for users to explore recommendations.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = "config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# API configuration
API_BASE_URL = f"http://{config['api']['host']}:{config['api']['port']}"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .movie-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "🎯 Get Recommendations", "🔍 Search Movies", 
             "⭐ Rate Movies", "📊 System Stats", "🤖 Model Performance"]
        )
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Get Recommendations":
        show_recommendations()
    elif page == "🔍 Search Movies":
        show_search()
    elif page == "⭐ Rate Movies":
        show_ratings()
    elif page == "📊 System Stats":
        show_system_stats()
    elif page == "🤖 Model Performance":
        show_model_performance()


def show_dashboard():
    """Show the main dashboard."""
    st.header("📊 Dashboard")
    
    # Get system stats
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", stats.get('total_users', 0))
            
            with col2:
                st.metric("Total Movies", stats.get('total_movies', 0))
            
            with col3:
                st.metric("Total Ratings", stats.get('total_ratings', 0))
            
            with col4:
                st.metric("Models Loaded", stats.get('models_loaded', 0))
            
            # Rating distribution
            if 'avg_rating' in stats:
                st.subheader("📈 Rating Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Rating", f"{stats['avg_rating']:.2f}")
                
                with col2:
                    st.metric("Rating Std Dev", f"{stats['rating_std']:.2f}")
                
                with col3:
                    st.metric("Sparsity", f"{stats['sparsity']:.2%}")
            
            # Popular movies
            st.subheader("🔥 Popular Movies")
            try:
                popular_response = requests.get(f"{API_BASE_URL}/movies/popular?limit=10")
                if popular_response.status_code == 200:
                    popular_movies = popular_response.json()['movies']
                    
                    # Create DataFrame for display
                    df = pd.DataFrame(popular_movies)
                    df['rating_count'] = df['rating_count'].astype(int)
                    
                    # Display as a table
                    st.dataframe(
                        df[['title', 'genres', 'year', 'rating_count']].head(10),
                        use_container_width=True
                    )
                    
                    # Create bar chart
                    fig = px.bar(
                        df.head(10),
                        x='title',
                        y='rating_count',
                        title="Top 10 Movies by Rating Count",
                        labels={'title': 'Movie Title', 'rating_count': 'Number of Ratings'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading popular movies: {e}")
        
        else:
            st.error("Failed to load system statistics")
            
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.info("Make sure the API server is running!")


def show_recommendations():
    """Show recommendation interface."""
    st.header("🎯 Get Personalized Recommendations")
    
    # User input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    
    with col2:
        n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
    
    with col3:
        # Get available models
        try:
            models_response = requests.get(f"{API_BASE_URL}/models")
            if models_response.status_code == 200:
                models = models_response.json()['models']
                model_name = st.selectbox("Model", models, index=0 if models else None)
            else:
                model_name = "SVD"
        except:
            model_name = "SVD"
    
    # Get recommendations button
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/recommend",
                    json={
                        "user_id": user_id,
                        "n_recommendations": n_recommendations,
                        "model_name": model_name
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Generated {len(result['recommendations'])} recommendations using {model_name}")
                    
                    # Display recommendations
                    st.subheader("🎬 Recommended Movies")
                    
                    for i, rec in enumerate(result['recommendations']):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{i+1}. {rec['title']}** ({rec['year']})")
                                st.markdown(f"*{rec['genres']}*")
                            
                            with col2:
                                if st.button(f"Rate", key=f"rate_{i}"):
                                    st.session_state.rate_movie = {
                                        'user_id': user_id,
                                        'movie_id': rec['movie_id'],
                                        'title': rec['title']
                                    }
                    
                    # Show rating interface if a movie is selected
                    if 'rate_movie' in st.session_state:
                        st.subheader("⭐ Rate Movie")
                        movie_info = st.session_state.rate_movie
                        
                        st.write(f"Rating: **{movie_info['title']}**")
                        rating = st.slider("Your Rating", 1.0, 5.0, 3.0, 0.5, key="rating_slider")
                        
                        if st.button("Submit Rating"):
                            try:
                                rating_response = requests.post(
                                    f"{API_BASE_URL}/ratings",
                                    json={
                                        "user_id": movie_info['user_id'],
                                        "movie_id": movie_info['movie_id'],
                                        "rating": rating
                                    }
                                )
                                
                                if rating_response.status_code == 200:
                                    st.success("Rating submitted successfully!")
                                    del st.session_state.rate_movie
                                else:
                                    st.error("Failed to submit rating")
                                    
                            except Exception as e:
                                st.error(f"Error submitting rating: {e}")
                
                else:
                    st.error("Failed to get recommendations")
                    
            except Exception as e:
                st.error(f"Error: {e}")


def show_search():
    """Show movie search interface."""
    st.header("🔍 Search Movies")
    
    # Search input
    search_query = st.text_input("Search for movies by title:")
    
    if search_query:
        try:
            response = requests.get(f"{API_BASE_URL}/search?q={search_query}&limit=20")
            
            if response.status_code == 200:
                result = response.json()
                
                st.success(f"Found {result['count']} movies")
                
                # Display search results
                for movie in result['movies']:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{movie['title']}** ({movie['year']})")
                            st.markdown(f"*{movie['genres']}*")
                        
                        with col2:
                            if st.button(f"Rate", key=f"search_rate_{movie['movie_id']}"):
                                st.session_state.rate_movie = {
                                    'user_id': 1,  # Default user
                                    'movie_id': movie['movie_id'],
                                    'title': movie['title']
                                }
                        
                        with col3:
                            if movie.get('imdb_url'):
                                st.markdown(f"[IMDb]({movie['imdb_url']})")
                
                # Rating interface
                if 'rate_movie' in st.session_state:
                    st.subheader("⭐ Rate Movie")
                    movie_info = st.session_state.rate_movie
                    
                    st.write(f"Rating: **{movie_info['title']}**")
                    rating = st.slider("Your Rating", 1.0, 5.0, 3.0, 0.5, key="search_rating_slider")
                    
                    if st.button("Submit Rating"):
                        try:
                            rating_response = requests.post(
                                f"{API_BASE_URL}/ratings",
                                json={
                                    "user_id": movie_info['user_id'],
                                    "movie_id": movie_info['movie_id'],
                                    "rating": rating
                                }
                            )
                            
                            if rating_response.status_code == 200:
                                st.success("Rating submitted successfully!")
                                del st.session_state.rate_movie
                            else:
                                st.error("Failed to submit rating")
                                
                        except Exception as e:
                            st.error(f"Error submitting rating: {e}")
            
            else:
                st.error("Search failed")
                
        except Exception as e:
            st.error(f"Error: {e}")


def show_ratings():
    """Show user ratings interface."""
    st.header("⭐ Rate Movies")
    
    # User selection
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    
    # Get user's existing ratings
    try:
        response = requests.get(f"{API_BASE_URL}/ratings/{user_id}")
        
        if response.status_code == 200:
            result = response.json()
            
            st.subheader(f"Your Ratings ({result['count']} movies)")
            
            if result['ratings']:
                # Create DataFrame for display
                df = pd.DataFrame(result['ratings'])
                
                # Display ratings
                for _, row in df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['title']}**")
                        
                        with col2:
                            st.write(f"Rating: {row['rating']}")
                        
                        with col3:
                            if st.button(f"Update", key=f"update_{row['movie_id']}"):
                                st.session_state.update_rating = {
                                    'user_id': user_id,
                                    'movie_id': row['movie_id'],
                                    'title': row['title'],
                                    'current_rating': row['rating']
                                }
                
                # Update rating interface
                if 'update_rating' in st.session_state:
                    st.subheader("📝 Update Rating")
                    rating_info = st.session_state.update_rating
                    
                    st.write(f"Movie: **{rating_info['title']}**")
                    st.write(f"Current Rating: {rating_info['current_rating']}")
                    
                    new_rating = st.slider("New Rating", 1.0, 5.0, rating_info['current_rating'], 0.5)
                    
                    if st.button("Update Rating"):
                        try:
                            rating_response = requests.post(
                                f"{API_BASE_URL}/ratings",
                                json={
                                    "user_id": rating_info['user_id'],
                                    "movie_id": rating_info['movie_id'],
                                    "rating": new_rating
                                }
                            )
                            
                            if rating_response.status_code == 200:
                                st.success("Rating updated successfully!")
                                del st.session_state.update_rating
                                st.rerun()
                            else:
                                st.error("Failed to update rating")
                                
                        except Exception as e:
                            st.error(f"Error updating rating: {e}")
            
            else:
                st.info("No ratings found for this user")
        
        else:
            st.error("Failed to load user ratings")
            
    except Exception as e:
        st.error(f"Error: {e}")


def show_system_stats():
    """Show detailed system statistics."""
    st.header("📊 System Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Basic stats
            st.subheader("📈 Basic Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Users", stats.get('total_users', 0))
                st.metric("Total Movies", stats.get('total_movies', 0))
            
            with col2:
                st.metric("Total Ratings", stats.get('total_ratings', 0))
                st.metric("Models Loaded", stats.get('models_loaded', 0))
            
            # Rating statistics
            if 'avg_rating' in stats:
                st.subheader("📊 Rating Statistics")
                
                # Create rating distribution chart
                fig = go.Figure()
                
                # Simulate rating distribution (in a real system, you'd get this from the database)
                ratings = [1, 2, 3, 4, 5]
                counts = [100, 200, 300, 400, 500]  # Example data
                
                fig.add_trace(go.Bar(
                    x=ratings,
                    y=counts,
                    name="Rating Distribution",
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Rating Distribution",
                    xaxis_title="Rating",
                    yaxis_title="Count",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation statistics
            if 'total_recommendations' in stats:
                st.subheader("🎯 Recommendation Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Recommendations", stats.get('total_recommendations', 0))
                
                with col2:
                    st.metric("Unique Users", stats.get('unique_users', 0))
                
                with col3:
                    st.metric("Unique Movies", stats.get('unique_movies', 0))
                
                # Model distribution
                if 'model_distribution' in stats:
                    st.subheader("🤖 Model Usage Distribution")
                    
                    model_data = stats['model_distribution']
                    if model_data:
                        fig = px.pie(
                            values=list(model_data.values()),
                            names=list(model_data.keys()),
                            title="Recommendations by Model"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Failed to load system statistics")
            
    except Exception as e:
        st.error(f"Error: {e}")


def show_model_performance():
    """Show model performance metrics."""
    st.header("🤖 Model Performance")
    
    # Get available models
    try:
        models_response = requests.get(f"{API_BASE_URL}/models")
        if models_response.status_code == 200:
            models = models_response.json()['models']
            
            if models:
                selected_model = st.selectbox("Select Model", models)
                
                # Get performance for selected model
                perf_response = requests.get(f"{API_BASE_URL}/models/{selected_model}/performance")
                
                if perf_response.status_code == 200:
                    performance = perf_response.json()
                    
                    st.subheader(f"Performance Metrics for {selected_model}")
                    
                    if performance['metrics']:
                        # Display metrics
                        for metric_name, metric_data in performance['metrics'].items():
                            st.write(f"**{metric_name}:**")
                            
                            # Create line chart for metric over time
                            if len(metric_data) > 1:
                                df = pd.DataFrame(metric_data)
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                
                                fig = px.line(
                                    df,
                                    x='timestamp',
                                    y='value',
                                    color='dataset_split',
                                    title=f"{metric_name} Over Time"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Display latest values
                            for data in metric_data:
                                st.write(f"  - {data['dataset_split']}: {data['value']:.4f} "
                                       f"({data['timestamp']})")
                    else:
                        st.info("No performance data available for this model")
                else:
                    st.error("Failed to load model performance")
            else:
                st.info("No models available")
        else:
            st.error("Failed to load models")
            
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main() 