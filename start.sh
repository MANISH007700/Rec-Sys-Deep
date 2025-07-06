#!/bin/bash

# Recommendation System Startup Script

echo "🎬 Starting Recommendation System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if Docker is installed (for database)
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Will use local setup."
    USE_DOCKER=false
else
    USE_DOCKER=true
fi

# Create necessary directories
mkdir -p data models logs

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set up environment
if [ ! -f .env ]; then
    echo "🔧 Setting up environment..."
    cp env.example .env
fi

# Start database if using Docker
if [ "$USE_DOCKER" = true ]; then
    echo "🐳 Starting PostgreSQL with Docker..."
    docker run -d --name postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=recsys -p 5432:5432 postgres:15
    
    # Wait for database to be ready
    echo "⏳ Waiting for database to be ready..."
    sleep 10
fi

# Run deployment
echo "🚀 Running deployment..."
python scripts/deploy.py

echo "✅ Recommendation System is ready!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs" 