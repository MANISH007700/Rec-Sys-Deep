# 🎯 End-to-End Recommendation System

A comprehensive, production-grade recommendation system that implements multiple algorithms, provides REST APIs, and includes a web interface.

## 🏗️ Project Structure

```
rec-sys/
├── data/                   # Dataset storage
├── models/                 # Trained models
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Recommendation algorithms
│   ├── evaluation/        # Metrics and evaluation
│   ├── api/               # FastAPI backend
│   ├── database/          # Database models and operations
│   └── frontend/          # Streamlit frontend
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── config/                # Configuration files
└── scripts/               # Utility scripts
```

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd rec-sys
   ```

2. **Start with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

3. **Access the services:**
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL database:**

   ```bash
   # Install PostgreSQL or use Docker
   docker run -d --name postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=recsys -p 5432:5432 postgres:15
   ```

3. **Set up environment variables:**

   ```bash
   cp env.example .env
   # Edit .env with your database credentials
   ```

4. **Run the complete deployment:**
   ```bash
   python scripts/deploy.py
   ```

### Option 3: Step-by-Step Setup

1. **Download and prepare data:**

   ```bash
   python scripts/download_data.py
   ```

2. **Train models:**

   ```bash
   python scripts/train_models.py
   ```

3. **Start the API server:**

   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Launch the frontend:**
   ```bash
   streamlit run src/frontend/app.py
   ```

## 📊 Features

- **Multiple Algorithms**:
  - Collaborative Filtering (SVD, ALS, User-based CF)
  - Content-based (TF-IDF, Genre-based)
  - Neural Networks (NCF, Autoencoder)
  - Hybrid Models (Ensemble, Cascade, Feature Fusion)
- **REST API**: FastAPI backend with comprehensive endpoints
- **Web Interface**: Streamlit frontend with interactive features
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Monitoring**: Real-time metrics and performance tracking
- **Docker Support**: Complete containerization
- **Unit Tests**: Comprehensive test coverage

## 🎯 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /models` - Available models

### Recommendation Endpoints

- `POST /recommend` - Get personalized recommendations
- `GET /recommend/{user_id}` - Simple recommendation endpoint

### Data Endpoints

- `POST /ratings` - Create/update rating
- `GET /ratings/{user_id}` - Get user ratings
- `GET /movies` - Get movies with filtering
- `GET /movies/popular` - Get popular movies
- `GET /movies/{movie_id}` - Get specific movie
- `GET /search` - Search movies by title

### Model Endpoints

- `GET /models/{model_name}/performance` - Get model performance

## 📈 Evaluation Metrics

### Rating Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Ranking Metrics

- **Precision@k**: Precision at k
- **Recall@k**: Recall at k
- **F1@k**: F1-score at k
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision
- **MRR@k**: Mean Reciprocal Rank

### Diversity Metrics

- **Intra-list Similarity**: Diversity within recommendations
- **Coverage**: Percentage of items recommended
- **Novelty**: Average item popularity

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Database settings
- API configurations
- Evaluation parameters

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_data_loader.py
```

## 📊 Monitoring

Monitor system performance:

```bash
# One-time check
python scripts/monitor.py

# Continuous monitoring
python scripts/monitor.py --mode continuous --interval 60

# Generate report
python scripts/monitor.py --report 24 --save
```

## 🐳 Docker Commands

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build recsys
```

## 📝 Usage Examples

### Using the API

```python
import requests

# Get recommendations
response = requests.post("http://localhost:8000/recommend", json={
    "user_id": 1,
    "n_recommendations": 10,
    "model_name": "SVD"
})

# Search movies
response = requests.get("http://localhost:8000/search?q=toy story")

# Rate a movie
response = requests.post("http://localhost:8000/ratings", json={
    "user_id": 1,
    "movie_id": 1,
    "rating": 4.5
})
```

### Using the Frontend

1. Open http://localhost:8501
2. Navigate to "Get Recommendations"
3. Enter a user ID and select a model
4. View personalized recommendations
5. Rate movies and see how recommendations change

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Error:**

   - Ensure PostgreSQL is running
   - Check database credentials in `.env`
   - Verify database exists

2. **Model Loading Error:**

   - Run `python scripts/train_models.py` first
   - Check that model files exist in `models/` directory

3. **API Connection Error:**

   - Ensure API server is running on port 8000
   - Check firewall settings
   - Verify CORS configuration

4. **Frontend Connection Error:**
   - Ensure Streamlit is running on port 8501
   - Check API URL configuration

### Logs

Check logs in the `logs/` directory:

```bash
tail -f logs/recsys.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
