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

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Download and prepare data:**

   ```bash
   python scripts/download_data.py
   ```

4. **Train models:**

   ```bash
   python scripts/train_models.py
   ```

5. **Start the API server:**

   ```bash
   uvicorn src.api.main:app --reload
   ```

6. **Launch the frontend:**
   ```bash
   streamlit run src/frontend/app.py
   ```

## 📊 Features

- **Multiple Algorithms**: Collaborative Filtering, Content-Based, Neural Networks
- **Text & Image Search**: Query-based recommendations using embeddings
- **REST API**: FastAPI backend with comprehensive endpoints
- **Web Interface**: Streamlit frontend with interactive features
- **Database Integration**: PostgreSQL/MongoDB for data persistence
- **Monitoring**: Real-time metrics and performance tracking
- **Model Optimization**: Hyperparameter tuning with Optuna

## 🎯 API Endpoints

- `GET /recommend/{user_id}` - Get personalized recommendations
- `GET /search?q={query}` - Text-based search
- `POST /upload_image` - Image-based search
- `POST /feedback` - Submit user feedback
- `GET /metrics` - Get system performance metrics

## 📈 Evaluation Metrics

- **Rating Metrics**: RMSE, MAE
- **Ranking Metrics**: Precision@k, Recall@k, nDCG@k, MAP, MRR

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Database settings
- API configurations
- Evaluation parameters
