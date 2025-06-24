# DeepWalk Movie Recommendation Engine

A FastAPI-powered movie recommendation system that leverages DeepWalk graph embeddings on the [MovieLens 100k dataset](https://files.grouplens.org/datasets/movielens/ml-100k.zip).

---

## Features

-	Graph-based Recommendations using DeepWalk for learning embeddings from user-movie interactions.
-	FastAPI REST Endpoints for interactions, recommendations, and listing movies.
-	Hyperparameter Tuning for optimizing DeepWalk training.
-	Structured Logging for traceability and debugging.
-	Comprehensive Testing using Pytest.

---

## System Architecture

![Architecture Diagram](DeepWalk_Movie_Recommendation_Architecture.png)

### Layer Overview

#### Data Layer

Handles user profiles, movie metadata, and interaction logs. These inputs form the basis for graph construction and embedding training.

#### Logic Layer

Processes user input and applies similarity calculations to produce ranked movie recommendations.

#### Model Layer

Encapsulates the DeepWalk model training, tuning, and embedding generation using the interaction graph.

#### API Layer

Exposes endpoints for external interaction. Bridges frontend consumers with backend logic and models.

#### Logging Layer

Captures system events and errors to aid monitoring and debugging.

---

## Project Structure

```
deepwalk_recommender/
├── ALGORITHM_WRITEUP.md
├── data/
│   ├── ml-100k/
│   ├── ml-100k.zip
│   └── processed_ratings.csv
├── Makefile
├── models/
│   └── deepwalk_model.model
├── poetry.lock
├── pyproject.toml
├── README.md
├── recommendations_output.json
├── requirements.txt
├── src/
│   └── deepwalk_recommender/
│       ├── __init__.py
│       ├── config.py
│       ├── data_preprocessing.py
│       ├── deepwalk_model.py
│       ├── evaluate_and_tune.py
│       ├── main.py
│       ├── recommendation_system.py
│       └── schemas.py
└── tests/
    ├── __init__.py
    ├── test_data_processing.py
    ├── test_deepwalk_model.py
    ├── test_evaluate_and_tune.py
    ├── test_main.py
    └── test_recommendation_system.py
```

---

## Setup Instructions

### Prerequisites

-	Python 3.11+
-	Poetry (for dependency management)

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
**or**

```bash
poetry install
```

### 3. Download & Preprocess Data

download data/ml-100k.zip

```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
mkdir -p data
unzip ml-100k.zip -d data/
python src/deepwalk_recommender/data_preprocessing.py
```

### 4. Train & Tune DeepWalk

```bash
python src/deepwalk_recommender/evaluate_and_tune.py
```

### 5. Start API Server

```bash
python src/deepwalk_recommender/main.py
```

### Alternative Setup (via Makefile)

```bash
make install-fast     # Install dependencies
make dev              # Start server (dev mode)
make prod             # Start server (production mode)
```

---

## API Endpoints

### 1. GET /

Returns API version info.

### 2. POST /interactions

Add a new user-movie interaction.

```json
{
  "user_id": 1,
  "movie_id": 100,
  "rating": 4.5
}
```

### 3. GET /recommendations/{user_id}

Returns personalized recommendations with metadata:

```json
{
  "user_id": 1,
  "user_info": {
    "age": 24,
    "gender": "M",
    "occupation": "technician",
    "zip_code": "85711"
  },
  "recommended_items": [
    {
      "similar_movie_id": 1616,
      "title": "Desert Winds (1995)",
      "genres": [
        "Drama"
      ],
      "score": 2.84,
      "prob": 61,
      "explanation": "Recommended because you liked: Get Shorty (1995), Copycat (1995)",
      "popularity": "Hidden gem"
    }
  ]
}
```

### 4. GET /items

Returns list of all movies.

---

💡 Example Usage

```bash
curl http://localhost:8000/                             # Check API
curl http://localhost:8000/items                        # Get movies
curl http://localhost:8000/recommendations/1            # Get recs
curl -X POST http://localhost:8000/interactions \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "movie_id": 100, "rating": 4.5}' # Add new user interaction
```

---

## DeepWalk Algorithm Overview

### Steps

1.	Graph Construction: User-item bipartite graph from ratings.
2.	Random Walks: Generate sequences of nodes.
3.	Embedding Training: Use Word2Vec (Skip-gram) on sequences.
4.	Recommendation: Use cosine similarity between user and item vectors.

### Model Config

- **Embedding Dimension**: 128 (configurable)
- **Walk Length**: 80 steps
- **Number of Walks**: 10 per node
- **Window Size**: 5 (for Word2Vec)


### Model Performance (on MovieLens 100k)
-	**Accuracy**: 83.89%
-	**Precision**: 82.81%
-	**Recall**: 85.72%
-	**F1-Score**: 84.24%

---

## Testing

Run unit and integration tests:

```bash
pytest tests/
```

**or**

```bash
make test
```

---

## Technical Notes

### Scalability

-	Handles large data via sparse matrices.
-	Stateless API: horizontally scalable.

### Performance

-	Precomputed embeddings, vectorized similarity.
-	Parallelized random walk generation.

### Planned Enhancements
-	Real-time updates
-	Cold-start handling
-	Content-based hybrid filtering
-	A/B testing for strategies

---

## Dependencies

-	FastAPI, Uvicorn
-	Gensim, NetworkX, NumPy, Pandas
-	Scikit-learn, Pytest
-	(Optional) TensorFlow for future extensions

---

## License

This project is provided for educational and evaluation purposes as part of a technical assessment.