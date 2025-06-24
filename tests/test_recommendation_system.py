"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:01:01
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 15:50:47
FilePath: tests/test_recommendation_system.py
Description: This script tests the RecommendationSystem class for movie recommendations using the MovieLens 100k dataset.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from gensim.models import Word2Vec

from src.deepwalk_recommender.config import PathConfig
from src.deepwalk_recommender.recommendation_system import RecommendationSystem
from src.deepwalk_recommender.schemas import UserInfo  # Import UserInfo


@pytest.fixture(scope="module")
def recommender():
    """Real recommendation system instance using actual configuration paths"""
    return RecommendationSystem(
        model_path=PathConfig.DEEPWALK_MODEL_PATH,
        data_path=PathConfig.PROCESSED_DATA_FILE,
        ml_100k_dir=PathConfig.ML_100K_DIR,
    )


@pytest.fixture
def mock_word2vec_model():
    """Mocked Word2Vec model with controlled embeddings"""
    model = MagicMock(spec=Word2Vec)
    model.wv = MagicMock()
    valid_keys = {"u_1", "m_1", "m_2"}
    model.wv.__contains__.side_effect = lambda key: key in valid_keys

    def get_embedding_side_effect(key):
        if key == "u_1":
            return np.array([0.1, 0.2])
        elif key == "m_1":
            return np.array([0.3, 0.4])
        else:
            return np.array([0.5, 0.6])

    model.wv.__getitem__.side_effect = get_embedding_side_effect
    return model


@pytest.fixture
def sample_processed_ratings_path(tmp_path):
    """Mock user-movie ratings data"""
    df_content = """user_id,movie_id,rating
1,1,5
1,2,4
2,1,3
2,3,5
3,4,4
"""
    file_path = tmp_path / "processed_ratings.csv"
    file_path.write_text(df_content)
    return file_path


@pytest.fixture
def sample_u_item_path(tmp_path):
    """Mock movie metadata with the correct format (24 columns)"""
    # Format: movie_id|title|release_date|video_release_date|imdb_url|...genres...
    item_content = """1|Toy Story|1995||http://us.imdb.com|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
2|GoldenEye|1995||http://us.imdb.com|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
3|Four Rooms|1995||http://us.imdb.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
4|Get Shorty|1995||http://us.imdb.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
"""
    file_path = tmp_path / "u.item"
    file_path.write_text(item_content)
    return file_path


@pytest.fixture
def sample_u_user_path(tmp_path):
    """Mock user metadata"""
    user_content = """1|24|M|technician|85711
2|53|F|other|94043
"""
    file_path = tmp_path / "u.user"
    file_path.write_text(user_content)
    return file_path


@pytest.fixture
def rec_system_instance(
    mock_word2vec_model,
    sample_processed_ratings_path,
    sample_u_item_path,
    sample_u_user_path,
    tmp_path,
):
    """Fully mocked recommendation system instance with controlled dependencies"""
    with patch("gensim.models.Word2Vec.load", return_value=mock_word2vec_model):
        ml_100k_dir = tmp_path / "ml-100k"
        ml_100k_dir.mkdir(exist_ok=True)
        (ml_100k_dir / "u.item").write_text(sample_u_item_path.read_text())
        (ml_100k_dir / "u.user").write_text(sample_u_user_path.read_text())

        return RecommendationSystem(
            model_path="dummy_model.model",
            data_path=str(sample_processed_ratings_path),
            ml_100k_dir=str(ml_100k_dir),
        )


def test_init(rec_system_instance):
    """Test system initialization and data loading"""
    assert isinstance(rec_system_instance.df, pd.DataFrame)
    assert isinstance(rec_system_instance.movies_df, pd.DataFrame)
    assert isinstance(rec_system_instance.users_df, pd.DataFrame)
    assert rec_system_instance.model is not None
    assert rec_system_instance.movies_df["movie_id"].dtype == int


def test_create_user_movie_matrix(rec_system_instance):
    """Test user-movie matrix creation"""
    matrix = rec_system_instance._create_user_movie_matrix()
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.loc[1, 1] == 5
    assert matrix.loc[1, 2] == 4
    assert matrix.loc[2, 1] == 3
    assert matrix.shape == (3, 4)  # 3 users, 4 movies


def test_load_movies():
    """Test movie metadata loading with the correct column alignment"""
    mock_data = StringIO(
        "1|Toy Story|1995||http://example.com|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        "2|GoldenEye|1995||http://example.com|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0"
    )
    column_names = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    df = pd.read_csv(mock_data, sep="|", names=column_names, encoding="latin-1")
    assert df.iloc[0]["title"] == "Toy Story"
    assert df.iloc[1]["title"] == "GoldenEye"
    assert df.iloc[0]["movie_id"] == 1
    assert "Children's" in df.columns


def test_load_users(rec_system_instance, sample_u_user_path):
    """Test user metadata loading"""
    users_df = rec_system_instance._load_users(str(sample_u_user_path))
    assert not users_df.empty
    assert "occupation" in users_df.columns
    assert users_df.iloc[0]["occupation"] == "technician"
    assert users_df.iloc[0]["user_id"] == 1
    # Verify zip_code is string type after loading and conversion
    assert users_df.iloc[0]["zip_code"] == "85711"
    assert isinstance(users_df.iloc[0]["zip_code"], str)


def test_get_user_embedding(rec_system_instance):
    """Test user embedding retrieval"""
    embedding = rec_system_instance.get_user_embedding(1)
    assert isinstance(embedding, np.ndarray)
    assert np.array_equal(embedding, np.array([0.1, 0.2]))
    assert rec_system_instance.get_user_embedding(999) is None


def test_get_movie_embedding(rec_system_instance):
    """Test movie embedding retrieval"""
    embedding = rec_system_instance.get_movie_embedding(1)
    assert isinstance(embedding, np.ndarray)
    assert np.array_equal(embedding, np.array([0.3, 0.4]))
    assert rec_system_instance.get_movie_embedding(999) is None


def test_predict_rating(rec_system_instance):
    """Test rating prediction logic"""
    # Test valid prediction
    with patch(
        "src.deepwalk_recommender.recommendation_system.cosine_similarity",
        return_value=np.array([[0.5]]),
    ) as mock_cosine:
        predicted_rating = rec_system_instance.predict_rating(1, 1)
        assert predicted_rating == pytest.approx(4.0)
        mock_cosine.assert_called_once()

    # Test missing user embedding
    with patch.object(rec_system_instance, "get_user_embedding", return_value=None):
        assert rec_system_instance.predict_rating(999, 1) == pytest.approx(3.0)

    # Test missing movie embedding
    with patch.object(rec_system_instance, "get_movie_embedding", return_value=None):
        assert rec_system_instance.predict_rating(1, 999) == pytest.approx(3.0)


def test_get_recommendations(rec_system_instance):
    """Test recommendation generation with diversity"""
    # Mock rating prediction to control scores
    with patch.object(rec_system_instance, "predict_rating") as mock_predict_rating:
        # Set a high score for movie 3
        mock_predict_rating.side_effect = lambda u, m: 4.5 if m == 3 else 3.0

        recommendations = rec_system_instance.get_recommendations(1, top_k=1)
        assert len(recommendations) == 1
        assert recommendations[0]["similar_movie_id"] == 3
        assert recommendations[0]["title"] == "Four Rooms"
        assert "score" in recommendations[0]
        assert "explanation" in recommendations[0]


def test_add_interaction(rec_system_instance):
    """Test adding new user interactions"""
    initial_rows = len(rec_system_instance.df)
    rec_system_instance.add_interaction(3, 1, 4)
    assert len(rec_system_instance.df) == initial_rows + 1
    assert rec_system_instance.df.iloc[-1]["user_id"] == 3
    assert rec_system_instance.df.iloc[-1]["movie_id"] == 1
    assert rec_system_instance.df.iloc[-1]["rating"] == 4

    # Verify matrix updated
    matrix = rec_system_instance.user_movie_matrix
    assert matrix.loc[3, 1] == 4


def test_get_all_movies(rec_system_instance):
    """Test retrieval of all movies with metadata"""
    all_movies = rec_system_instance.get_all_movies()
    assert isinstance(all_movies, list)
    assert len(all_movies) == 4
    assert all_movies[0]["movie_id"] == 1
    assert all_movies[0]["title"] == "Toy Story"
    assert isinstance(all_movies[0]["movie_id"], int)


def test_get_user_info(rec_system_instance):
    """Test user metadata retrieval and Pydantic model creation"""
    user_info = rec_system_instance.get_user_info(1)
    assert isinstance(user_info, UserInfo)  # Assert it's a UserInfo object
    assert user_info.occupation == "technician"
    assert user_info.age == 24
    assert user_info.zip_code == "85711"  # Ensure zip_code is string

    # Test non-existent user
    assert rec_system_instance.get_user_info(999) is None
