"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:03:31
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 00:22:37
FilePath: tests/test_deepwalk_model.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.deepwalk_recommender.deepwalk_model import DeepWalk
from src.deepwalk_recommender.main import app

client = TestClient(app)


@pytest.fixture
def mock_rec_system():
    """
    Pytest fixture that mocks the 'rec_system' object in the DeepWalk recommender main module.

    Yields:
        MagicMock: A mock instance of the 'rec_system' for use in tests.
    """
    with patch("src.deepwalk_recommender.main.rec_system", autospec=True) as mock_rec:
        yield mock_rec


def test_read_root():
    """
    Test the root endpoint ("/") of the API to ensure it returns the expected welcome message and version information.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
    }


@patch("gensim.models.Word2Vec.load")
def test_deepwalk_get_embedding(mock_word2vec_load):
    """
    Unit test for the DeepWalk.get_embedding method.

    Mocks the Word2Vec model to verify that get_embedding returns a list of the correct length
    when the node exists in the model and checks the output type and shape.
    """
    # Create a mock Word2Vec model
    mock_model = MagicMock()
    mock_model.wv.__contains__.side_effect = lambda key: key == "0"
    mock_model.wv.__getitem__.side_effect = lambda key: np.array(
        [0.1] * 64
    )  # Use NumPy array

    # Patch load to return mock model
    mock_word2vec_load.return_value = mock_model

    # Instantiate DeepWalk using the mocked model
    dw = DeepWalk(
        model_path="mock/path/model"
    )  # The path doesn't matter due to the patch

    # Run test
    embedding = dw.get_embedding("0")

    # Assert
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == 64


def test_add_interaction_success(mock_rec_system):
    """
    Test that adding a user-movie interaction via the API returns a success message and calls the underlying method with correct parameters.
    """
    mock_rec_system.add_interaction.return_value = True
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Interaction added successfully"}
    mock_rec_system.add_interaction.assert_called_once_with(1, 100, 4.5)


def test_add_interaction_failure(mock_rec_system):
    """
    Test that the API returns a 400 error and the appropriate message when adding a user-movie interaction fails.
    """
    mock_rec_system.add_interaction.return_value = False
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Failed to add interaction"}


def test_add_interaction_exception(mock_rec_system):
    """
    Test that the API returns a 500 error and correct message when an exception occurs during add_interaction.
    """
    mock_rec_system.add_interaction.side_effect = Exception("Test Exception")
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Exception"}


def test_get_recommendations_success(mock_rec_system):
    """
    Test the API endpoint for retrieving movie recommendations for a user.

    Verifies that the response contains the correct user ID, a list of recommended items,
    and that each recommended item includes the expected fields and values.
    """
    mock_rec_system.get_recommendations.return_value = [
        {
            "similar_movie_id": 1,
            "title": "Movie A",
            "genres": ["Action"],
            "score": 4.0,
            "prob": 80,
            "explanation": "Recommended because you liked: Movie X",
            "popularity": "Popular",
        }
    ]
    mock_rec_system.get_user_info.return_value = {
        "age": 30,
        "gender": "M",
        "occupation": "engineer",
        "zip_code": "90210",
    }
    response = client.get("/recommendations/1")
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == 1
    assert "recommended_items" in data
    assert isinstance(data["recommended_items"], list)
    assert len(data["recommended_items"]) == 1

    item = data["recommended_items"][0]
    assert item["title"] == "Movie A"
    assert item["genres"] == ["Action"]
    assert item["score"] == pytest.approx(4.5, abs=1e-2)
    assert item["prob"] == 80
    assert item["explanation"] == "Recommended because you liked: Movie X"
    assert item["popularity"] == "Popular"


def test_get_recommendations_user_not_found(mock_rec_system):
    """
    Test that the API returns a 404 error and the appropriate message when recommendations are requested for a non-existent user.
    """
    mock_rec_system.get_recommendations.return_value = []
    mock_rec_system.get_user_info.return_value = None
    response = client.get("/recommendations/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}


def test_get_recommendations_exception(mock_rec_system):
    """
    Test that the API returns a 500 error and correct message when an exception occurs during recommendation retrieval.
    """
    mock_rec_system.get_recommendations.side_effect = Exception("Test Exception")
    mock_rec_system.get_user_info.return_value = {
        "user_id": 1,
        "age": 30,
        "occupation": "engineer",
    }
    response = client.get("/recommendations/1")
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Exception"}


def test_get_all_items_success(mock_rec_system):
    """
    Test that the API endpoint for retrieving all movies returns the correct data and status code and verifies the underlying method is called once.
    """
    mock_rec_system.get_all_movies.return_value = [
        {"movie_id": 1, "title": "Movie A", "release_date": "01-Jan-1995"},
        {"movie_id": 2, "title": "Movie B", "release_date": "01-Feb-1995"},
    ]
    response = client.get("/items")
    assert response.status_code == 200
    assert response.json() == {
        "movies": [
            {"movie_id": 1, "title": "Movie A", "release_date": "01-Jan-1995"},
            {"movie_id": 2, "title": "Movie B", "release_date": "01-Feb-1995"},
        ],
        "count": 2,
    }
    mock_rec_system.get_all_movies.assert_called_once()


def test_get_all_items_exception(mock_rec_system):
    """
    Test that the API returns a 500 error and correct message when an exception occurs during retrieval of all movies.
    """
    mock_rec_system.get_all_movies.side_effect = Exception("Test Exception")
    response = client.get("/items")
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Exception"}
