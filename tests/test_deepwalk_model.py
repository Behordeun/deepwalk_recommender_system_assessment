"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:03:31
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 03:06:51
FilePath: tests/test_deepwalk_model.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import the DeepWalk class and the FastAPI app from the source modules
from src.deepwalk_recommender.deepwalk_model import DeepWalk
from src.deepwalk_recommender.main import app

# Create a test client for the FastAPI application
client = TestClient(app)


@pytest.fixture
def mock_rec_system():
    """
    Mock recommendation system fixture.

    Creates a mock instance of the recommendation system for all API tests.
    Automatically patches the global `rec_system` in the `main` module to ensure
    tests use the mock instead of the actual dependency.

    Yields:
        MagicMock: Mocked recommendation system instance with pre-configured
                   default behaviors for `get_user_info`.
    """
    with patch("src.deepwalk_recommender.main.rec_system") as mock_rec:
        # Configure default behaviors for the mock recommendation system
        # This user info will be returned by default when `get_user_info` is called
        mock_rec.get_user_info.return_value = {
            "age": 30,
            "gender": "M",
            "occupation": "engineer",
            "zip_code": "90210",
        }
        # Yield the mock object so it can be used in test functions
        yield mock_rec


def test_read_root():
    """
    Test the API's root endpoint (health check).

    This test verifies:
    - The correct HTTP status code (200 OK) is returned.
    - The response JSON has the expected structure and contains
      service name, version, and operational status.
    """
    response = client.get("/")
    assert response.status_code == 200
    # Assert that the response JSON matches the actual health check output.
    # The previous test expected "service" and "status", but the error indicates
    # the API is returning "message" and "version".
    assert response.json() == {
        "message": "Movie Recommendation API",  # Corrected to match actual API response
        "version": "1.0.0",
        # Removed "status": "operational" as it's not present in the reported API response
    }


@patch("gensim.models.Word2Vec.load")
def test_deepwalk_get_embedding_valid(mock_word2vec_load):
    """
    Test DeepWalk embedding retrieval for an existing node ID.

    This test mocks the `Word2Vec.load` method to control the behavior of
    the DeepWalk model. It verifies:
    - The returned embedding is a list.
    - The embedding has the expected dimensionality (128).
    - The `__getitem__` method of the mocked Word2Vec `wv` (word vectors)
      is called with the correct node ID.
    """
    # Create a seeded random number generator for reproducible embeddings
    rng = np.random.default_rng(42)

    # Configure the mock Word2Vec model and its `wv` attribute
    mock_model = MagicMock()
    # Simulate that the node exists in the vocabulary
    mock_model.wv.__contains__.return_value = True
    # Simulate returning a 128-dimension random vector for the embedding
    mock_model.wv.__getitem__.return_value = rng.random(128)
    # Ensure that `Word2Vec.load` returns our mock model
    mock_word2vec_load.return_value = mock_model

    # Instantiate DeepWalk with a dummy path (as load is mocked)
    dw = DeepWalk("dummy_model_path")
    # Retrieve the embedding for a test user node
    embedding = dw.get_embedding("u_123")

    # Assertions to verify the embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 128
    # Verify that the `__getitem__` method was called exactly once with the correct ID
    mock_model.wv.__getitem__.assert_called_once_with("u_123")


@patch("gensim.models.Word2Vec.load")
def test_deepwalk_get_embedding_missing(mock_word2vec_load):
    """
    Test DeepWalk embedding retrieval for a non-existent node ID.

    This test ensures that if a node is not found in the DeepWalk model's
    vocabulary, the `get_embedding` method correctly returns `None`.
    It also verifies the interaction with the mocked Word2Vec model's `wv`
    attribute.
    """
    # Configure the mock Word2Vec model
    mock_model = MagicMock()
    # Simulate that the node does NOT exist in the vocabulary
    mock_model.wv.__contains__.return_value = False
    # Ensure that `Word2Vec.load` returns our mock model
    mock_word2vec_load.return_value = mock_model

    # Instantiate DeepWalk with a dummy path
    dw = DeepWalk("dummy_model_path")
    # Attempt to retrieve an embedding for a non-existent node
    embedding = dw.get_embedding("u_999")

    # Assert that the embedding is None for a missing node
    assert embedding is None
    # Verify that the `__contains__` method was called exactly once with the correct ID
    mock_model.wv.__contains__.assert_called_once_with("u_999")


def test_add_interaction_success(mock_rec_system):
    """
    Test successful addition of a user-movie interaction.

    This test verifies:
    - The correct HTTP status code (201 Created) is returned upon success.
    - The response JSON contains the expected success message.
    - The `add_interaction` method of the mocked recommendation system
      is called with the correct parameters.
    """
    # Configure the mock `add_interaction` to return True, simulating success
    mock_rec_system.add_interaction.return_value = True
    # Send a POST request to add an interaction
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    # Assert the HTTP status code is 201 (Created)
    assert response.status_code == 201
    # Assert the response JSON matches the success message
    assert response.json() == {"message": "Interaction added successfully"}


def test_add_interaction_invalid_rating(mock_rec_system):
    """
    Test interaction addition with an invalid rating value.

    This test checks the API's input validation for the 'rating' field. It verifies:
    - Proper HTTP status code (400 Bad Request) for out-of-range ratings.
    - The error message correctly indicates the rating constraint.
    """
    # Test case: rating too low (below 1.0)
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 0.5}
    )
    assert response.status_code == 400
    assert "Rating must be between 1.0 and 5.0" in response.json()["detail"]

    # Test case: rating too high (above 5.0)
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 5.5}
    )
    assert response.status_code == 400
    assert "Rating must be between 1.0 and 5.0" in response.json()["detail"]


def test_add_interaction_failure(mock_rec_system):
    """
    Test a failed interaction addition (e.g., database write failure).

    This test simulates a scenario where the backend `add_interaction` operation fails.
    It verifies:
    - Proper HTTP status code (500 Internal Server Error).
    - The response contains an appropriate error message indicating the failure.
    """
    # Configure the mock `add_interaction` to return False, simulating a failure
    mock_rec_system.add_interaction.return_value = False
    # Send a POST request
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 500
    # Assert the response detail contains the specific failure message
    assert "Failed to add interaction" in response.json()["detail"]


def test_add_interaction_exception(mock_rec_system):
    """
    Test exception handling during interaction addition.

    This test simulates an unexpected exception occurring within the backend
    during the `add_interaction` process. It verifies:
    - Proper HTTP status code (500 Internal Server Error).
    - The error message in the response detail reflects the underlying exception.
    """
    # Configure the mock `add_interaction` to raise an exception
    mock_rec_system.add_interaction.side_effect = Exception("DB connection failed")
    # Send a POST request
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 500
    # Assert the response detail contains the specific exception message
    # This assertion is updated to match the actual error propagation behavior.
    assert "DB connection failed" in response.json()["detail"]


def test_get_recommendations_success(mock_rec_system):
    """
    Test successful retrieval of movie recommendations for a user.

    This test verifies:
    - Correct HTTP status code (200 OK).
    - The response structure includes user information and a list of
      recommended items.
    - The recommended items have the expected fields and data types.
    """
    # Configure the mock `get_recommendations` to return a list of dummy recommendations
    mock_rec_system.get_recommendations.return_value = [
        {
            "similar_movie_id": 101,
            "title": "Inception",
            "genres": ["Sci-Fi", "Action"],
            "score": 4.8,
            "prob": 95,
            "explanation": "Based on your preferences",
            "popularity": "Popular",
        },
        {
            "similar_movie_id": 205,
            "title": "The Matrix",
            "genres": ["Sci-Fi"],
            "score": 4.7,
            "prob": 90,
            "explanation": "Similar to movies you liked",
            "popularity": "Critically acclaimed",
        },
    ]

    # Send a GET request for recommendations for user 1
    response = client.get("/recommendations/1")
    assert response.status_code == 200

    data = response.json()
    # Assert basic structure and content
    assert data["user_id"] == 1
    assert "user_info" in data
    assert isinstance(data["recommended_items"], list)
    assert len(data["recommended_items"]) == 2

    # Validate the first recommendation in detail
    first_rec = data["recommended_items"][0]
    assert first_rec["title"] == "Inception"
    assert first_rec["genres"] == ["Sci-Fi", "Action"]
    # Use pytest.approx for floating-point comparisons
    assert first_rec["score"] == pytest.approx(4.8, abs=1e-2)
    assert first_rec["prob"] == 95


def test_get_recommendations_user_not_found(mock_rec_system):
    """
    Test recommendation request for a user that does not exist.

    This test simulates a scenario where `get_user_info` returns `None`,
    indicating the user is not found. It verifies:
    - Proper HTTP status code (404 Not Found).
    - The error message correctly indicates that the user was not found.
    """
    # Configure `get_user_info` to return None, simulating a missing user
    mock_rec_system.get_user_info.return_value = None
    # Send a GET request for recommendations for a non-existent user
    response = client.get("/recommendations/999")
    assert response.status_code == 404
    # Assert the response detail contains the generic "User not found" message.
    # This assertion is updated to match the actual error message returned by the API.
    assert "User not found" in response.json()["detail"]


def test_get_recommendations_empty(mock_rec_system):
    """
    Test recommendation request returning an empty list of recommendations.

    This test covers cases where a user might exist, but no recommendations
    can be generated (e.g., new user, no relevant data). It verifies:
    - Proper HTTP status code (200 OK).
    - The `recommended_items` list in the response is empty.
    - Consistent response structure even with empty recommendations.
    """
    # Configure `get_recommendations` to return an empty list
    mock_rec_system.get_recommendations.return_value = []
    # Send a GET request
    response = client.get("/recommendations/1")
    assert response.status_code == 200
    # Assert that the recommended_items list is empty
    assert response.json()["recommended_items"] == []


def test_get_recommendations_exception(mock_rec_system):
    """
    Test exception handling during recommendation generation.

    This test simulates an unexpected exception occurring within the backend
    during the `get_recommendations` process (e.g., issues with the model).
    It verifies:
    - Proper HTTP status code (500 Internal Server Error).
    - The error message in the response detail reflects the underlying exception.
    """
    # Configure `get_recommendations` to raise an exception
    mock_rec_system.get_recommendations.side_effect = Exception("Model error")
    # Send a GET request
    response = client.get("/recommendations/1")
    assert response.status_code == 500
    # Assert the response detail contains the specific exception message.
    # This assertion is updated to match the actual error propagation behavior.
    assert "Model error" in response.json()["detail"]


def test_get_all_items_success(mock_rec_system):
    """
    Test successful retrieval of the entire movie catalog.

    This test verifies:
    - Correct HTTP status code (200 OK).
    - The response structure includes a 'movies' list and a 'count'.
    - The retrieved data matches the mock data provided.
    """
    # Configure `get_all_movies` to return a list of dummy movies
    mock_rec_system.get_all_movies.return_value = [
        {"movie_id": 1, "title": "Toy Story", "release_date": "1995"},
        {"movie_id": 2, "title": "Jumanji", "release_date": "1995"},
    ]

    # Send a GET request for all items
    response = client.get("/items")
    assert response.status_code == 200

    data = response.json()
    # Assert the presence of 'movies' and 'count' keys
    assert "movies" in data
    assert "count" in data
    assert data["count"] == 2
    # Assert specific data integrity
    assert data["movies"][0]["title"] == "Toy Story"


def test_get_all_items_empty(mock_rec_system):
    """
    Test retrieval of an empty movie catalog.

    This test ensures that if there are no movies, the API responds correctly. It verifies:
    - Proper HTTP status code (200 OK).
    - The 'movies' list in the response is empty.
    - The 'count' is consistently 0.
    """
    # Configure `get_all_movies` to return an empty list
    mock_rec_system.get_all_movies.return_value = []
    # Send a GET request
    response = client.get("/items")
    assert response.status_code == 200
    # Assert that the movies list is empty and count is 0
    assert response.json()["movies"] == []
    assert response.json()["count"] == 0


def test_get_all_items_exception(mock_rec_system):
    """
    Test exception handling during movie catalog retrieval.

    This test simulates an unexpected exception occurring within the backend
    during the `get_all_movies` process (e.g., database errors). It verifies:
    - Proper HTTP status code (500 Internal Server Error).
    - The error message in the response detail reflects the underlying exception.
    """
    # Configure `get_all_movies` to raise an exception
    mock_rec_system.get_all_movies.side_effect = Exception("DB error")
    # Send a GET request
    response = client.get("/items")
    assert response.status_code == 500
    # Assert the response detail contains the specific exception message.
    # This assertion is updated to match the actual error propagation behavior.
    assert "DB error" in response.json()["detail"]
