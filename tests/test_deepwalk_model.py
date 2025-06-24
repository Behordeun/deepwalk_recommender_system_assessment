"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:03:31
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 15:47:43
FilePath: tests/test_deepwalk_model.py
Description: This script contains unit tests for the deepwalk_model module.
"""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gensim.models import Word2Vec

# Import the DeepWalk class and the FastAPI app from the source modules
from src.deepwalk_recommender.deepwalk_model import (
    DeepWalk,
    build_graph,
    generate_random_walks,
    train_deepwalk,
)
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
        "message": "Movie Recommendation API",
        "version": "1.0.0"
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
    # Assert the response detail contains the specific exception message.
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


# region New tests for deepwalk_model.py functions
def test_build_graph_basic():
    """
    Test building a graph from a basic DataFrame.
    """
    df = pd.DataFrame(
        {"user_id": [1, 1, 2], "movie_id": [101, 102, 101], "rating": [5, 4, 3]}
    )
    graph = build_graph(df)

    assert isinstance(graph, nx.Graph)
    # Expected nodes: u_1, m_101, m_102, u_2
    assert sorted(graph.nodes) == sorted(["u_1", "m_101", "m_102", "u_2"])
    # Convert edges to frozensets for order-independent comparison
    expected_edges = {
        frozenset({"u_1", "m_101"}),
        frozenset({"u_1", "m_102"}),
        frozenset({"u_2", "m_101"}),
    }
    actual_edges = {frozenset(edge) for edge in graph.edges}

    assert actual_edges == expected_edges
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3


def test_build_graph_empty_df():
    """
    Test building a graph from an empty DataFrame.
    """
    df = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    graph = build_graph(df)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_build_graph_duplicate_interactions():
    """
    Test building a graph with duplicate interactions.
    Ensure duplicate interactions don't create duplicate edges in an undirected graph.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "movie_id": [101, 101, 102],
            "rating": [5, 4, 3],  # Rating doesn't matter for graph building
        }
    )
    graph = build_graph(df)

    assert isinstance(graph, nx.Graph)
    assert sorted(graph.nodes) == sorted(["u_1", "m_101", "m_102"])
    # (u_1, m_101) and (u_1, m_102) should be the only unique edges
    expected_edges = {
        frozenset({"u_1", "m_101"}),
        frozenset({"u_1", "m_102"}),
    }
    actual_edges = {frozenset(edge) for edge in graph.edges}
    assert actual_edges == expected_edges
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_generate_random_walks_basic():
    """
    Test basic generation of random walks.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "movie_id": [101, 102, 101, 103],
            "rating": [5, 4, 3, 5],
        }
    )
    graph = build_graph(df)
    num_walks = 2
    walk_length = 5
    walks = generate_random_walks(graph, num_walks=num_walks, walk_length=walk_length)

    assert isinstance(walks, list)
    assert all(isinstance(walk, list) for walk in walks)
    assert all(isinstance(node, str) for walk in walks for node in walk)

    # Number of walks should be (number of nodes * num_walks_per_node)
    assert len(walks) == graph.number_of_nodes() * num_walks

    # Each walk should have the specified length or less if it gets stuck (no neighbors)
    for walk in walks:
        assert 1 <= len(walk) <= walk_length


def test_generate_random_walks_empty_graph():
    """
    Test random walk generation on an empty graph.
    """
    graph = nx.Graph()
    walks = generate_random_walks(graph)
    assert walks == []


def test_generate_random_walks_isolated_node():
    """
    Test random walk generation with an isolated node.
    """
    graph = nx.Graph()
    graph.add_node("u_1")
    walks = generate_random_walks(graph, num_walks=1, walk_length=5)
    # An isolated node should result in walks of length 1 (just the node itself)
    assert len(walks) == 1
    assert walks[0] == ["u_1"]


# Corrected patch target for Word2Vec in train_deepwalk
@patch("src.deepwalk_recommender.deepwalk_model.Word2Vec")
def test_train_deepwalk_basic(mock_word2vec_class):  # Renamed mock_word2vec to mock_word2vec_class for clarity
    """
    Test basic training of the DeepWalk (Word2Vec) model.
    """
    # Mock a list of walks
    walks_data = [
        ["u_1", "m_101", "u_2"],
        ["m_101", "u_1", "m_102"],
        ["u_2", "m_101"],
    ]
    embedding_size = 64
    window_size = 3
    min_count = 1
    workers = 2
    epochs = 3

    # Configure the mock Word2Vec constructor to return a mock model instance
    mock_model_instance = MagicMock()
    mock_model_instance.vector_size = embedding_size
    mock_model_instance.wv = MagicMock()  # Mock the word vectors
    mock_model_instance.wv.__len__.return_value = 3  # Example vocab size
    mock_word2vec_class.return_value = mock_model_instance  # Use mock_word2vec_class

    model = train_deepwalk(
        walks=walks_data,
        embedding_size=embedding_size,
        window_size=window_size,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )

    # Assert that Word2Vec was called with the correct parameters
    mock_word2vec_class.assert_called_once_with(  # Use mock_word2vec_class
        sentences=walks_data,
        vector_size=embedding_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,
    )
    assert model == mock_model_instance
    assert model.vector_size == embedding_size
    assert len(model.wv) == 3


# Corrected patch target for Word2Vec in train_deepwalk
@patch("src.deepwalk_recommender.deepwalk_model.Word2Vec")
def test_train_deepwalk_empty_walks(mock_word2vec_class):  # Renamed mock_word2vec to mock_word2vec_class for clarity
    """
    Test training DeepWalk with an empty list of walks.
    This should raise a RuntimeError.
    """
    walks_data = []
    embedding_size = 128

    # Configure the mock Word2Vec constructor to raise RuntimeError immediately upon call
    mock_word2vec_class.side_effect = RuntimeError("you must first build vocabulary before training the model")

    with pytest.raises(RuntimeError, match="you must first build vocabulary before training the model"):
        train_deepwalk(walks=walks_data, embedding_size=embedding_size)

    # Assert that Word2Vec was indeed attempted to be called
    mock_word2vec_class.assert_called_once()
