"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:04:12
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 03:02:24
FilePath: tests/test_main.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.deepwalk_recommender.main import app

client = TestClient(app)


@pytest.fixture
def mock_rec_system():
    """
    Pytest fixture that mocks the 'rec_system' object in the main module for testing purposes.

    Yields:
        MagicMock: A mock instance of the recommendation system.
    """
    with patch("src.deepwalk_recommender.main.rec_system", autospec=True) as mock_rec:
        yield mock_rec


def test_read_root():
    """
    Test the root endpoint to ensure it returns the correct API message and version.
    This test now expects the 'message' key as per the updated main.py.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
    }


def test_add_interaction_success(mock_rec_system):
    """
    Test the /interactions endpoint for successful addition of a user-movie interaction.

    Verifies that the API returns a 201 status (Created) and the correct success message,
    and that the add_interaction method is called with the expected arguments.
    """
    mock_rec_system.add_interaction.return_value = True
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 201
    assert response.json() == {"message": "Interaction added successfully"}
    mock_rec_system.add_interaction.assert_called_once_with(1, 100, 4.5)


@patch("src.deepwalk_recommender.main.rec_system")
def test_get_recommendations_success(mock_rec_system):
    """
    Test the /recommendations/{user_id} endpoint for successful retrieval of movie recommendations.

    Mocks the recommendation system to return a sample recommendation and user info,
    then asserts that the API response contains the expected user data and recommended items.
    """
    mock_rec_system.get_recommendations.return_value = [
        {
            "similar_movie_id": 101,
            "title": "The Matrix",
            "genres": ["Action", "Sci-Fi"],
            "score": 4.5,
            "prob": 90,
            "explanation": "Recommended because you liked: Terminator 2",
            "popularity": "Critically acclaimed",
        }
    ]

    mock_rec_system.get_user_info.return_value = {
        "age": 37,
        "gender": "M",
        "occupation": "developer",
        "zip_code": "10001",
    }

    response = client.get("/recommendations/1")
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == 1
    assert "user_info" in data
    assert "recommended_items" in data
    assert len(data["recommended_items"]) == 1

    item = data["recommended_items"][0]
    assert item["title"] == "The Matrix"
    assert item["genres"] == ["Action", "Sci-Fi"]
    assert item["score"] == pytest.approx(4.5, abs=1e-2)
    assert item["prob"] == 90
    assert item["explanation"] == "Recommended because you liked: Terminator 2"
    assert item["popularity"] == "Critically acclaimed"


def test_add_interaction_exception(mock_rec_system):
    """
    Test the /interactions endpoint to ensure a 500-error and correct error message are returned when an exception occurs during interaction addition.
    This test is updated to expect the exact exception message string, as per main.py's changes.
    """
    mock_exception_message = "DB connection failed"
    mock_rec_system.add_interaction.side_effect = Exception(mock_exception_message)
    response = client.post(
        "/interactions", json={"user_id": 1, "movie_id": 100, "rating": 4.5}
    )
    assert response.status_code == 500
    # Updated assertion to match the exact exception message passed from main.py
    assert response.json()["detail"] == mock_exception_message


def test_get_recommendations_user_not_found(mock_rec_system):
    """
    Test the /recommendations/{user_id} endpoint to ensure a 404 error and
    the generic "User not found" message are returned when the user is not found.
    This matches the updated main.py behavior.
    """
    mock_rec_system.get_recommendations.return_value = []
    mock_rec_system.get_user_info.return_value = None
    response = client.get("/recommendations/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}


def test_get_recommendations_exception(mock_rec_system):
    """
    Test the /recommendations/{user_id} endpoint to ensure a 500-error and
    the exact exception message are returned when an exception occurs during recommendation retrieval.
    This matches the updated main.py behavior.
    """
    mock_exception_message = "Test Exception"
    mock_rec_system.get_recommendations.side_effect = Exception(mock_exception_message)
    mock_rec_system.get_user_info.return_value = {
        "user_id": 1,
        "age": 30,
        "occupation": "engineer",
    }
    response = client.get("/recommendations/1")
    assert response.status_code == 500
    assert response.json() == {"detail": mock_exception_message}


def test_get_all_items_success(mock_rec_system):
    """
    Test the /items endpoint for successful retrieval of all movies.

    Verifies that the API returns a 200 status, the correct list of movies, and the expected count.
    Ensures get_all_movies is called exactly once on the recommendation system mock.
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
    Test the /items endpoint to ensure a 500 error and the exact exception message
    are returned when an exception occurs during movie retrieval.
    This matches the updated main.py behavior.
    """
    mock_exception_message = "Test Exception"
    mock_rec_system.get_all_movies.side_effect = Exception(mock_exception_message)
    response = client.get("/items")
    assert response.status_code == 500
    assert response.json() == {"detail": mock_exception_message}
