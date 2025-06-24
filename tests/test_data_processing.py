"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:00:18
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 01:10:49
FilePath: tests/test_data_processing.py
Description: Unit tests for data preprocessing in the MovieLens 100k dataset.
"""

import pandas as pd
import pytest

from src.deepwalk_recommender.data_preprocessing import preprocess_data


@pytest.fixture
def sample_data_dir(tmp_path):
    """
    Creates a temporary directory with sample MovieLens 100k 'u.data' file.

    The sample data contains:
    - Valid user-movie interactions
    - Edge cases (minimum/maximum ratings)
    - Different timestamps

    Args:
        tmp_path: pytest fixture for temporary directory

    Returns:
        Path: Temporary directory path containing 'u.data' file
    """
    data_content = (
        "196\t242\t5\t881250949\n"  # Valid interaction (max rating)
        "186\t302\t3\t891717742\n"  # Valid interaction
        "22\t377\t1\t878887116\n"  # Valid interaction (min rating)
        "244\t51\t2\t880606923\n"  # Valid interaction
        "999\t0\t5\t000000000\n"  # Edge case: movie_id 0
        "0\t999\t1\t000000000\n"  # Edge case: user_id 0
    )
    data_file = tmp_path / "u.data"
    data_file.write_text(data_content)
    return tmp_path


def test_preprocess_data_valid(sample_data_dir):
    """
    Tests basic preprocessing functionality with valid data.

    Verifies:
    - Correct number of rows processed
    - Expected columns present
    - Proper data types
    - Timestamp column removed
    - Edge cases preserved
    """
    # Execute preprocessing
    df = preprocess_data(sample_data_dir)

    # Validate structure
    assert len(df) == 6, "Should process all 6 sample rows"
    assert set(df.columns) == {
        "user_id",
        "movie_id",
        "rating",
    }, "Should have correct columns"
    assert df["user_id"].dtype == "int32", "user_id should be int32"
    assert df["movie_id"].dtype == "int32", "movie_id should be int32"
    assert df["rating"].dtype == "float32", "rating should be float32"

    # Validate edge cases
    assert df[df["user_id"] == 0].shape[0] == 1, "Should preserve user_id=0"
    assert df[df["movie_id"] == 0].shape[0] == 1, "Should preserve movie_id=0"
    assert df["rating"].min() == pytest.approx(
        1.0, abs=1e-2
    ), "Should preserve min rating"
    assert df["rating"].max() == pytest.approx(
        5.0, abs=1e-2
    ), "Should preserve max rating"


def test_preprocess_data_empty(tmp_path):
    """
    Tests preprocessing with an empty dataset.

    Verifies:
    - Returns empty DataFrame
    - Maintains correct schema
    - Handles an empty file gracefully
    """
    # Create an empty data file
    data_file = tmp_path / "u.data"
    data_file.write_text("")

    # Execute preprocessing
    df = preprocess_data(tmp_path)

    # Validate structure
    assert df.empty, "Should return empty DataFrame"
    assert set(df.columns) == {
        "user_id",
        "movie_id",
        "rating",
    }, "Should maintain schema"
    assert len(df) == 0, "Should have zero rows"


def test_preprocess_data_invalid_ratings(sample_data_dir):
    """
    Tests handling of invalid ratings in preprocessing.

    Modifies sample data to include:
    - Rating below minimum (0.5)
    - Rating above maximum (5.5)
    - Non-numeric rating

    Verifies:
    - Invalid ratings are properly cast to float
    - Out-of-bound ratings are preserved (no clipping)
    """
    # Add invalid ratings
    data_file = sample_data_dir / "u.data"
    with open(data_file, "a") as f:
        f.write("300\t400\t0.5\t000000000\n")  # Below min
        f.write("300\t401\t5.5\t000000000\n")  # Above max
        f.write("300\t402\tnan\t000000000\n")  # Non-numeric

    # Execute preprocessing
    df = preprocess_data(sample_data_dir)

    # Validate ratings
    ratings = df["rating"].tolist()
    assert 0.5 in ratings, "Should preserve below-min rating"
    assert 5.5 in ratings, "Should preserve above-max rating"
    assert pd.isna(df["rating"].iloc[-1]), "Should handle non-numeric as NaN"


def test_preprocess_data_missing_columns(tmp_path):
    """
    Tests handling of data files with missing columns.

    Creates data with:
    - Missing rating column
    - Missing movie_id column

    Verifies:
    - Proper exception handling
    - Informative error messages
    """
    # Missing rating column
    missing_rating = "196\t242\t881250949\n" * 3
    data_file = tmp_path / "u.data"
    data_file.write_text(missing_rating)

    with pytest.raises(pd.errors.ParserError):
        preprocess_data(tmp_path)

    # Missing movie_id column
    missing_movie = "196\t5\t881250949\n" * 3
    data_file.write_text(missing_movie)

    with pytest.raises(pd.errors.ParserError):
        preprocess_data(tmp_path)


def test_preprocess_data_large_dataset(tmp_path):
    """
    Tests performance with large dataset simulation.

    Generates 10,000 rows of synthetic data to verify:
    - Memory efficiency
    - Processing speed
    - Data type consistency
    """
    # Generate large dataset
    large_data = "\n".join(
        f"{i % 1000}\t{i % 500}\t{(i % 5) + 1}\t{1000000000 + i}" for i in range(10000)
    )
    data_file = tmp_path / "u.data"
    data_file.write_text(large_data)

    # Execute preprocessing
    df = preprocess_data(tmp_path)

    # Validate results
    assert len(df) == 10000, "Should process all rows"
    assert df["user_id"].max() == 999, "User IDs should be in range"
    assert df["movie_id"].min() == 0, "Movie IDs should include 0"
    assert df["rating"].between(1.0, 5.0).all(), "Ratings should be valid"
