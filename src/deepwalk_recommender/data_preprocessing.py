"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 01:43:44
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 04:26:00
FilePath: src/deepwalk_recommender/data_preprocessing.py
Description: Data preprocessing for MovieLens 100k dataset
"""

from pathlib import Path

import pandas as pd

from src.deepwalk_recommender.config import PathConfig


def preprocess_data(_ml_100k_dir: str | Path) -> pd.DataFrame:
    """
    Processes the MovieLens 100k dataset into structured interaction records.

    Transforms the raw 'u.data' file containing user-movie interactions into a
    cleaned DataFrame optimized for recommendation systems. The processing includes:
    - Loading tab-separated data with proper column naming
    - Selecting relevant features (user_id, movie_id, rating)
    - Ensuring proper data typing for efficient storage

    Args:
       ` _ml_100k_dir (str | Path)`: Path to the MovieLens 100k directory containing 'u.data'

    Returns:
        `pd.DataFrame`: Processed data with columns:
            `user_id`: Integer user identifier
            `movie_id`: Integer movie identifier
            `rating`: Numeric rating (1-5 scale)

    Example output:
        user_id | movie_id | rating
        --------|----------|-------
        196 | 242 | 3
        186 | 302 | 3
        22 | 377 | 1
    """
    data_path = Path(_ml_100k_dir) / "u.data"

    # Define column names and dtypes for both scenarios (with and without the timestamp)
    cols_with_timestamp = ["user_id", "movie_id", "rating", "timestamp"]
    dtypes_with_timestamp = {
        "user_id": "int32",
        "movie_id": "int32",
        "rating": "float32",
        "timestamp": "int64",
    }

    cols_without_timestamp = ["user_id", "movie_id", "rating"]
    dtypes_without_timestamp = {
        "user_id": "int32",
        "movie_id": "int32",
        "rating": "float32",
    }

    try:
        # Attempt to read with timestamp column
        df = pd.read_csv(
            data_path,
            sep="\t",
            header=None,
            names=cols_with_timestamp,
            dtype=dtypes_with_timestamp,
            usecols=[0, 1, 2, 3],  # Include timestamp column for initial read
        )
    except pd.errors.ParserError:
        # If reading with timestamp fails due to ParserError, re-raise it
        raise
    except ValueError:
        # If reading with the timestamp fails due to ValueError (e.g., too few columns),
        # try reading without the timestamp column.
        try:
            df = pd.read_csv(
                data_path,
                sep="\t",
                header=None,
                names=cols_without_timestamp,
                dtype=dtypes_without_timestamp,
            )
        except Exception as inner_e:
            # If even reading without the timestamp fails, re-raise as ParserError
            raise pd.errors.ParserError(f"Failed to parse data: {inner_e}") from inner_e

    return df[["user_id", "movie_id", "rating"]]


if __name__ == "__main__":
    """Execute the data preprocessing pipeline"""
    # Process and save data
    processed_df = preprocess_data(PathConfig.ML_100K_DIR)
    processed_df.to_csv(PathConfig.PROCESSED_DATA_FILE, index=False)

    # Output confirmation
    print(
        f"Data preprocessing complete. Processed data saved to: "
        f"{PathConfig.PROCESSED_DATA_FILE}\n"
        f"Processed {len(processed_df):,} interactions "
        f"between {processed_df['user_id'].nunique():,} users "
        f"and {processed_df['movie_id'].nunique():,} movies."
    )
