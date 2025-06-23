"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 01:43:44
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:03:02
FilePath: src/deepwalk_recommender/data_preprocessing.py
Description: Data preprocessing for MovieLens 100k dataset
"""

import pandas as pd

from src.deepwalk_recommender.config import PathConfig


def preprocess_data(_ml_100k_dir):
    """
    Loads and preprocesses the MovieLens 100k dataset by reading the raw data file,
    selecting user, movie, and rating columns, and returning the resulting DataFrame.

    Args:
        _ml_100k_dir: Path to the MovieLens 100k directory (unused).

    Returns:
        pd.DataFrame: DataFrame containing user_id, movie_id, and rating columns.
    """
    data_path = PathConfig.RAW_DATA_FILE
    # Load the MovieLens 100k dataset
    # The u.data file contains user_id, item_id, rating, timestamp
    df = pd.read_csv(
        data_path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Select relevant columns
    df = df[["user_id", "movie_id", "rating"]]

    return df


if __name__ == "__main__":
    ml_100k_dir = PathConfig.ML_100K_DIR
    processed_df = preprocess_data(ml_100k_dir)
    processed_df.to_csv(PathConfig.PROCESSED_DATA_FILE, index=False)
    print(
        "Data preprocessing complete. Processed data saved to %",
        PathConfig.PROCESSED_DATA_FILE,
    )
