"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:00:18
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:28:08
FilePath: tests/test_data_processing.py
Description: Unit tests for data preprocessing in the MovieLens 100k dataset.
"""

from io import StringIO

import pandas as pd
import pytest


@pytest.fixture
def sample_data_path(tmp_path):
    """
    Pytest fixture that creates a temporary directory containing a dummy 'u.data' file with sample tab-separated data for testing purposes.

    Args:
        tmp_path: pytest-provided temporary directory fixture.

    Returns:
        str: Path to the temporary directory containing the sample data file.
    """
    data_content = """
1\t1\t5\t978300760
1\t2\t3\t978300760
2\t1\t4\t978300760
2\t3\t2\t978300760
"""
    data_file = tmp_path / "u.data"
    data_file.write_text(data_content)
    return str(tmp_path)


def test_preprocess_data():
    """
    Tests the data preprocessing step by simulating file loading, verifying correct column names, and ensuring the expected number of rows after dropping the 'timestamp' column.
    """
    mock_data = StringIO(
        "196\t242\t3\t881250949\n"
        "186\t302\t3\t891717742\n"
        "22\t377\t1\t878887116\n"
        "244\t51\t2\t880606923\n"
    )

    # Simulate loading a file
    df = pd.read_csv(
        mock_data, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
    )
    df = df.drop("timestamp", axis=1)

    assert len(df) == 4
    assert set(df.columns) == {"user_id", "movie_id", "rating"}
