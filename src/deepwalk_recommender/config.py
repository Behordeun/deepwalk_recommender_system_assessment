"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 13:32:51
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 14:15:44
FilePath: src/deepwalk_recommender/config.py
Description: Configuration for paths and file locations in the DeepWalk recommender system
"""

from pathlib import Path


class PathConfig:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"

    # Data files
    RAW_DATA_FILE = DATA_DIR / "ml-100k" / "u.data"
    PROCESSED_DATA_FILE = DATA_DIR / "processed_ratings.csv"

    # Model files
    DEEPWALK_MODEL_PATH = MODEL_DIR / "best_deepwalk_model.model"
    DEEPWALK_BEST_MODEL_NAME = "best_deepwalk_model.model"
    DEEPWALK_BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"

    # Dataset folder
    ML_100K_DIR = DATA_DIR / "ml-100k"

    # Log files (can be used if needed explicitly)
    INFO_LOG = LOG_DIR / "info.log"
    ERROR_LOG = LOG_DIR / "error.log"
    WARNING_LOG = LOG_DIR / "warning.log"
