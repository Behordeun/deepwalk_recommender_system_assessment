"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:32:06
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:23:29
FilePath: src/deepwalk_recommender/main.py
Description: This script sets up a FastAPI application for movie recommendations using a DeepWalk-based recommendation system.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from src.deepwalk_recommender.config import PathConfig
from src.deepwalk_recommender.errorlogger import log_error, system_logger
from src.deepwalk_recommender.recommendation_system import RecommendationSystem
from src.deepwalk_recommender.schemas import InteractionRequest, RecommendationResponse

# Path configs
ML_100K_DIR = PathConfig.ML_100K_DIR
MODEL_PATH = PathConfig.DEEPWALK_MODEL_PATH
PROCESSED_DATA_PATH = PathConfig.PROCESSED_DATA_FILE

# Global container for the rec system instance
rec_system: RecommendationSystem | None = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI application lifespan.

    Initializes the RecommendationSystem on startup and logs the process. Handles initialization errors by logging them and re-raising the exception. Logs shutdown message when the application is stopping.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """
    global rec_system
    try:
        system_logger.info("Initializing RecommendationSystem...")
        rec_system = RecommendationSystem(
            model_path=MODEL_PATH,
            data_path=PROCESSED_DATA_PATH,
            ml_100k_dir=ML_100K_DIR,
        )
        system_logger.info("RecommendationSystem initialized successfully.")
    except Exception as e:
        log_error(
            "Failed to initialize RecommendationSystem during startup", error=str(e)
        )
        raise

    yield  # App is running

    system_logger.info("Movie Recommendation API is shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Movie Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message and the API version.

    Returns:
        dict: A message and version information for the Movie Recommendation API.
    """
    return {"message": "Movie Recommendation API", "version": "1.0.0"}


@app.post("/interactions")
async def add_interaction(interaction: InteractionRequest):
    """
    Handle POST requests to add a new user-movie interaction.

    Args:
        interaction (InteractionRequest): The interaction data containing user_id, movie_id, and rating.

    Returns:
        dict: Success message if the interaction is added.

    Raises:
        HTTPException: If adding the interaction fails or an unexpected error occurs.
    """
    try:
        success = rec_system.add_interaction(
            interaction.user_id, interaction.movie_id, interaction.rating
        )
        if success:
            return {"message": "Interaction added successfully"}
        else:
            log_error(
                "Failed to add interaction",
                user_id=interaction.user_id,
                movie_id=interaction.movie_id,
            )
            raise HTTPException(status_code=400, detail="Failed to add interaction")
    except HTTPException:
        raise
    except Exception as e:
        log_error(
            "Unexpected error in add_interaction",
            error=str(e),
            user_id=interaction.user_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int):
    """
    Handle GET requests for movie recommendations for a specified user.

    Args:
        user_id (int): The user ID for whom to generate recommendations.

    Returns:
        RecommendationResponse: Recommended items and user information for the given user.

    Raises:
        HTTPException: If the user is not found or an internal error occurs.
    """
    try:
        recommendations = rec_system.get_recommendations(user_id, top_k=5)
        user_info = rec_system.get_user_info(user_id)
        if user_info is None:
            raise HTTPException(status_code=404, detail="User not found")
        user_info.pop("user_id", None)
        return RecommendationResponse(
            user_id=user_id, user_info=user_info, recommended_items=recommendations
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error(
            "Error during recommendation generation", error=str(e), user_id=user_id
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items")
async def get_all_items():
    """
    Retrieve all movies from the recommendation system.

    Returns a JSON response containing a list of movies and their count.
    Logs and raises an HTTPException with status 500 if retrieval fails.
    """
    try:
        movies = rec_system.get_all_movies()
        return jsonable_encoder({"movies": movies, "count": len(movies)})
    except Exception as e:
        log_error("Failed to retrieve items", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    system_logger.info("Launching Movie Recommendation API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
