"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:33:28
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 04:05:25
FilePath: src/deepwalk_recommender/schemas.py
Description: This module defines Pydantic models for request and response schemas used in the FastAPI application for movie recommendations.
"""

from typing import Any, Dict, List

from pydantic import BaseModel


# Pydantic models for request/response
class InteractionRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class RecommendedItem(BaseModel):
    similar_movie_id: int
    title: str
    genres: List[str]
    score: float
    prob: int
    explanation: str
    popularity: str


class RecommendationResponse(BaseModel):
    user_id: int
    user_info: Dict[str, Any]
    recommended_items: List[RecommendedItem]
