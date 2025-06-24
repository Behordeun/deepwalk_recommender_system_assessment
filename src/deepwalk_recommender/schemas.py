"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:33:28
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 04:18:06
FilePath: src/deepwalk_recommender/schemas.py
Description: This module defines Pydantic models for request and response schemas used in the FastAPI application for movie recommendations.
"""

from typing import List

from pydantic import BaseModel


class UserInfo(BaseModel):
    age: int
    gender: str
    occupation: str
    zip_code: str


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
    user_info: UserInfo
    recommended_items: List[RecommendedItem]
