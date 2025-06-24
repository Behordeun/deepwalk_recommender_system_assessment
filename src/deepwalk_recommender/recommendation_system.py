"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 08:16:08
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 10:54:24
FilePath: src/deepwalk_recommender/recommendation_system.py
Description: This script implements a recommendation system using a DeepWalk-based approach.
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.deepwalk_recommender.schemas import UserInfo


class RecommendationSystem:
    """
    Provides movie recommendations using DeepWalk-based embeddings and user-movie interactions.

    This system combines graph-based embeddings with content-based features to generate
    personalized recommendations. It leverages:
    - User and movie embeddings from DeepWalk
    - Movie genre information
    - Rating popularity statistics

    Attributes:
        model (Word2Vec): Trained DeepWalk model
        df (pd.DataFrame): User-movie ratings data
        user_movie_matrix (pd.DataFrame): Pivoted user-movie interaction matrix
        movies_df (pd.DataFrame): Movie metadata (ID, title, genres, etc.)
        users_df (pd.DataFrame): User metadata (ID, age, gender, occupation, etc.)
    """

    def __init__(self, model_path, data_path, ml_100k_dir):
        """
        Initialize the RecommendationSystem by loading embeddings, ratings, and metadata.

        Args:
            model_path (str or Path): Path to the trained Word2Vec model
            data_path (str or Path): Path to user-movie ratings CSV file
            ml_100k_dir (str or Path): Directory containing MovieLens 100k files
                                       ("u.item" for movies, "u.user" for users)
        """
        self.model = Word2Vec.load(str(model_path))
        self.df = pd.read_csv(str(data_path))
        self.user_movie_matrix = self._create_user_movie_matrix()
        self.movies_df = self._load_movies(str(Path(ml_100k_dir) / "u.item"))
        self.users_df = self._load_users(str(Path(ml_100k_dir) / "u.user"))

    def _create_user_movie_matrix(self):
        """
        Generate the user-movie interaction matrix with ratings.

        Creates a pivot table where:
        - Rows represent users
        - Columns represent movies
        - Values represent ratings (0 for unrated)

        Returns:
            pd.DataFrame: User-movie matrix with dimensions [users x movies]
        """
        return self.df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

    @staticmethod
    def _load_movies(movies_path):
        """
        Load movie metadata from MovieLens 100k format.

        Args:
            movies_path (str or Path): Path to 'u.item' file

        Returns:
            pd.DataFrame: Movie metadata with columns:
                movie_id, title, release_date, video_release_date, imdb_url,
                and 19 genre indicators (Action, Adventure, etc.)
        """
        column_names = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        movies_df = pd.read_csv(
            movies_path,
            sep="|",
            names=column_names,
            encoding="latin-1",
            usecols=range(len(column_names)),  # Ensure correct column alignment
        )
        movies_df["movie_id"] = movies_df["movie_id"].astype(int)
        return movies_df

    @staticmethod
    def _load_users(users_path):
        """
        Load user metadata from MovieLens 100k format.

        Args:
            users_path (str or Path): Path to 'u.user' file

        Returns:
            pd.DataFrame: User metadata with columns:
                user_id, age, gender, occupation, zip_code
        """
        column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
        users_df = pd.read_csv(users_path, sep="|", names=column_names)
        # Ensure 'zip_code' is treated as a string to match the Pydantic schema
        users_df["zip_code"] = users_df["zip_code"].astype(str)
        return users_df

    def get_user_embedding(self, user_id):
        """
        Retrieve DeepWalk embedding for a user.

        Args:
            user_id (int): User ID

        Returns:
            np.ndarray or None: Embedding vector if exists, else None
        """
        user_node = f"u_{user_id}"
        if user_node in self.model.wv:
            return self.model.wv[user_node]
        return None

    def get_movie_embedding(self, movie_id):
        """
        Retrieve DeepWalk embedding for a movie.

        Args:
            movie_id (int): Movie ID

        Returns:
            np.ndarray or None: Embedding vector if exists, else None
        """
        movie_node = f"m_{movie_id}"
        if movie_node in self.model.wv:
            return self.model.wv[movie_node]
        return None

    def predict_rating(self, user_id, movie_id):
        """
        Predict rating a user would give to a movie using embedding similarity.

        Formula:
            rating = 1 + 4 * (cosine_similarity + 1) / 2
            (maps [-1,1] similarity to [1,5] rating scales)

        Args:
            user_id (int): User ID
            movie_id (int): Movie ID

        Returns:
            float: Predicted rating between 1-5, or 3.0 if embeddings missing
        """
        user_embedding = self.get_user_embedding(user_id)
        movie_embedding = self.get_movie_embedding(movie_id)

        if user_embedding is not None and movie_embedding is not None:
            similarity = cosine_similarity([user_embedding], [movie_embedding])[0][0]
            predicted_rating = 1 + 4 * (similarity + 1) / 2
            return max(1, min(5, predicted_rating))
        return 3.0  # Default rating for missing embeddings

    def get_recommendations(self, user_id, top_k=5):
        """
        Generate diverse movie recommendations for a user.

        Recommendation score combines:
        - Predicted rating (60%)
        - Genre match with user's preferences (20%)
        - Movie popularity (20%)

        Ensures diversity by limiting one movie per primary genre.

        Args:
            user_id (int): User ID to generate recommendations for
            top_k (int): Number of recommendations to return

        Returns:
            list[dict]: Recommended movies with:
                similar_movie_id: Movie ID
                title: Movie title
                genres: List of genres
                score: Recommendation score [0-5]
                prob: Confidence percentage [50-99]
                popularity: Popularity category
                explanation: Recommendation rationale
        """
        # Get the movies the user hasn't rated
        user_movies = set(self.df[self.df["user_id"] == user_id]["movie_id"])
        valid_movies = set(self.movies_df["movie_id"])
        unrated_movies = valid_movies - user_movies

        # Determine user's genre preferences from rated movies
        genre_cols = self.movies_df.columns[5:]
        rated_movies = self.movies_df[self.movies_df["movie_id"].isin(user_movies)]
        user_genres = []
        for _, row in rated_movies.iterrows():
            user_genres.extend([g for g in genre_cols if row[g] == 1])
        top_user_genres = set([g for g, _ in Counter(user_genres).most_common(3)])

        # Calculate normalized popularity scores
        rating_counts = self.df["movie_id"].value_counts()
        popularity = rating_counts / rating_counts.max()

        def get_popularity_label(movie_id):
            """Categorize movie based on rating count."""
            count = rating_counts.get(movie_id, 0)
            if count >= 500:
                return "Critically acclaimed"
            elif count >= 100:
                return "Popular"
            elif count > 20:
                return "Rarely rated"
            else:
                return "Hidden gem"

        recommendations = []

        for movie_id in unrated_movies:
            # Calculate predicted rating
            predicted_rating = self.predict_rating(user_id, movie_id)
            confidence = min(99, max(50, int(predicted_rating * 20)))  # 50-99%

            # Get movie metadata
            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id].iloc[0]
            movie_genres = [g for g in genre_cols if movie_info[g] == 1]

            # Calculate genre match score (0-5)
            genre_overlap = len(top_user_genres.intersection(movie_genres)) / max(
                len(movie_genres), 1
            )
            genre_score = 5 * genre_overlap

            # Calculate popularity score (0-5)
            popularity_score = popularity.get(movie_id, 0) * 5

            # Combine scores for final recommendation score
            final_score = (
                0.6 * predicted_rating + 0.2 * genre_score + 0.2 * popularity_score
            )

            # Generate explanation based on similar rated movies
            similar_titles = rated_movies[
                rated_movies[genre_cols].apply(
                    lambda row, genres=movie_genres: any(row[g] == 1 for g in genres),
                    axis=1,
                )
            ]["title"].tolist()
            explanation = (
                f"Recommended because you liked: {', '.join(similar_titles[:2])}"
                if similar_titles
                else "Based on your overall preferences"
            )

            recommendations.append(
                {
                    "similar_movie_id": int(movie_id),
                    "title": movie_info["title"],
                    "genres": movie_genres,
                    "score": round(final_score, 2),
                    "prob": confidence,
                    "popularity": get_popularity_label(movie_id),
                    "explanation": explanation,
                }
            )

        # Sort by score and ensure genre diversity
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        unique_genre_recommendations = {}
        for rec in recommendations:
            primary_genre = rec["genres"][0] if rec["genres"] else "Unknown"
            if primary_genre not in unique_genre_recommendations:
                unique_genre_recommendations[primary_genre] = rec

        return list(unique_genre_recommendations.values())[:top_k]

    def add_interaction(self, user_id, movie_id, rating):
        """
        Add new user-movie interaction to the system.

        Updates both the rating DataFrame and user-movie matrix.

        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            rating (float): Rating (1-5)

        Returns:
            bool: Always returns True (success)
        """
        new_row = pd.DataFrame(
            {"user_id": [user_id], "movie_id": [movie_id], "rating": [rating]}
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.user_movie_matrix = self._create_user_movie_matrix()
        return True

    def get_all_movies(self):
        """
        Retrieve all movies with basic metadata.

        Returns:
            list[dict]: Movies with:
                movie_id: Integer movie ID
                title: Movie title
                release_date: Release date (empty string if missing)
        """
        cleaned_df = self.movies_df[["movie_id", "title", "release_date"]].copy()
        cleaned_df["movie_id"] = cleaned_df["movie_id"].astype(int)
        cleaned_df = cleaned_df.fillna("")
        return cleaned_df.to_dict(orient="records")

    def get_user_info(self, user_id: int) -> Optional[UserInfo]:
        """
        Retrieve metadata for a specific user and return it as a UserInfo Pydantic model.

        Args:
            user_id (int): User ID

        Returns:
            UserInfo or None: User metadata as a Pydantic model if found, else None
        """
        user_data = self.users_df[self.users_df["user_id"] == user_id]
        if not user_data.empty:
            user_dict = user_data.iloc[0].to_dict()
            # Remove 'user_id' from the dictionary before passing to UserInfo,
            # as UserInfo schema is expected to no longer contain 'user_id'.
            if "user_id" in user_dict:
                del user_dict["user_id"]
            return UserInfo(**user_dict)
        return None
