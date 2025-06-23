"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 08:16:08
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:23:29
FilePath: src/deepwalk_recommender/recommendation_system.py
Description: This script implements a recommendation system using a DeepWalk-based approach.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationSystem:
    """
    RecommendationSystem provides movie recommendations using DeepWalk-based embeddings and user-movie interaction data.
    It supports loading MovieLens 100k data, predicting ratings, generating diverse recommendations, adding new interactions, and retrieving user or movie information.
    """
    def __init__(self, model_path, data_path, ml_100k_dir):
        """
        Initialize the RecommendationSystem by loading the DeepWalk model, user-movie interaction data, and MovieLens 100k user and movie metadata.

        Args:
            model_path (str or Path): Path to the trained Word2Vec model.
            data_path (str or Path): Path to the user-movie ratings CSV file.
            ml_100k_dir (str or Path): Directory containing MovieLens 100k files ("u.item" and "u.user").
        """
        self.model = Word2Vec.load(str(model_path))
        self.df = pd.read_csv(str(data_path))
        self.user_movie_matrix = self._create_user_movie_matrix()
        self.movies_df = self._load_movies(str(Path(ml_100k_dir) / "u.item"))
        self.users_df = self._load_users(str(Path(ml_100k_dir) / "u.user"))

    def _create_user_movie_matrix(self):
        """
        Generate a user-movie interaction matrix with users as rows, movies as columns, and ratings as values. Unrated entries are filled with 0.
        """
        return self.df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

    @staticmethod
    def _load_movies(movies_path):
        """
        Load the MovieLens 100k movie metadata from the specified file path.

        Args:
            movies_path (str or Path): Path to the MovieLens 'u.item' file.

        Returns:
            pd.DataFrame: DataFrame containing movie information with columns for movie ID, title, release dates, genres, and other metadata.
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
            movies_path, sep="|", names=column_names, encoding="latin-1"
        )
        return movies_df

    @staticmethod
    def _load_users(users_path):
        """
        Load the MovieLens 100k user metadata from the specified file path.

        Args:
            users_path (str or Path): Path to the MovieLens 'u.user' file.

        Returns:
            pd.DataFrame: DataFrame containing user information with columns for user ID, age, gender, occupation, and zip code.
        """
        column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
        users_df = pd.read_csv(users_path, sep="|", names=column_names)
        return users_df

    def get_user_embedding(self, user_id):
        """
        Return the DeepWalk embedding vector for the specified user.

        Args:
            user_id (int): The user ID.

        Returns:
            np.ndarray or None: The embedding vector if available, otherwise None.
        """
        user_node = f"u_{user_id}"
        if user_node in self.model.wv:
            return self.model.wv[user_node]
        return None

    def get_movie_embedding(self, movie_id):
        """
        Return the DeepWalk embedding vector for the specified movie.

        Args:
            movie_id (int): The movie ID.

        Returns:
            np.ndarray or None: The embedding vector if available, otherwise None.
        """
        movie_node = f"m_{movie_id}"
        if movie_node in self.model.wv:
            return self.model.wv[movie_node]
        return None

    def predict_rating(self, user_id, movie_id):
        """
        Predict the rating a user would give to a movie using cosine similarity between their DeepWalk embeddings.

        Args:
            user_id (int): The user ID.
            movie_id (int): The movie ID.

        Returns:
            float: Predicted rating in the range [1, 5], or 3.0 if embeddings are unavailable.
        """
        user_embedding = self.get_user_embedding(user_id)
        movie_embedding = self.get_movie_embedding(movie_id)

        if user_embedding is not None and movie_embedding is not None:
            similarity = cosine_similarity([user_embedding], [movie_embedding])[0][0]
            # Fixed formula: Map [-1, 1] to [1, 5]
            predicted_rating = 1 + 4 * (similarity + 1) / 2
            return max(1, min(5, predicted_rating))
        return 3.0  # Default rating if embeddings not found

    def get_recommendations(self, user_id, top_k=5):
        """
        Generate a diverse list of top-k movie recommendations for a user based on predicted ratings, genre preferences, and movie popularity.

        Args:
            user_id (int): The user ID for whom to generate recommendations.
            top_k (int, optional): Number of recommendations to return. Defaults to 5.

        Returns:
            list of dict: Recommended movies with details including title, genres, score, confidence, popularity label, and explanation.
        """
        user_movies = set(self.df[self.df["user_id"] == user_id]["movie_id"])
        valid_movies = set(self.movies_df["movie_id"])
        unrated_movies = valid_movies - user_movies

        # Genre preferences from rated movies
        genre_cols = self.movies_df.columns[5:]
        rated_movies = self.movies_df[self.movies_df["movie_id"].isin(user_movies)]
        user_genres = []
        for _, row in rated_movies.iterrows():
            user_genres.extend([g for g in genre_cols if row[g] == 1])
        top_user_genres = set([g for g, _ in Counter(user_genres).most_common(3)])

        # Popularity mapping (normalize by total count)
        rating_counts = self.df["movie_id"].value_counts()
        popularity = rating_counts / rating_counts.max()

        def get_popularity_label(movie_id):
            """
            Return a popularity label for a movie based on its rating count.

            Args:
                movie_id (int): The movie ID.

            Returns:
                str: Popularity label such as "Critically acclaimed", "Popular", "Rarely rated", or "Hidden gem".
            """
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
            predicted_rating = self.predict_rating(user_id, movie_id)
            confidence = min(99, max(50, int(predicted_rating * 20)))

            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id].iloc[0]
            movie_genres = [g for g in genre_cols if movie_info[g] == 1]

            # Genre overlap score (0–5)
            genre_overlap = len(top_user_genres.intersection(movie_genres)) / max(
                len(movie_genres), 1
            )
            genre_score = 5 * genre_overlap

            # Popularity score (0–5)
            popularity_score = popularity.get(movie_id, 0) * 5

            # Final score
            final_score = (
                0.6 * predicted_rating + 0.2 * genre_score + 0.2 * popularity_score
            )

            # Explanation: similar titles by genre
            similar_titles = rated_movies[
                rated_movies[genre_cols].apply(
                    lambda row, genres=movie_genres: any(row[g] == 1 for g in genres),
                    axis=1,
                )
            ]["title"].tolist()
            explanation = (
                f"Recommended because you liked: {', '.join(similar_titles[:2])}"
                if similar_titles
                else "Similar to your preferences"
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

        # Sort all by score, then enforce genre diversity
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        unique_genre_recommendations = {}
        for rec in recommendations:
            primary_genre = rec["genres"][0] if rec["genres"] else "Unknown"
            if primary_genre not in unique_genre_recommendations:
                unique_genre_recommendations[primary_genre] = rec

        # Final top-k with diversity
        diverse_top_k = list(unique_genre_recommendations.values())[:top_k]
        return diverse_top_k

    def add_interaction(self, user_id, movie_id, rating):
        """
        Add a new user-movie interaction to the dataset and update the user-movie interaction matrix.

        Args:
            user_id (int): The user ID.
            movie_id (int): The movie ID.
            rating (float or int): The rating given by the user.

        Returns:
            bool: True if the interaction was added successfully.
        """
        new_row = pd.DataFrame(
            {"user_id": [user_id], "movie_id": [movie_id], "rating": [rating]}
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.user_movie_matrix = self._create_user_movie_matrix()
        return True

    def get_all_movies(self):
        """
        Return a list of all movies in the dataset with their IDs, titles, and release dates, replacing any missing values with empty strings. Logs the DataFrame type and columns for debugging.
        """
        from src.deepwalk_recommender.errorlogger import system_logger

        system_logger.info(f"Movies DF Type: {type(self.movies_df)}")
        system_logger.info(f"Movies DF Columns: {self.movies_df.columns.tolist()}")

        cleaned_df = self.movies_df[["movie_id", "title", "release_date"]].fillna("")
        return cleaned_df.to_dict(orient="records")

    def get_user_info(self, user_id):
        """
        Retrieve user metadata for the specified user ID.

        Args:
            user_id (int): The user ID.

        Returns:
            dict or None: Dictionary of user information if found, otherwise None.
        """
        user_info = self.users_df[self.users_df["user_id"] == user_id]
        if not user_info.empty:
            return user_info.iloc[0].to_dict()
        return None
