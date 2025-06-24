"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:28:50
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 03:47:39
FilePath: src/deepwalk_recommender/evaluate_and_tune.py
Description: This script evaluates and tunes the DeepWalk model for movie recommendations.
"""

import itertools
import json
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.deepwalk_recommender.config import PathConfig
from src.deepwalk_recommender.error_logger import system_logger
from src.deepwalk_recommender.schemas import EvaluationMetrics


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a bipartite graph from user-movie interactions.

    Creates an undirected graph where:
    - User nodes: "u_{user_id}"
    - Movie nodes: "m_{movie_id}"
    - Edges: Connect users to movies they've rated

    Args:
        df (pd.DataFrame): Interaction data with columns:
            user_id: Integer user identifiers
            movie_id: Integer movie identifiers

    Returns:
        nx.Graph: Bipartite graph of user-movie interactions
    """
    graph = nx.Graph()
    for _, row in df.iterrows():
        user_node = f"u_{row['user_id']}"
        movie_node = f"m_{row['movie_id']}"
        graph.add_node(user_node, bipartite=0)
        graph.add_node(movie_node, bipartite=1)
        graph.add_edge(user_node, movie_node)
    return graph


def generate_random_walks(
    graph: nx.Graph, num_walks: int = 10, walk_length: int = 80
) -> list[list[str]]:
    """
    Generates random walks for DeepWalk training.

    For each node in the graph, starts 'num_walks' random walks of length 'walk_length'.

    Args:
        graph (nx.Graph): Input graph from build_graph()
        num_walks (int): Number of walks per node (default: 10)
        walk_length (int): Length of each walk in nodes (default: 80)

    Returns:
        list[list[str]]: List of walks, each walk is a list of node IDs
    """
    walks = []
    nodes = list(graph.nodes)
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            current_node = node
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current_node))
                if neighbors:
                    current_node = random.choice(neighbors)
                    walk.append(str(current_node))
                else:
                    break
            walks.append(walk)
    return walks


def train_deepwalk(
    walks: list[list[str]],
    embedding_size: int = 128,
    window_size: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 5,
) -> Word2Vec:
    """
    Trains Word2Vec model on generated random walks.

    Args:
        walks (list[list[str]]): Random walks from generate_random_walks()
        embedding_size (int): Dimension of embedding vectors (default: 128)
        window_size (int): Context window size for Skip-gram (default: 5)
        min_count (int): Ignore nodes with frequency < min_count (default: 1)
        workers (int): Parallel worker threads (default: 4)
        epochs (int): Training iterations (default: 5)

    Returns:
        Word2Vec: Trained model containing node embeddings
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # Use Skip-gram
    )
    return model


def generate_negative_samples(
    df: pd.DataFrame, positive_interactions: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """
    Generates negative samples for evaluation dataset.

    Creates balanced negative samples that don't exist in the original dataset.

    Args:
        df (pd.DataFrame): Full interaction dataset
        positive_interactions (pd.DataFrame): Positive interactions (rating >= 3.5)
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Negative samples with rating=0
    """
    all_users = df["user_id"].unique()
    all_movies = df["movie_id"].unique()
    existing_interactions = set(tuple(x) for x in df[["user_id", "movie_id"]].values)

    rng = np.random.default_rng(seed)
    negative_samples = []
    sample_count = len(positive_interactions)

    # Efficient negative sampling
    attempts = 0
    max_attempts = sample_count * 5  # Prevent infinite loops

    while len(negative_samples) < sample_count and attempts < max_attempts:
        user = rng.choice(all_users)
        movie = rng.choice(all_movies)
        if (user, movie) not in existing_interactions:
            negative_samples.append({"user_id": user, "movie_id": movie, "rating": 0})
        attempts += 1

    return pd.DataFrame(negative_samples)


def evaluate_embeddings(model: Word2Vec, df: pd.DataFrame) -> tuple:
    """
    Evaluates DeepWalk embeddings using logistic regression classifier.

    Creates a balanced dataset with positive interactions and negative samples.
    Trains classifier on concatenated user-movie embeddings.

    Args:
        model (Word2Vec): Trained DeepWalk model
        df (pd.DataFrame): Full interaction dataset

    Returns:
        tuple: (accuracy, precision, recall, f1) scores
    """
    # Create balanced dataset
    positive_interactions = df[df["rating"] >= 3.5].copy()
    negative_samples = generate_negative_samples(df, positive_interactions)

    # Combine positive and negative samples
    combined_df = pd.concat(
        [positive_interactions, negative_samples], ignore_index=True
    )
    combined_df["label"] = (combined_df["rating"] >= 3.5).astype(int)

    # Prepare feature vectors
    features = []
    labels = []
    missing_embeddings = 0

    for _, row in combined_df.iterrows():
        user_node = f"u_{row['user_id']}"
        movie_node = f"m_{row['movie_id']}"

        if user_node in model.wv and movie_node in model.wv:
            user_embedding = model.wv[user_node]
            movie_embedding = model.wv[movie_node]
            features.append(np.concatenate((user_embedding, movie_embedding)))
            labels.append(row["label"])
        else:
            missing_embeddings += 1

    if missing_embeddings:
        system_logger.warning(
            f"Skipped {missing_embeddings} interactions due to missing embeddings"
        )

    if not features:
        system_logger.error("No valid embeddings found for evaluation")
        return 0, 0, 0, 0

    # Split and train classifier
    features = np.array(features)
    labels = np.array(labels)

    # Add this check before train_test_split
    if len(np.unique(labels)) < 2:
        system_logger.warning(
            "Only one class present in evaluation labels; cannot train classifier."
        )
        return 0, 0, 0, 0

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    classifier = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
    classifier.fit(train_features, train_labels)
    predicted_labels = classifier.predict(test_features)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    return accuracy, precision, recall, f1


def run_tuning_pipeline():
    """DeepWalk hyperparameter tuning pipeline"""
    # Load processed interaction data
    df = pd.read_csv(PathConfig.PROCESSED_DATA_FILE)
    system_logger.info(
        f"Loaded {len(df):,} interactions between "
        f"{df['user_id'].nunique():,} users and "
        f"{df['movie_id'].nunique():,} movies"
    )

    # Build interaction graph
    graph = build_graph(df)
    system_logger.info(
        f"Graph built with {graph.number_of_nodes():,} nodes "
        f"and {graph.number_of_edges():,} edges"
    )

    # Define DeepWalk hyperparameter grid
    param_grid = {
        "embedding_size": [64, 128],
        "window_size": [5, 10],
        "num_walks": [5, 10],
        "walk_length": [40, 80],
        "epochs": [5, 10],
    }

    # Track best model and metrics
    best_model = None
    best_f1 = -1
    best_metrics = {}
    best_params = {}

    # Iterate over hyperparameter combinations
    keys = param_grid.keys()
    total_combinations = len(list(itertools.product(*param_grid.values())))
    current_iteration = 0

    for values in itertools.product(*param_grid.values()):
        current_iteration += 1
        params = dict(zip(keys, values))
        system_logger.info(
            f"Training DeepWalk ({current_iteration}/{total_combinations}) "
            f"with params: {params}"
        )

        # Generate random walks
        walks = generate_random_walks(
            graph, num_walks=params["num_walks"], walk_length=params["walk_length"]
        )
        system_logger.info(f"Generated {len(walks):,} random walks")

        # Train DeepWalk model
        model = train_deepwalk(
            walks,
            embedding_size=params["embedding_size"],
            window_size=params["window_size"],
            epochs=params["epochs"],
        )

        # Evaluate embeddings
        accuracy, precision, recall, f1 = evaluate_embeddings(model, df)
        system_logger.info(
            f"Evaluation metrics: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            system_logger.info("New best model found!")

    # Save best model and results
    if best_model:
        # Ensure models directory exists
        PathConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Save model
        best_model.save(str(PathConfig.DEEPWALK_MODEL_PATH))
        system_logger.info(
            f"Best DeepWalk model saved to: {PathConfig.DEEPWALK_MODEL_PATH}"
        )

        # Prepare results dictionary
        results = {
            "hyperparameters": best_params,
            "metrics": best_metrics,
            "dataset_stats": {
                "interactions": len(df),
                "users": df["user_id"].nunique(),
                "movies": df["movie_id"].nunique(),
            },
        }

        # Save results to JSON
        with open(PathConfig.DEEPWALK_BEST_PARAMS_PATH, "w") as f:
            json.dump(results, f, indent=4)
        system_logger.info(
            f"Best parameters and metrics saved to: "
            f"{PathConfig.DEEPWALK_BEST_PARAMS_PATH}"
        )

        # Print summary
        print("\n=== Tuning Results Summary ===")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"Model saved to: {PathConfig.DEEPWALK_MODEL_PATH}")
        print(f"Parameters saved to: {PathConfig.DEEPWALK_BEST_PARAMS_PATH}")


if __name__ == "__main__":
    try:
        run_tuning_pipeline()
    except Exception:
        system_logger.exception("Hyperparameter tuning failed")
        raise
