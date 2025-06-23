"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:28:50
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:23:29
FilePath: src/deepwalk_recommender/evaluate_and_tune.py
Description: This script evaluates and tunes the DeepWalk model for movie recommendations.
"""

import itertools
import json
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.deepwalk_recommender.config import PathConfig


def build_graph(df):
    """
    Constructs a bipartite graph from a DataFrame of user-movie interactions.

    Each user and movie is represented as a node, with edges connecting users to movies they have interacted with.

    Args:
        df (pd.DataFrame): DataFrame containing 'user_id' and 'movie_id' columns.

    Returns:
        nx.Graph: An undirected graph with user and movie nodes connected by edges.
    """
    graph = nx.Graph()
    for _, row in df.iterrows():
        user = f"u_{row['user_id']}"
        movie = f"m_{row['movie_id']}"
        graph.add_edge(user, movie)
    return graph


def generate_random_walks(graph, num_walks=10, walk_length=80):
    """
    Generates random walks from each node in the given graph.

    Args:
        graph (networkx.Graph): The input graph.
        num_walks (int, optional): Number of walks to start from each node. Defaults to 10.
        walk_length (int, optional): Length of each walk. Defaults to 80.

    Returns:
        list: A list of walks, where each walk is a list of node IDs as strings.
    """
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(node))
                if len(neighbors) > 0:
                    node = random.choice(neighbors)
                    walk.append(str(node))
                else:
                    break
            walks.append(walk)
    return walks


def train_deepwalk(walks, embedding_size=128, window_size=5, min_count=1, workers=4):
    """
    Trains a DeepWalk model using Word2Vec on the provided random walks.

    Args:
        walks (list of list of str): Sequences of node IDs representing random walks.
        embedding_size (int, optional): Dimensionality of the embedding vectors. Defaults to 128.
        window_size (int, optional): Maximum distance between the current and predicted node. Defaults to 5.
        min_count (int, optional): Ignores nodes with total frequency lower than this. Defaults to 1.
        workers (int, optional): Number of worker threads to train the model. Defaults to 4.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model containing node embeddings.
    """
    model = Word2Vec(
        walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
    )
    return model


# Load processed data
df = pd.read_csv(PathConfig.PROCESSED_DATA_FILE)

# Build graph once
graph = build_graph(df)
print(
    f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
)

# Define DeepWalk hyperparameter grid
param_grid_deepwalk = {
    "embedding_size": [64, 128],
    "window_size": [5, 10],
    "num_walks": [5, 10],
    "walk_length": [40, 80],
}

best_deepwalk_model = None
best_accuracy_score = -1
best_recall_score = -1
best_precision_score = -1
best_f1_score = -1
best_params = {}

# Iterate over all combinations of DeepWalk hyperparameters
keys = param_grid_deepwalk.keys()
for values in itertools.product(*param_grid_deepwalk.values()):
    params = dict(zip(keys, values))
    print(f"\nTraining DeepWalk with parameters: {params}")

    # Generate random walks with current parameters
    walks = generate_random_walks(
        graph, num_walks=params["num_walks"], walk_length=params["walk_length"]
    )
    print(f"Generated {len(walks)} random walks.")

    # Train the DeepWalk model with current parameters
    current_deepwalk_model = train_deepwalk(
        walks,
        embedding_size=params["embedding_size"],
        window_size=params["window_size"],
    )

    # Prepare data for evaluation (same as before)
    positive_interactions = df[df["rating"] >= 3.5]
    all_users = df["user_id"].unique()
    all_movies = df["movie_id"].unique()
    existing_interactions = set(tuple(x) for x in df[["user_id", "movie_id"]].values)

    SEED = 42
    rng = np.random.default_rng(
        SEED
    )  # Initialize a random number generator for reproducibility

    negative_samples = []
    # Use numpy's random choice for more efficient sampling of users and movies
    # Also, ensure 'user' and 'movie' are actual values from the arrays
    while len(negative_samples) < len(positive_interactions):
        user = rng.choice(all_users)
        movie = rng.choice(all_movies)
        if (user, movie) not in existing_interactions:
            negative_samples.append({"user_id": user, "movie_id": movie, "rating": 0})

    combined_df = pd.concat(
        [positive_interactions, pd.DataFrame(negative_samples)], ignore_index=True
    )

    X = []
    y = []

    for index, row in combined_df.iterrows():
        user_node = f"u_{row['user_id']}"
        movie_node = f"m_{row['movie_id']}"

        if (
            user_node in current_deepwalk_model.wv
            and movie_node in current_deepwalk_model.wv
        ):
            user_embedding = current_deepwalk_model.wv[user_node]
            movie_embedding = current_deepwalk_model.wv[movie_node]
            X.append(np.concatenate((user_embedding, movie_embedding)))
            y.append(1 if row["rating"] >= 3.5 else 0)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate Logistic Regression classifier
    classifier = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred)
    current_precision_score = precision_score(y_test, y_pred)
    current_recall_score = recall_score(y_test, y_pred)
    current_f1_score = f1_score(y_test, y_pred)

    print(f"F1-Score for current DeepWalk model: {current_f1_score:.4f}")

    # Check if the current model is better
    if current_f1_score > best_f1_score:
        best_accuracy_score = current_accuracy
        best_recall_score = current_recall_score
        best_precision_score = current_precision_score
        best_f1_score = current_f1_score

        best_deepwalk_model = current_deepwalk_model
        best_params = params

print("\n--- DeepWalk Hyperparameter Tuning Results ---")
print(f"Best DeepWalk Parameters: {best_params}")
print(f"Best Accuracy-Score achieved: {best_accuracy_score:.4f}")
print(f"Best Recall-Score achieved: {best_recall_score:.4f}")
print(f"Best Precision-Score achieved: {best_precision_score:.4f}")
print(f"Best F1-Score achieved: {best_f1_score:.4f}")

# Ensure the 'models' directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the best DeepWalk model
if best_deepwalk_model:
    best_deepwalk_model.save(str(PathConfig.DEEPWALK_MODEL_PATH))
    print("Best DeepWalk model saved as %", PathConfig.DEEPWALK_BEST_MODEL_NAME)

# Prepare a dict containing both params and metrics
best_results = {
    "hyperparameters": best_params,
    "metrics": {
        "accuracy_score": best_accuracy_score,
        "precision_score": best_precision_score,
        "recall_score": best_recall_score,
        "f1_score": best_f1_score,
    },
}

# Save the best_results to a JSON file
best_params_path = PathConfig.DEEPWALK_BEST_PARAMS_PATH
with open(best_params_path, "w") as f:
    json.dump(best_results, f, indent=4)
print("Best parameters and metrics saved to %", best_params_path)
