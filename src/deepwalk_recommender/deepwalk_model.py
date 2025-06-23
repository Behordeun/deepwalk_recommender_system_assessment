"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:30:48
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:05:04
FilePath: src/deepwalk_recommender/deepwalk_model.py
Description: This script builds a graph from user-movie interactions, generates random walks, and trains a DeepWalk model for movie recommendations.
"""

import random

import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from src.deepwalk_recommender.config import PathConfig


class DeepWalk:
    """
    DeepWalk class for loading a pre-trained Word2Vec model and retrieving node embeddings.

    Args:
        model_path (str): Path to the saved Word2Vec model.

    Methods:
        get_embedding(node_id): Returns the embedding vector for the specified node ID as a list, or None if the node is not present in the model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the DeepWalk instance by loading a pre-trained Word2Vec model from the specified file path.

        Args:
            model_path (str): Path to the saved Word2Vec model file.
        """
        self.model = Word2Vec.load(model_path)

    def get_embedding(self, node_id: str):
        """
        Retrieve the embedding vector for the specified node ID.

        Args:
            node_id (str): The ID of the node to retrieve the embedding for.

        Returns:
            list or None: The embedding vector as a list if the node exists in the model, otherwise None.
        """
        if str(node_id) in self.model.wv:
            return self.model.wv[str(node_id)].tolist()
        return None


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
    Trains a Word2Vec model on the provided DeepWalk walks.

    Args:
        walks (list of list of str): Sequences of node identifiers generated from random walks.
        embedding_size (int, optional): Dimensionality of the embedding vectors. Defaults to 128.
        window_size (int, optional): Maximum distance between the current and predicted word within a sentence. Defaults to 5.
        min_count (int, optional): Ignores all words with total frequency lower than this. Defaults to 1.
        workers (int, optional): Number of worker threads to train the model. Defaults to 4.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(
        walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
    )
    return model


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(PathConfig.PROCESSED_DATA_FILE)

    # Build graph
    graph = build_graph(df)
    print(
        f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
    )

    # Generate random walks
    walks = generate_random_walks(graph)
    print(f"Generated {len(walks)} random walks.")

    # Train DeepWalk model
    model = train_deepwalk(walks)
    model.save(PathConfig.DEEPWALK_MODEL_PATH)
    print("DeepWalk model trained and saved as %", PathConfig.DEEPWALK_BEST_MODEL_NAME)
