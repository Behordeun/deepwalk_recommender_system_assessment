"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 21:30:48
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 02:26:20
FilePath: src/deepwalk_recommender/deepwalk_model.py
Description: This script builds a graph from user-movie interactions, generates random walks, and trains a DeepWalk model for movie recommendations.
"""

import random
from pathlib import Path

import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from src.deepwalk_recommender.config import PathConfig
from src.deepwalk_recommender.error_logger import system_logger


class DeepWalk:
    """
    Interface for working with pre-trained DeepWalk embeddings.

    Provides methods to load a trained Word2Vec model and retrieve node embeddings.

    Attributes:
        model (Word2Vec): Pre-trained Word2Vec model containing node embeddings

    Args:
        model_path (str | Path): Path to the saved Word2Vec model
    """

    def __init__(self, model_path: str | Path):
        """
        Loads pre-trained DeepWalk embeddings from the specified file.

        Args:
            model_path (str | Path): Path to the saved Word2Vec model file
        """
        self.model = Word2Vec.load(str(model_path))

    def get_embedding(self, node_id: str | int):
        """
        Retrieves embedding vector for the specified node.

        Args:
            node_id (str | int): Node identifier (user or movie)
                Format: "u_{user_id}" for users, "m_{movie_id}" for movies

        Returns:
            list or None: Embedding vector as a list if the node exists, else None
        """
        node_str = str(node_id)
        if node_str in self.model.wv:
            return self.model.wv[node_str].tolist()
        return None


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constructs bipartite graph from user-movie interaction data.

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

    Example:
        Input DataFrame:
            user_id | movie_id | rating
            --------|----------|-------
            1 | 101 | 5
            1 | 202 | 4

        Graph will contain:
            Nodes: ['u_1', 'm_101', 'm_202']
            Edges: [('u_1', 'm_101'), ('u_1', 'm_202')]
    """
    graph = nx.Graph()

    # Add nodes and edges
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
    Each walk is a sequence of node IDs representing a path through the graph.

    Args:
        graph (nx.Graph): Input graph from build_graph()
        num_walks (int): Number of walks per node (default: 10)
        walk_length (int): Length of each walk in nodes (default: 80)

    Returns:
        list[list[str]]: List of walks, each walk is a list of node IDs

    Process:
        1. Collect all nodes from the graph
        2. Shuffle nodes
        3. For each node:
            - Start 'num_walks' walks
            - At each step, randomly select a neighbor
            - Continue until walk length reached
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

    Applies the Skip-gram algorithm to learn node embeddings from walk sequences.

    Args:
        walks (list[list[str]]): Random walks from generate_random_walks()
        embedding_size (int): Dimension of embedding vectors (default: 128)
        window_size (int): Context window size for the Skip gram (default: 5)
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
        sg=1,  # Use Skip-gram (1) instead of CBOW (0)
    )
    return model


if __name__ == "__main__":
    """DeepWalk training pipeline execution"""
    try:
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

        # Generate random walks
        walks = generate_random_walks(graph)
        system_logger.info(f"Generated {len(walks):,} random walks")

        # Train DeepWalk model
        model = train_deepwalk(walks)
        model.save(str(PathConfig.DEEPWALK_MODEL_PATH))
        system_logger.info(
            f"DeepWalk model trained and saved: {PathConfig.DEEPWALK_MODEL_PATH}\n"
            f"Embedding size: {model.vector_size}, Vocabulary size: {len(model.wv)}"
        )

    except Exception as e:
        system_logger.error(
            str(e), additional_info={"context": "DeepWalk training pipeline failed"}
        )
        raise
