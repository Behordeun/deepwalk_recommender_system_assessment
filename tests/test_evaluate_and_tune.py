"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-23 09:34:29
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:33:42
FilePath: tests/test_evaluate_and_tune.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

# tests/test_evaluate_and_tune.py

import networkx as nx
import pandas as pd
import pytest
from gensim.models import Word2Vec

from src.deepwalk_recommender.evaluate_and_tune import (
    build_graph,
    generate_random_walks,
    train_deepwalk,
)


@pytest.fixture
def sample_df():
    """
    Pytest fixture that returns a sample DataFrame with user-movie ratings for testing purposes.
    """
    return pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "movie_id": [10, 20, 10, 30], "rating": [5, 3, 4, 2]}
    )


def test_build_graph(sample_df):
    """
    Tests the build_graph function to ensure it creates a bipartite graph with correct user and movie nodes and edges from the sample DataFrame.
    """
    graph = build_graph(sample_df)
    assert isinstance(graph, nx.Graph)
    # Should have 2 users and 3 movies as nodes
    assert graph.number_of_nodes() == 5
    # Should have 4 edges (one per row)
    assert graph.number_of_edges() == 4
    assert "u_1" in graph.nodes
    assert "m_10" in graph.nodes
    assert graph.has_edge("u_1", "m_10")


def test_generate_random_walks(sample_df):
    """
    Tests the generate_random_walks function to ensure it produces the correct number and length of walks.

    Asserts that the number of walks equals num_walks times the number of nodes in the graph,
    and that each walk does not exceed the specified walk_length.
    """
    graph = build_graph(sample_df)
    walks = generate_random_walks(graph, num_walks=2, walk_length=5)
    # Should generate num_walks * num_nodes walks
    assert len(walks) == 2 * graph.number_of_nodes()
    # Each walk should have walk_length nodes
    for walk in walks:
        assert len(walk) <= 5  # Can be shorter if dead end


def test_train_deepwalk(sample_df):
    """
    Tests the train_deepwalk function by verifying that it returns a Word2Vec model
    and that all user and movie nodes from the constructed graph are present in the model's vocabulary.
    """
    graph = build_graph(sample_df)
    walks = generate_random_walks(graph, num_walks=1, walk_length=3)
    model = train_deepwalk(
        walks, embedding_size=8, window_size=2, min_count=1, workers=1
    )
    assert isinstance(model, Word2Vec)
    # Check that user and movie nodes are in the vocabulary
    for node in graph.nodes:
        assert str(node) in model.wv
