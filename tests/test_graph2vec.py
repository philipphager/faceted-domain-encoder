import torch
from torch import nn

from faceted_domain_encoder.graph2vec import CategoryGraph2Vec
from faceted_domain_encoder.util.linalg import split_categories


def test_category_domain2vec_category_distance():
    embeddings = torch.tensor([
        [
            [0.1, 0.1, 0.1], [1, 2, 3], [3, 2, 1]
        ],
        [
            [10, 10, 10], [1, 2, 3], [3, 3, 3]
        ]
    ])

    mock_layer = nn.Embedding(100, 1600)
    graph2vec = CategoryGraph2Vec(mock_layer, None)
    graph2vec.embeddings = embeddings
    graph2vec.k_neighbours = 20

    distances = graph2vec.get_distance(0, 1)
    assert distances[0] == 0
    assert distances[1] == 0
    assert distances[2] != 0


def test_category_domain2vec_category_distance_to_items():
    embeddings = torch.tensor([
        [
            [0.1, 0.1, 0.1], [1, 2, 3], [3, 2, 1]
        ],
        [
            [10, 10, 10], [1, 2, 3], [3, 3, 3]
        ],
        [
            [10, 10, 10], [1, 2, 3], [0, 0, 0]
        ]
    ])

    mock_layer = nn.Embedding(100, 1600)
    graph2vec = CategoryGraph2Vec(mock_layer, None)
    graph2vec.embeddings = embeddings
    graph2vec.k_neighbours = 20

    distances = graph2vec.get_distance_to_items(0, torch.tensor([1, 2]))
    assert distances[0, 0] == 0
    assert distances[1, 0] == 0
    assert distances[0, 1] == 0
    assert distances[1, 1] == 0
    assert distances[0, 2] != 0
    assert distances[1, 2] == 1


def test_category_domain2vec_category_knns_benchmark(benchmark):
    embeddings = split_categories(torch.randn(10000, 1600), 16)

    mock_layer = nn.Embedding(100, 1600)
    graph2vec = CategoryGraph2Vec(mock_layer, None)
    graph2vec.embeddings = embeddings
    graph2vec.k_neighbours = 20

    neighbours = benchmark(graph2vec.get_knns, 123)
    assert neighbours.size(0) == 20


def test_category_domain2vec_category_knns():
    embeddings = torch.tensor([
        [
            [0.1, 0.1, 0.1], [1, 2, 3], [3, 2, 1]
        ],
        [
            [10, 10, 10], [1, 2, 3], [3, 3, 3]
        ],
        [
            [10, 10, 10], [1, 2, 3], [0, 0, 0]
        ]
    ])

    mock_layer = nn.Embedding(100, 1600)
    graph2vec = CategoryGraph2Vec(mock_layer, None)
    graph2vec.embeddings = embeddings
    graph2vec.k_neighbours = 20

    neighbours = graph2vec.get_knns(0)
    assert torch.equal(neighbours, torch.tensor([1, 2]))
