import torch

from faceted_domain_encoder.domain2vec import CategoryDomain2Vec
from faceted_domain_encoder.util.linalg import CategoryDistance, knns


def test_category_domain2vec_category_distance():
    embeddings = torch.tensor([
        [10, 20, 10, 20, 0, 0],
        [0.1, 0.2, 49, -1, 20, 20],
        [0.4, 0.2, 10, 20, 10, -20]
    ])

    category_distance = CategoryDistance(3, 6, torch.tensor([0.33, 0.33, 0.33]))
    domain2vec = CategoryDomain2Vec(None, category_distance)
    domain2vec.embeddings = embeddings

    distances = domain2vec.get_distance(0, 1)
    assert distances[0] == 0
    assert distances[2] == 1


def test_category_domain2vec_category_distance_to_items():
    embeddings = torch.tensor([
        [10, 20, 10, 20, 0, 0],
        [0.1, 0.2, 49, -1, 20, 20],
        [0.4, 0.2, 10, 20, 10, -20]
    ])

    category_distance = CategoryDistance(3, 6, torch.tensor([0.33, 0.33, 0.33]))
    domain2vec = CategoryDomain2Vec(None, category_distance)
    domain2vec.embeddings = embeddings

    distances = domain2vec.get_distance_to_items(0, torch.tensor([1, 2]))
    assert distances[0, 0] == 0
    assert distances[0, 2] == 1
    assert distances[1, 1] == 0
    assert distances[1, 2] == 1


def test_category_domain2vec_knns():
    embeddings = torch.tensor([
        [10, 20, 10, 20, 0, 0],
        [0.1, 0.2, 49, -1, 20, 20],
        [0.4, 0.2, 10, 20, 10, -20]
    ])

    category_distance = CategoryDistance(3, 6, torch.tensor([0.33, 0.33, 0.33]))
    domain2vec = CategoryDomain2Vec(None, category_distance)
    domain2vec.embeddings = embeddings
    domain2vec.k_neighbours = 2
    domain2vec.nearest_neighbours, _ = knns(embeddings, 2)

    neighbours = domain2vec.get_knns(2)
    assert torch.equal(neighbours, torch.tensor([0, 1]))


def test_category_domain2vec_category_knns_benchmark(benchmark):
    embeddings = torch.randn(10000, 512)

    category_distance = CategoryDistance(3, 6, torch.tensor([0.33, 0.33, 0.33]))
    domain2vec = CategoryDomain2Vec(None, category_distance)
    domain2vec.embeddings = embeddings
    domain2vec.k_neighbours = 20

    neighbours = benchmark(domain2vec.get_knns, 123)
    assert neighbours.size(0) == 20
