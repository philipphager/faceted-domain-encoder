import torch
from torch import nn

from faceted_domain_encoder.util.linalg import normalize, knns, CategoryDistance


def test_normalize():
    embedding = torch.rand(30, 512)

    embedding_norms = torch.norm(embedding, dim=1)
    assert not (embedding_norms.round() == 1.0).all()

    embedding = normalize(embedding)

    embedding_norms = torch.norm(embedding, dim=1)
    assert embedding_norms.size(0) == 30
    assert (embedding_norms.round() == 1.0).all()


def test_normalize_keep_zero_vectors():
    embedding = torch.rand(30, 512)
    embedding[0, :] = 0
    embedding[1, :] = 0

    embedding = normalize(embedding)
    embedding_norms = torch.norm(embedding, dim=1)

    assert (embedding[0] == 0).all()
    assert (embedding[1] == 0).all()
    assert (embedding_norms[2:].round() == 1.0).all()


def test_knns():
    embeddings = torch.tensor([
        [0.1, 0.2, 0.1],
        [0.4, 0.25, 0.3],
        [0.1, 0.25, 0.1],
    ])

    expected_neighbours = torch.tensor([
        [2, 1],
        [0, 2],
        [0, 1]
    ])

    cosine = nn.CosineSimilarity(-1)

    neighbours, distances = knns(embeddings, 2)
    assert torch.equal(expected_neighbours, neighbours)
    assert torch.equal(torch.argsort(distances, dim=1), torch.tensor([[0, 1], [0, 1], [0, 1]]))
    assert torch.isclose(distances[0, 0], (1 - cosine(embeddings[2], embeddings[0])))
    assert torch.isclose(distances[2, 1], (1 - cosine(embeddings[2], embeddings[1])))


def test_knns_preforms_normalization():
    embeddings = torch.tensor([
        [0.1, 0.2, 0.1],
        [0.04, 0.025, 0.03],
        [10, 25, 1],
    ])

    expected_neighbours = torch.tensor([
        [2, 1],
        [0, 2],
        [0, 1]
    ])

    cosine = nn.CosineSimilarity(-1)

    neighbours, distances = knns(embeddings, 2)
    assert torch.equal(expected_neighbours, neighbours)
    assert torch.equal(torch.argsort(distances, dim=1), torch.tensor([[0, 1], [0, 1], [0, 1]]))
    assert torch.isclose(distances[0, 0], (1 - cosine(embeddings[2], embeddings[0])))
    assert torch.isclose(distances[2, 1], (1 - cosine(embeddings[2], embeddings[1])))


def test_category_distance_all_equal():
    num_categories = 10
    num_dimensions = 500
    percentages = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    embedding = torch.randn(50)
    x1 = embedding.repeat(10)
    x2 = embedding.repeat(10)

    category_distance = CategoryDistance(num_categories, num_dimensions, percentages)
    distances = category_distance(x1, x2)

    assert distances.size(0) == 1
    assert distances.size(1) == num_categories
    assert (distances == 0).all()


def test_category_distance_one_differ():
    num_categories = 10
    num_dimensions = 500
    min_dimensions = 10
    percentages = torch.tensor([0.2, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    embedding = torch.randn(50)
    cat1 = torch.randn(90)
    cat2 = torch.randn(90)

    x1 = embedding.repeat(10)
    x2 = embedding.repeat(10)
    x1[:90] = cat1
    x2[:90] = cat2

    category_distance = CategoryDistance(num_categories, num_dimensions, percentages, min_dimensions)
    distances = category_distance(x1, x2)

    assert distances.size(0) == 1
    assert distances.size(1) == num_categories
    assert (distances[0, 0].item() != 0)
    assert (distances[0, 1:] == 0).all()


def test_category_distance_one_similar():
    num_categories = 10
    num_dimensions = 500
    min_dimensions = 10
    percentages = torch.tensor([0, 0, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1])

    x1 = torch.randn(500)
    x2 = torch.randn(500)
    cat = torch.randn(130)

    x1[220: 350] = cat
    x2[220: 350] = cat

    category_distance = CategoryDistance(num_categories, num_dimensions, percentages, min_dimensions)
    distances = category_distance(x1, x2)

    assert distances.size(0) == 1
    assert distances.size(1) == num_categories
    assert (distances[0, 6].item() == 0)
    assert (distances[0, :6] != 0).all()
    assert (distances[0, 7:] != 0).all()


def test_category_distance_spillover():
    num_categories = 10
    num_dimensions = 500
    min_dimensions = 10
    percentages = torch.tensor([0, 0, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1])

    embedding = torch.randn(50)
    cat1 = torch.randn(131)
    cat2 = torch.randn(131)

    x1 = embedding.repeat(10)
    x2 = embedding.repeat(10)
    x1[220: 351] = cat1
    x2[220: 351] = cat2

    category_distance = CategoryDistance(num_categories, num_dimensions, percentages, min_dimensions)
    distances = category_distance(x1, x2)

    assert distances.size(0) == 1
    assert distances.size(1) == num_categories
    assert (distances[0, 6].item() != 0)
    assert (distances[0, 7].item() != 0)
    assert (distances[0, :6] == 0).all()
    assert (distances[0, 8:] == 0).all()
