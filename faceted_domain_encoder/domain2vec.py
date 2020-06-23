import torch
from torch.utils import data

from .data import EmbeddingDataset
from .util.linalg import normalize


class CategoryDomain2Vec:
    def __init__(self, model, category_distance):
        self.model = model
        self.category_distance = category_distance

        self.batch_size = 100
        self.k_neighbours = 0
        self.embeddings = None
        self.category_embeddings = None
        self.nearest_neighbours = None

    def train(self, indices, categories, lengths, k_neighbours=24):
        """
        Pre-compute model embeddings for the next epoch to enable nearest neighbour search.
        Finding k-nearest neighbours in two steps:
        1. Calculate the distances in for each category to all documents in corpus.
        2. Return the neighbours with the lowest added distance across all categories.
        """
        assert k_neighbours < len(indices), 'K neighbours cannot be larger than documents in corpus minus one'
        self.k_neighbours = k_neighbours
        self.embeddings = self.embed(indices, categories, lengths)
        self.embeddings = normalize(self.embeddings)

    def embed(self, indices, categories, lengths):
        batches = []
        dataset = EmbeddingDataset(indices, categories, lengths)
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        with torch.set_grad_enabled(False):
            for document, category, length in loader:
                batch = self.model.forward_batch(document.cuda(), category.cuda(), length.cuda())
                batches.append(batch)

        embeddings = torch.cat(batches)
        self.model.train()
        return embeddings.cpu().detach()

    def get_distance(self, d1, d2):
        return self.category_distance(self.embeddings[d1], self.embeddings[d2]).squeeze()

    def get_distance_to_items(self, d, documents):
        return self.category_distance(self.embeddings[d], self.embeddings[documents])

    def get_knns(self, d):
        category_distances = self.category_distance(self.embeddings[d], self.embeddings).sum(1)
        sort_keys = torch.argsort(category_distances)
        return sort_keys[1:self.k_neighbours + 1]

    def get_category_knns(self, d, category):
        category_distances = self.category_distance(self.embeddings[d], self.embeddings)[:, category]
        sort_keys = torch.argsort(category_distances)
        return sort_keys[1:self.k_neighbours + 1]
