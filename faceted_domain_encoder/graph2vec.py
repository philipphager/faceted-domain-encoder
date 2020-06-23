import torch
from torch import nn

from .util.linalg import split_categories, pairwise_cosine_distance


class CategoryGraph2Vec:
    def __init__(self,
                 graph_embedding,
                 vocabulary,
                 category_dims=16):
        self.graph_embedding = graph_embedding.cpu()
        self.vocabulary = vocabulary
        self.category_dims = category_dims

        self.graph_embedding_dims = 100
        self.batch_size = 100
        self.k_neighbours = 0
        self.cosine = nn.CosineSimilarity(-1)
        self.embeddings = None

    def train(self, indices, categories, lengths, k_neighbours=24):
        assert k_neighbours < len(indices), 'K neighbours cannot be larger than documents in corpus minus one'
        self.k_neighbours = k_neighbours
        self.embeddings = self.embed(indices, categories, lengths)
        self.embeddings = split_categories(self.embeddings, self.category_dims)
        self.mean_category_distances = self.get_mean_category_distance(self.embeddings)

    def embed(self, indices, categories, lengths):
        document_dims = indices.size(0)
        embedding_dims = self.category_dims * self.graph_embedding_dims
        embeddings = torch.zeros(document_dims, embedding_dims)

        for document, (index, category, length) in enumerate(zip(indices, categories, lengths)):
            category_vectors = []

            # Compute average embedding per category
            for category_index in range(self.category_dims):
                entities_in_category = index[category == category_index]
                num_entities = entities_in_category.size(0)

                if num_entities > 0:
                    # TF-IDF weighting of graph entities
                    idf = torch.tensor([self.vocabulary.index2idf(entity.item()) for entity in entities_in_category])
                    x = self.graph_embedding(entities_in_category)
                    x = x * idf.unsqueeze(-1)
                    x = torch.mean(x, dim=0)
                    category_vectors.append(x)
                else:
                    # All zero vector for non-graph entities
                    category_vectors.append(torch.zeros(100))

            embeddings[document] = torch.cat(category_vectors)

        return embeddings.cpu().detach()

    def get_mean_category_distance(self, embeddings):
        distances = torch.zeros(self.category_dims)

        for i in range(self.category_dims):
            category_embeddings = embeddings[:, i, :]
            category_distances = pairwise_cosine_distance(category_embeddings, category_embeddings)
            distances[i] = category_distances[category_distances != 1].mean()

        # No mention of a category from any document results in NaN
        distances[distances != distances] = 1
        return distances

    def get_distance(self, d1, d2):
        return 1 - self.cosine(self.embeddings[d1], self.embeddings[d2])

    def get_distance_to_items(self, d, documents):
        return 1 - self.cosine(self.embeddings[d], self.embeddings[documents])

    def get_knns(self, d):
        category_distances = 1 - self.cosine(self.embeddings[d], self.embeddings).sum(1)
        sort_keys = torch.argsort(category_distances)
        return sort_keys[1:self.k_neighbours + 1]

    def get_category_knns(self, d, category, n):
        category_distances = 1 - self.cosine(self.embeddings[d, category], self.embeddings[:, category])
        sort_keys = torch.argsort(category_distances)
        sort_keys = sort_keys[1:n + 1]
        sort_keys = sort_keys[sort_keys != d]
        return sort_keys
