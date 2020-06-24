import logging

import torch
import torch.nn.functional as F

from torch import nn

logger = logging.getLogger(__name__)

class PassThroughNormalizer:
    def __init__(self):
        pass

    def __call__(self, x, documents, categories):
        return x


class CorpusTfidfNormalizer:
    def __init__(self, vocabulary, document_embedding_dims, num_categories):
        self.vocabulary = vocabulary
        self.document_embedding_dims = document_embedding_dims
        self.num_categories = num_categories
        self.category_dims = self.document_embedding_dims // self.num_categories
        self.cosine = nn.CosineSimilarity(-1)
        self.weights = self._category_weights()

    def __call__(self, x, documents, categories):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        self.weights = self.weights.type_as(x)

        x = self._split_categories(x)
        x = F.normalize(x, p=2, dim=2)
        x = x * self.weights.unsqueeze(-1)
        x = self._concat_categories(x)
        return x

    def _category_weights(self):
        category_percentage = torch.zeros(self.num_categories)

        for index, category in self.vocabulary.index2category.items():
            if category != -1:
                category_percentage[category] += self.vocabulary.index2idf(index)

        category_percentage = category_percentage / category_percentage.sum()
        logger.info('CorpusTfidfNormalization: %s', category_percentage)
        return category_percentage

    def _split_categories(self, x):
        # Move from (batch x embeddings) to (batch x  category x category embeddings)
        return x.view(-1, self.num_categories, self.category_dims)

    def _concat_categories(self, x):
        batch_dims = x.size(0)
        # Move from (batch x  category x category embeddings) to (batch x embeddings)
        return x.view(batch_dims, self.document_embedding_dims)


class DocumentTfidfNormalizer:
    def __init__(self, vocabulary, document_embedding_dims, num_categories):
        self.vocabulary = vocabulary
        self.document_embedding_dims = document_embedding_dims
        self.num_categories = num_categories
        self.category_dims = self.document_embedding_dims // self.num_categories
        self.cosine = nn.CosineSimilarity(-1)
        self.idf = self.vocabulary.get_idf()

    def __call__(self, x, documents, categories):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        weights = self._category_weights(x, documents, categories)

        x = self._split_categories(x)
        x = F.normalize(x, p=2, dim=2)
        x = x * weights.unsqueeze(-1)
        x = self._concat_categories(x)
        return x

    def _category_weights(self, _type, documents, categories):
        weights = torch.zeros(self.num_categories, documents.size(0)).type_as(_type)
        self.idf = self.idf.type_as(_type)
        idf = self.idf[documents]

        for category in range(self.num_categories):
            tokens_in_category = (categories == category)
            weights[category] = (idf * tokens_in_category).sum(-1)

        # Add weights for non-category words
        tokens_in_category = (categories == -1)
        weights += (idf * tokens_in_category).sum(-1).unsqueeze(0) / self.num_categories
        weights = weights / weights.sum(0)
        return weights.permute(1, 0)

    def _split_categories(self, x):
        # Move from (batch x embeddings) to (batch x  category x category embeddings)
        return x.view(-1, self.num_categories, self.category_dims)

    def _concat_categories(self, x):
        batch_dims = x.size(0)
        # Move from (batch x  category x category embeddings) to (batch x embeddings)
        return x.view(batch_dims, self.document_embedding_dims)
