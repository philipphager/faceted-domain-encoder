import math
from enum import Enum

import torch
from torch.utils import data


class SamplingStrategy(Enum):
    RANDOM = 'random'
    UNIFORM = 'uniform'
    ERROR_WEIGHTED = 'error_weighted'


class SiameseDataset(data.Dataset):
    def __init__(self,
                 df,
                 domain2vec,
                 graph2vec,
                 sampling_strategy,
                 samples_per_document,
                 graph_k,
                 embedding_k):

        self.documents = df.text.values
        self.document_index = torch.stack(df.document_index.to_list())
        self.document_category = torch.stack(df.document_category.to_list())
        self.document_length = torch.tensor(df.document_length.to_list(), dtype=torch.int)
        self.num_documents = len(df)

        self.domain2vec = domain2vec
        self.graph2vec = graph2vec
        self.sampling_strategy = sampling_strategy
        self.samples_per_document = samples_per_document
        self.graph_k = graph_k
        self.embedding_k = embedding_k

        self.on_epoch(0)

    def __len__(self):
        return self.num_documents * self.samples_per_document

    def __getitem__(self, i):
        document_id = math.floor(i / self.samples_per_document)

        if self.sampling_strategy == SamplingStrategy.ERROR_WEIGHTED:
            document_id, neighbour_id, distance = self._sample_error_weighted_neighbour(document_id)
        elif self.sampling_strategy == SamplingStrategy.UNIFORM:
            document_id, neighbour_id, distance = self._sample_uniform_neighbour(document_id)
        elif self.sampling_strategy == SamplingStrategy.RANDOM:
            document_id, neighbour_id, distance = self._sample_random(document_id)
        else:
            raise RuntimeError(f'Unknown sampling strategy {self.sampling_strategy}')

        return self.document_index[document_id], \
               self.document_category[document_id], \
               self.document_length[document_id], \
               self.document_index[neighbour_id], \
               self.document_category[neighbour_id], \
               self.document_length[neighbour_id], \
               distance

    def _sample_error_weighted_neighbour(self, document_id):
        neighbours = self.document2neighbours[document_id]
        documents = neighbours['documents']
        probabilities = neighbours['probabilities']
        graph_distances = neighbours['graph']

        sample_position = torch.multinomial(probabilities, 1).item()
        neighbour_id = documents[sample_position].item()
        distance = graph_distances[sample_position]
        return document_id, neighbour_id, distance

    def _sample_uniform_neighbour(self, document_id):
        neighbours = self.document2neighbours[document_id]
        documents = neighbours['documents']
        graph_distances = neighbours['graph']

        sample_position = torch.randint(low=0, high=len(documents), size=(1,)).item()
        neighbour_id = documents[sample_position].item()
        distance = graph_distances[sample_position]
        return document_id, neighbour_id, distance

    def _sample_random(self, document_id):
        neighbour_id = torch.randint(self.num_documents, (1,)).item()
        distance = self.graph2vec.get_distance(document_id, neighbour_id)
        return document_id, neighbour_id, distance

    def on_epoch(self, epoch):
        if epoch == 0:
            # Graph embeddings are static, train only once
            self.graph2vec.train(
                self.document_index,
                self.document_category,
                self.document_length,
                k_neighbours=self.graph_k
            )

        self.domain2vec.train(
            self.document_index,
            self.document_category,
            self.document_length,
            k_neighbours=self.embedding_k
        )

        self.document2neighbours = self.get_neighbours()

    def get_neighbours(self):
        document2neighbours = {}

        for document in range(self.num_documents):
            categories = self.document_category[document]
            categories = categories[categories != -1].unique()
            num_categories = categories.size(0)

            if num_categories > 0:
                graph_nns = []
                category_samples = self.graph_k // num_categories

                for category in categories:
                    graph_nns.append(self.graph2vec.get_category_knns(document, category, category_samples))

                graph_nns = torch.cat(graph_nns)
            else:
                graph_nns = self.graph2vec.get_knns(document)

            embedding_nns = self.domain2vec.get_knns(document)
            neighbours = torch.cat([graph_nns, embedding_nns]).unique()
            graph_distances = self.graph2vec.get_distance_to_items(document, neighbours)
            domain_distances = self.domain2vec.get_distance_to_items(document, neighbours)

            # Precompute probabilities for error weighted sampling
            probabilities = ((graph_distances - domain_distances) ** 2).sum(-1)
            probabilities = probabilities / probabilities.sum()

            document2neighbours[document] = {
                'documents': torch.IntTensor(list(neighbours)),
                'graph': graph_distances,
                'domain': domain_distances,
                'probabilities': probabilities
            }

        return document2neighbours


class EmbeddingDataset(data.Dataset):
    def __init__(self, indices, categories, lengths):
        self.indices = indices
        self.categories = categories
        self.lengths = lengths

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.indices[i], \
               self.categories[i], \
               self.lengths[i]
