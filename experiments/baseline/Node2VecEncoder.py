import hydra
import numpy as np
import torch
from gensim.models import Word2Vec

from faceted_domain_encoder.graph import MeSHFactory
from faceted_domain_encoder.util.preprocessing import Tokenizer, EntityLinker


class Node2VecEncoder:
    def __init__(self, dimensions=100):
        super().__init__()
        self.dimensions = dimensions
        self.model = None
        self.tokenizer = None
        self.entity_linker = None

    def fit(self, config):
        path = hydra.utils.to_absolute_path(config.graph.embedding)
        self.model = Word2Vec.load(path)
        self.tokenizer = Tokenizer()
        self.entity_linker = EntityLinker(self._get_graph(config))

    def embed(self, documents):
        embeddings = np.zeros([len(documents), self.dimensions])

        for i, document in enumerate(documents):
            tokens = self.tokenizer(document)
            tokens, entities, categories = self.entity_linker(tokens)
            entity_embeddings = [self.model[e] for e in entities if e]
            if len(entity_embeddings) > 0:
                embeddings[i] = np.mean(entity_embeddings, axis=0)

        return torch.from_numpy(embeddings)

    def _get_graph(self, config):
        graph_path = hydra.utils.to_absolute_path(config.graph.path)
        in_path = hydra.utils.to_absolute_path(config.graph.raw_path)
        factory = MeSHFactory(graph_path, in_path)
        return factory.load()
