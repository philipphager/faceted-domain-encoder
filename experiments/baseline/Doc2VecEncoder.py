import numpy as np
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from faceted_domain_encoder.util.preprocessing import Tokenizer


class Doc2VecEncoder:
    def __init__(self, dimensions=100):
        super().__init__()
        self.dimensions = dimensions
        self.tokenizer = Tokenizer()
        self.model = None

    def fit(self, documents):
        documents = [self.tokenizer(d) for d in documents]
        documents = [TaggedDocument(d, [i]) for i, d in enumerate(documents)]
        self.model = Doc2Vec(documents, vector_size=self.dimensions, min_count=0, window=4, workers=4)

    def embed(self, documents):
        embeddings = np.empty([len(documents), self.dimensions])

        for i, document in enumerate(documents):
            tokens = self.tokenizer(document)
            embeddings[i] = self.model.infer_vector(tokens)

        return torch.from_numpy(embeddings)
