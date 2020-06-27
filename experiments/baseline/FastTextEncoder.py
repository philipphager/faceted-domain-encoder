import fasttext
import hydra
import numpy as np
import torch

from faceted_domain_encoder.util.preprocessing import Tokenizer


class FastTextEncoder:
    def __init__(self, pretrained=False, dimensions=300):
        super().__init__()
        self.pretrained = pretrained
        self.dimensions = dimensions
        self.tokenizer = Tokenizer()
        self.model = None

    def fit(self, config):
        if self.pretrained:
            path = hydra.utils.to_absolute_path(config.word.embedding)
            self.model = fasttext.load_model(path)
        else:
            path = hydra.utils.to_absolute_path(config.data.train_path)
            self.model = fasttext.train_unsupervised(path, dim=self.dimensions)

    def embed(self, documents):
        embeddings = np.empty([len(documents), self.dimensions])

        for i, document in enumerate(documents):
            tokens = self.tokenizer(document)
            word_embeddings = [self.model[token] for token in tokens]
            embeddings[i] = np.mean(word_embeddings, axis=0)

        return torch.from_numpy(embeddings)
