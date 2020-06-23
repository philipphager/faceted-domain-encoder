import torch
from torch import nn

from .attention import CategoricalGraphAttention


class MeanPooling(nn.Module):
    def __init__(self, document_embedding_dims: int):
        super().__init__()
        self.document_embedding_dims = document_embedding_dims

    def forward(self, x, lengths):
        batch_dims = x.size(0)

        assert x.dim() == 3
        # Mean all word embeddings into document embedding
        x = torch.sum(x, dim=1) / lengths.reshape(-1, 1)
        assert x.size(0) == batch_dims
        assert x.size(1) == self.document_embedding_dims
        assert x.dim() == 2
        return x


class MaxPooling(nn.Module):
    def __init__(self, document_embedding_dims: int):
        super().__init__()
        self.document_embedding_dims = document_embedding_dims

    def forward(self, x, length):
        batch_dims = x.size(0)

        assert x.dim() == 3
        # Max all word embeddings into document embedding
        x, _ = torch.max(x, dim=1)
        assert x.size(0) == batch_dims
        assert x.size(1) == self.document_embedding_dims
        assert x.dim() == 2
        return x


class CategoryAttentionPooling(nn.Module):
    def __init__(self,
                 document_embedding_dims: int,
                 graph_embedding_dims: int,
                 num_categories: int,
                 dropout: float):
        super().__init__()

        self.document_embedding_dims = document_embedding_dims
        self.graph_embedding_dims = graph_embedding_dims
        self.num_heads = num_categories
        self.dropout = dropout

        self.category_attention = CategoricalGraphAttention(
            document_embedding_dims,
            graph_embedding_dims,
            num_categories,
            dropout)

    def forward(self, x, x_graph, lengths):
        batch_dims = x.size(0)
        sequence_dims = x.size(1)

        assert x.size(2) == self.document_embedding_dims
        assert x_graph.size(2) == self.graph_embedding_dims

        x, attention = self.category_attention(x, x_graph, lengths)

        assert x.size(0) == batch_dims
        assert x.size(1) == sequence_dims
        assert x.size(2) == self.document_embedding_dims

        # Average all attention-scaled word embeddings into document embedding
        x = torch.sum(x, dim=1) / lengths.reshape(-1, 1)

        assert x.size(0) == batch_dims
        assert x.size(1) == self.document_embedding_dims
        assert x.dim() == 2
        return x, attention
