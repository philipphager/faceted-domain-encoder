import torch
from torch import nn


class CategoricalGraphAttention(nn.Module):
    def __init__(self,
                 document_embedding_dims: int,
                 graph_embedding_dims: int,
                 num_categories: int,
                 dropout: float):
        super().__init__()

        assert document_embedding_dims % num_categories == 0, 'Input dims is not multiple of number of heads'
        self.document_embedding_dims = document_embedding_dims
        self.graph_embedding_dims = graph_embedding_dims
        self.num_categories = num_categories
        self.softmax = nn.Softmax(dim=-1)
        self.category_dims = self.document_embedding_dims // self.num_categories
        self.category_classifier = CategoryClassifier(
            document_embedding_dims + graph_embedding_dims,
            num_categories,
            dropout)

    def forward(self, x, x_graph, lengths):
        x_context = torch.cat((x, x_graph), dim=-1)
        attention = self.category_classifier(x_context)

        # Permute to (batch x category x sequence)
        attention = attention.permute(0, 2, 1)
        attention = self._mask_padding(attention, lengths)
        attention = self.softmax(attention)

        x = self._split_categories(x)
        x = attention.unsqueeze(-1) * x
        x = self._concat_categories(x)

        return x, attention

    def _mask_padding(self, attention, value_lengths):
        """
        Mask padding tokens before Softmax with -inf.
        """
        mask = torch.zeros_like(attention).type_as(attention)

        for i, length in enumerate(value_lengths):
            mask[i, :, length:] = float('-inf')

        attention = attention + mask
        return attention

    def _split_categories(self, x):
        batch_dims = x.size(0)
        sequence_dims = x.size(1)

        # Move from (batch x sequence x embeddings) to (batch x  heads x sequence x embedding chunk)
        x = x.view(batch_dims, sequence_dims, self.num_categories, self.category_dims)
        return x.permute(0, 2, 1, 3)

    def _concat_categories(self, x):
        batch_dims = x.size(0)
        sequence_dims = x.size(2)
        embedding_dims = x.size(1) * x.size(3)

        # Move from (batch x heads x sequence x embedding chunk) to (batch x sequence x embeddings)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_dims, sequence_dims, embedding_dims)


class CategoryClassifier(nn.Module):
    def __init__(self, in_dims, out_dims, dropout):
        super(CategoryClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.classifier(x)
