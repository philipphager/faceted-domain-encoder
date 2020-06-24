import math

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


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 query_key_value_dims: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()

        self.query_key_value_dims = query_key_value_dims
        self.num_heads = num_heads

        assert query_key_value_dims % num_heads == 0, 'Input dims is not multiple of number of heads'
        self.head_dims = query_key_value_dims // num_heads

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.in_query = nn.Linear(self.query_key_value_dims, self.query_key_value_dims)
        self.in_key = nn.Linear(self.query_key_value_dims, self.query_key_value_dims)
        self.in_value = nn.Linear(self.query_key_value_dims, self.query_key_value_dims)
        self.out_value = nn.Linear(self.query_key_value_dims, self.query_key_value_dims)

    def forward(self, query, key, value, value_lengths):
        assert query.shape == key.shape == value.shape, 'Self-attention requires q, k, v to have same shape'
        assert query.size(-1) == self.query_key_value_dims

        batch_dims = value.size(0)
        sequence_dims = value.size(1)
        embedding_dims = value.size(2)

        # Linear transform all inputs
        query = self.in_query(query)
        key = self.in_key(key)
        value = self.in_value(value)

        query = self._split_heads(query, self.num_heads, self.head_dims)
        key = self._split_heads(key, self.num_heads, self.head_dims)
        value = self._split_heads(value, self.num_heads, self.head_dims)

        # Scaled dot product attention
        attention_weights = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = attention_weights / math.sqrt(self.head_dims)

        # Set attention for padding to -inf so Softmax will return 0
        padding_mask = self._mask_padding(value_lengths, sequence_dims)
        attention_weights = attention_weights.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)
        x = torch.matmul(attention_weights, value)
        x = self._concat_heads(x)
        x = self.out_value(x)

        assert x.size(0) == batch_dims
        assert x.size(1) == sequence_dims
        assert x.size(2) == embedding_dims

        return x, attention_weights

    def _split_heads(self, x, num_attention_heads, attention_head_dims):
        batch_dims = x.size(0)
        sequence_dims = x.size(1)

        # Move from (batch x sequence x embeddings) to (batch x  heads x sequence x embedding chunk)
        x = x.view(batch_dims, sequence_dims, num_attention_heads, attention_head_dims)
        return x.permute(0, 2, 1, 3)

    def _concat_heads(self, x):
        batch_dims = x.size(0)
        sequence_dims = x.size(2)
        embedding_dims = x.size(1) * x.size(3)

        # Move from (batch x heads x sequence x embedding chunk) to (batch x sequence x embeddings)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_dims, sequence_dims, embedding_dims)

    def _mask_padding(self, value_lengths, sequence_dims):
        '''
        Boolean mask to hide padding values during attention computation
        Shape: Batch X Sentence Length, padding values = True, normal words = False
        '''
        mask = torch.arange(sequence_dims).to(self._get_device())
        mask = (mask[None, :] >= value_lengths[:, None]).bool()
        return mask

    def _get_device(self):
        return next(self.parameters()).device
