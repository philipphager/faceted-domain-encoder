import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class GRUEncoder(nn.Module):
    def __init__(self,
                 word_embedding_dims: int,
                 document_embedding_dims: int,
                 is_bidirectional: bool,
                 dropout: float,
                 num_layers: int):
        super().__init__()

        self.word_embedding_dims = word_embedding_dims
        self.document_embedding_dims = document_embedding_dims
        self.hidden_dims = self.document_embedding_dims
        self.is_bidirectional = is_bidirectional
        self.direction_dims = 2 if self.is_bidirectional else 1
        self.dropout = dropout
        self.num_layers = num_layers

        self.gru = nn.GRU(self.word_embedding_dims,
                          self.hidden_dims,
                          self.num_layers,
                          batch_first=True,
                          bidirectional=self.is_bidirectional,
                          dropout=dropout)

    def forward(self, documents, document_lengths) -> torch.Tensor:
        batch_dims = documents.size(0)
        sequence_dims = documents.size(1)
        embedding_dims = documents.size(2)

        assert embedding_dims == self.word_embedding_dims

        x = pack_padded_sequence(documents, document_lengths, batch_first=True, enforce_sorted=False)
        x, hidden_state = self.gru(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=sequence_dims)

        # Forward and backward pass is concatenated in output vector. Split and average.
        x = (x[:, :, :self.document_embedding_dims] + x[:, :, self.document_embedding_dims:]) / 2

        assert x.size(0) == batch_dims
        assert x.size(1) == sequence_dims
        assert x.size(2) == self.document_embedding_dims
        assert x.dim() == 3
        return x


class LSTMEncoder(nn.Module):
    def __init__(self,
                 word_embedding_dims: int,
                 document_embedding_dims: int,
                 is_bidirectional: bool,
                 dropout: float,
                 num_layers: int):
        super().__init__()

        self.word_embedding_dims = word_embedding_dims
        self.document_embedding_dims = document_embedding_dims
        self.hidden_dims = self.document_embedding_dims
        self.is_bidirectional = is_bidirectional
        self.direction_dims = 2 if self.is_bidirectional else 1
        self.dropout = dropout
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.word_embedding_dims,
                            self.hidden_dims,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=self.is_bidirectional,
                            dropout=dropout)

    def forward(self, documents, document_lengths) -> torch.Tensor:
        batch_dims = documents.size(0)
        sequence_dims = documents.size(1)
        embedding_dims = documents.size(2)

        assert embedding_dims == self.word_embedding_dims

        x = pack_padded_sequence(documents, document_lengths, batch_first=True, enforce_sorted=False)
        x, (hidden_state, cell_state) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=sequence_dims)

        # Forward and backward pass is concatenated in output vector. Split and average.
        x = (x[:, :, :self.document_embedding_dims] + x[:, :, self.document_embedding_dims:]) / 2

        assert x.size(0) == batch_dims
        assert x.size(1) == sequence_dims
        assert x.size(2) == self.document_embedding_dims
        assert documents.dim() == 3
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 word_embedding_dims: int,
                 document_embedding_dims: int,
                 num_heads: int,
                 dropout: float,
                 num_layers: int):
        super().__init__()

        self.word_embedding_dims = word_embedding_dims
        self.document_embedding_dims = document_embedding_dims
        self.dropout = dropout
        self.num_layers = num_layers
        self.expand = nn.Linear(self.word_embedding_dims, self.document_embedding_dims)
        self.positional_encoding = PositionalEncoding(hidden_dims=self.document_embedding_dims, dropout=self.dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.document_embedding_dims,
            dim_feedforward=self.document_embedding_dims,
            nhead=num_heads,
            dropout=self.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)

    def forward(self, documents, document_lengths) -> torch.Tensor:
        batch_dims = documents.size(0)
        sequence_dims = documents.size(1)
        embedding_dims = documents.size(2)

        assert embedding_dims == self.word_embedding_dims

        # Word to document embedding dimensions
        x = self.expand(documents)
        # Transpose (batch x sequence x word) to (sequence x batch x word)
        x = x.transpose(0, 1)

        assert x.size(0) == sequence_dims
        assert x.size(1) == batch_dims
        assert x.size(2) == self.document_embedding_dims

        x = self.positional_encoding(x)
        attention_mask = self._mask_attention(sequence_dims, documents)
        padding_mask = self._mask_padding(document_lengths, sequence_dims, documents)
        x = self.transformer(x, mask=attention_mask, src_key_padding_mask=padding_mask)

        # Transpose (sequence x batch x word) to (batch x sequence x word)
        x = x.transpose(0, 1)

        # PyTorch Transformer outputs non-zero values for padding even with mask, manually discard those Tensors
        for i, length in enumerate(document_lengths):
            x[i][length:] = torch.zeros(self.document_embedding_dims)

        assert x.size(0) == batch_dims
        assert x.size(1) == sequence_dims
        assert x.size(2) == self.document_embedding_dims
        assert x.dim() == 3
        return x

    def _mask_padding(self, document_lengths, sequence_dims, type):
        '''
        Boolean mask to hide padding values during attention computation
        Shape: Batch X Sentence Length, padding values = True, normal words = False
        '''
        mask = torch.arange(sequence_dims).type_as(type)
        return (mask[None, :] >= document_lengths[:, None]).bool()

    def _mask_attention(self, sequence_dims, type):
        '''
        Float mask to hide attention values for softmax calculation.
        The mask is triangular shaped to hide upcoming words in a sentence,
        forcing attention calculation from left to right.
        '''
        mask = (torch.triu(torch.ones(sequence_dims, sequence_dims)) == 1).transpose(0, 1).type_as(type)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding as referenced in PyTorch documentation:
    https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/transformer_tutorial.ipynb
    '''

    def __init__(self,
                 hidden_dims,
                 dropout=0.1,
                 max_length=500):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        positional_encoder = torch.zeros(max_length, hidden_dims)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dims, 2).float() * (-math.log(10000.0) / hidden_dims))
        positional_encoder[:, 0::2] = torch.sin(position * div_term)
        positional_encoder[:, 1::2] = torch.cos(position * div_term)
        positional_encoder = positional_encoder.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoder', positional_encoder)

    def forward(self, x):
        self.positional_encoder = self.positional_encoder.type_as(x)
        x = x + self.positional_encoder[:x.size(0), :]
        return self.dropout(x)
