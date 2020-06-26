import copy
import logging
import os
from enum import Enum

import hydra
# noinspection PyUnresolvedReferences
import swifter
import torch
from omegaconf import OmegaConf
from pytorch_lightning.core import LightningModule
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .criterion import MaskedMSELoss
from .data import SiameseDataset, SamplingStrategy
from .domain2vec import CategoryDomain2Vec
from .encoder import TransformerEncoder, GRUEncoder, LSTMEncoder
from .graph.aviation import AviationFactory
from .graph.mesh import MeSHFactory
from .graph2vec import CategoryGraph2Vec
from .normalization import DocumentTfidfNormalizer, CorpusTfidfNormalizer, PassThroughNormalizer
from .optimizer import RAdam
from .pooling import MeanPooling, MaxPooling, CategoryAttentionPooling, SelfAttentionPooling
from .util.linalg import CategoryDistance
from .util.plotting import plot_text, plot_attention, plot_category_weight
from .util.preprocessing import load_data_file, TextProcessor, load_embeddings

logger = logging.getLogger(__name__)


class NormalizationStrategy(Enum):
    PASS = 'pass'
    DOCUMENT = 'document'
    CORPUS = 'corpus'


class PoolingStrategy(Enum):
    CATEGORY_ATTENTION = 'category_attention'
    SELF_ATTENTION = 'self_attention'
    MAX = 'max'
    MEAN = 'mean'


class EncoderStrategy(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    TRANSFORMER = 'transformer'


class FacetedDomainEncoder(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder_strategy = EncoderStrategy(self.hparams.model.encoder)
        self.pooling_strategy = PoolingStrategy(self.hparams.model.pooling)
        self.normalization_strategy = NormalizationStrategy(self.hparams.model.normalizer)

        # Define components
        self.graph = self._get_graph(self.hparams.graph.name)
        self.encoder = self._get_encoder(self.encoder_strategy)
        self.pooling = self._get_pooling(self.pooling_strategy)
        self.word_dropout = nn.Dropout(self.hparams.model.dropout)
        self.word_normalize = nn.LayerNorm(self.hparams.model.word_embedding_dims)
        self.mse_loss = MaskedMSELoss()
        self.category_distance = CategoryDistance(
            self.hparams.model.document_embedding_dims,
            self.hparams.graph.num_categories)

        # Define dataset and loaders
        self.df = None
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.train_dataset = None
        self.val_dataset = None

        # Init text pre-processing pipeline
        self.processor = TextProcessor(
            self.graph,
            OmegaConf.to_container(self.hparams.graph.categories),
            self.hparams.data.max_length,
            self.hparams.data.should_stop)

        # Load vocabulary if already exists
        path = self.hparams.model.vocabulary_path

        if os.path.exists(path):
            logger.info('Vocabulary found on disk. Loading: %s', path)
            self.processor.load(path)
            self._init_normalizer()
            self._init_embeddings()
        else:
            # Define empty embedding layers, embeddings are created after pre-processing
            self.word_embedding = nn.Embedding(len(self.processor.vocabulary), self.hparams.model.word_embedding_dims)
            self.graph_embedding = nn.Embedding(len(self.processor.vocabulary), self.hparams.model.graph_embedding_dims)
        logger.info('Created model: %s, %s, %s', hparams.model.encoder, hparams.model.pooling, hparams.model.normalizer)

    def forward(self,
                doc1, doc1_categories, doc1_lengths,
                doc2, doc2_categories, doc2_lengths) -> torch.Tensor:
        x1 = self.forward_batch(doc1, doc1_categories, doc1_lengths)
        x2 = self.forward_batch(doc2, doc2_categories, doc2_lengths)
        return self.category_distance(x1, x2)

    def forward_batch(self, documents, categories, lengths, attention=False):
        x = self.word_embedding(documents)
        x_graph = self.graph_embedding(documents)
        x, x_graph, categories = self._word_dropout(x, x_graph, categories, self.hparams.model.dropout)
        x = self.word_normalize(x)
        x = self.encoder(x, lengths)

        if self.pooling_strategy == PoolingStrategy.CATEGORY_ATTENTION:
            x, attention_weights = self.pooling(x, x_graph, lengths)
            x = self.normalizer(x, documents, categories)
            return (x, attention_weights) if attention else x
        elif self.pooling_strategy == PoolingStrategy.SELF_ATTENTION:
            x, attention_weights = self.pooling(x, lengths)
            x = self.normalizer(x, documents, categories)
            return (x, attention_weights) if attention else x

        x = self.pooling(x, lengths)
        x = self.normalizer(x, documents, categories)
        return x

    def _word_dropout(self, x, x_graph, categories, p):
        if p < 0. or p > 1.:
            raise ValueError('Dropout probability has to be between 0 and 1')

        if self.training:
            batch_dim = x.size(0)
            sequence_dim = x.size(1)
            mask = torch.rand(batch_dim, sequence_dim).type_as(x)
            mask = mask > p
            x = x * mask.unsqueeze(-1)
            x_graph = x_graph * mask.unsqueeze(-1)
            categories[mask] = -1

        return x, x_graph, categories

    def embed(self, documents, attention=False):
        self.eval()
        self.processor.vocabulary.freeze()
        indices, categories, lengths = zip(*[self.processor(d) for d in documents])
        indices = torch.stack(indices)
        categories = torch.stack(categories)
        lengths = torch.tensor(lengths, dtype=torch.int)

        # Move to device (handled by Lightning)
        indices = indices
        categories = categories
        lengths = lengths
        return self.forward_batch(indices, categories, lengths, attention)

    def prepare_data(self):
        self.df = load_data_file(hydra.utils.to_absolute_path(self.hparams.data.train_path))

        # Optionally sample training data
        if self.hparams.data.should_sample:
            logger.info('Sampling training file: %s', self.hparams.data.samples)
            self.df = self.df.sample(self.hparams.data.samples, random_state=self.hparams.data.random_state)

        logger.info('Preprocess text')
        columns = ['document_index', 'document_category', 'document_length']
        self.df[columns] = self.df.swifter.apply(lambda row: list(self.processor(row.text)),
                                                 axis=1,
                                                 result_type='expand')

        if self.hparams.data.test_path:
            logger.info('Read test file into vocabulary, ignored during training')
            self.test_df = load_data_file(hydra.utils.to_absolute_path(self.hparams.data.test_path))
            self.test_df[columns] = self.test_df.swifter.apply(lambda row: list(self.processor(row.text)),
                                                               axis=1,
                                                               result_type='expand')

        # Train / validation split
        self.train_df, self.validation_df = train_test_split(
            self.df,
            test_size=self.hparams.data.split,
            random_state=self.hparams.data.random_state,
            shuffle=self.hparams.data.shuffle_before_split)

        # Normalizers require the vocabulary, init after pre-processing
        self._init_normalizer()
        self._init_embeddings()
        self.processor.save(self.hparams.model.vocabulary_path)

    def _init_embeddings(self):
        # Embed vocabulary with FastText and Node2Vec
        word_embeddings, graph_embeddings = load_embeddings(
            vocabulary=self.processor.vocabulary,
            word_embedding_path=hydra.utils.to_absolute_path(self.hparams.word.embedding),
            graph_embedding_path=hydra.utils.to_absolute_path(self.hparams.graph.embedding))

        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, padding_idx=0, freeze=True)
        self.graph_embedding = nn.Embedding.from_pretrained(graph_embeddings, padding_idx=0, freeze=True)

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = self._get_dataset(
            frame=self.train_df,
            sampling_strategy=SamplingStrategy(self.hparams.training.sampling),
            samples_per_document=self.hparams.training.samples_per_document,
            graph_k=self.hparams.training.k_graph,
            embedding_k=self.hparams.training.k_embedding)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.training.batch_size,
            num_workers=self.hparams.training.num_workers,
            shuffle=True)

    def val_dataloader(self):
        self.val_dataset = self._get_dataset(
            frame=self.validation_df,
            sampling_strategy=SamplingStrategy(self.hparams.validation.sampling),
            samples_per_document=self.hparams.validation.samples_per_document,
            graph_k=self.hparams.validation.k_graph,
            embedding_k=self.hparams.validation.k_embedding)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.validation.batch_size,
            num_workers=self.hparams.validation.num_workers)

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.training.learning_rate)

    def training_step(self, batch, step):
        x1, x1_category, x1_length, x2, x2_category, x2_length, y = batch
        y_predict = self.forward(x1, x1_category, x1_length, x2, x2_category, x2_length)
        loss = self.mse_loss(y_predict, y)
        log = {'step/train/loss': loss}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, steps):
        self.train_dataset.on_epoch(1)
        mean_loss = torch.stack([step['loss'] for step in steps]).mean()
        log = {'epoch/train/loss': mean_loss}
        return {'loss': mean_loss, 'log': log}

    def validation_step(self, batch, step):
        x1, x1_category, x1_length, x2, x2_category, x2_length, y = batch
        y_predict = self.forward(x1, x1_category, x1_length, x2, x2_category, x2_length)
        loss = self.mse_loss(y_predict, y)
        log = {'step/val/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_epoch_end(self, steps):
        mean_loss = torch.stack([step['loss'] for step in steps]).mean()
        log = {'epoch/val/loss': mean_loss}
        return {'val_loss': mean_loss, 'log': log}

    def on_train_end(self):
        logger.info('Training end')

        if self.hparams.trainer.export_embeddings:
            logger.info('Training end: Embed full dataset')
            index = torch.stack(self.df.document_index.to_list())
            category = torch.stack(self.df.document_category.to_list())
            length = torch.tensor(self.df.document_length.to_list(), dtype=torch.int)

            domain2vec = CategoryDomain2Vec(self, self.category_distance)
            embeddings = domain2vec.embed(index, category, length)

            self.logger.experiment.add_embedding(embeddings)
            self.df.to_pickle('documents.pkl')
            torch.save(embeddings, 'embeddings.pt')
            logger.info('Training end: Embed full dataset finished')

    def _get_dataset(self, frame, sampling_strategy, samples_per_document, graph_k, embedding_k):
        domain2vec = CategoryDomain2Vec(self, self.category_distance)
        graph2vec = CategoryGraph2Vec(
            copy.deepcopy(self.graph_embedding),
            self.processor.vocabulary,
            self.hparams.graph.num_categories)

        return SiameseDataset(
            frame,
            domain2vec,
            graph2vec,
            sampling_strategy=sampling_strategy,
            samples_per_document=samples_per_document,
            graph_k=graph_k,
            embedding_k=embedding_k)

    def _get_graph(self, name):
        path = hydra.utils.to_absolute_path(self.hparams.graph.path)

        if name == 'mesh':
            graph_factory = MeSHFactory(graph_path=path)
        elif name == 'aviation':
            graph_factory = AviationFactory(graph_path=path)
        else:
            raise RuntimeError(f'Unknown graph {name}')

        return graph_factory.load()

    def _get_encoder(self, strategy):
        if strategy == EncoderStrategy.GRU:
            encoder = GRUEncoder(
                self.hparams.model.word_embedding_dims,
                self.hparams.model.document_embedding_dims,
                self.hparams.model.is_bidirectional,
                self.hparams.model.dropout,
                self.hparams.model.encoder_layers)
        elif strategy == EncoderStrategy.LSTM:
            encoder = LSTMEncoder(
                self.hparams.model.word_embedding_dims,
                self.hparams.model.document_embedding_dims,
                self.hparams.model.is_bidirectional,
                self.hparams.model.dropout,
                self.hparams.model.encoder_layers)
        elif strategy == EncoderStrategy.TRANSFORMER:
            encoder = TransformerEncoder(
                self.hparams.model.word_embedding_dims,
                self.hparams.model.document_embedding_dims,
                self.hparams.graph.num_categories,
                self.hparams.model.dropout,
                self.hparams.model.encoder_layers)
        else:
            raise RuntimeError(f'Unknown encoder strategy {strategy}')

        return encoder

    def _get_pooling(self, strategy):
        if strategy == PoolingStrategy.MEAN:
            pooling = MeanPooling(self.hparams.model.document_embedding_dims)
        elif strategy == PoolingStrategy.MAX:
            pooling = MaxPooling(self.hparams.model.document_embedding_dims)
        elif strategy == PoolingStrategy.CATEGORY_ATTENTION:
            pooling = CategoryAttentionPooling(
                self.hparams.model.document_embedding_dims,
                self.hparams.model.graph_embedding_dims,
                self.hparams.graph.num_categories,
                self.hparams.model.dropout)
        elif strategy == PoolingStrategy.SELF_ATTENTION:
            pooling = SelfAttentionPooling(
                self.hparams.model.document_embedding_dims,
                self.hparams.graph.num_categories,
                self.hparams.model.dropout)
        else:
            raise RuntimeError(f'Unknown pooling strategy {strategy}')

        return pooling

    def _init_normalizer(self):
        if self.normalization_strategy == NormalizationStrategy.CORPUS:
            self.normalizer = CorpusTfidfNormalizer(
                self.processor.vocabulary,
                self.hparams.model.document_embedding_dims,
                self.hparams.graph.num_categories)
        elif self.normalization_strategy == NormalizationStrategy.DOCUMENT:
            self.normalizer = DocumentTfidfNormalizer(
                self.processor.vocabulary,
                self.hparams.model.document_embedding_dims,
                self.hparams.graph.num_categories)
        elif self.normalization_strategy == NormalizationStrategy.PASS:
            self.normalizer = PassThroughNormalizer()

    def display(self, document, filter_category=None, raw_html=False):
        return plot_text(self, document, filter_category, raw_html)

    def display_attention(self, document):
        return plot_attention(self, document)

    def display_category_weight(self, document):
        return plot_category_weight(self, document)
