import json
import logging
import re

import numpy as np
import pandas as pd
import spacy
import torch
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from faceted_domain_encoder.graph.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def load_data_file(path):
    # Expect file with one document per line
    logger.info('Load text file: %s', path)
    df = pd.read_csv(path, sep='\r\n', header=None)
    df.columns = ['text']
    return df


def load_embeddings(vocabulary, word_embedding_path, graph_embedding_path, padding_index=0):
    import fasttext
    from gensim.models import Word2Vec

    logger.info('Embed vocabulary with FastText and Node2Vec')
    fasttext_model = fasttext.load_model(word_embedding_path)
    node2vec_model = Word2Vec.load(graph_embedding_path)

    word_dimensions = 300
    graph_dimensions = 100
    word_embeddings = torch.Tensor(len(vocabulary), word_dimensions)
    graph_embeddings = torch.Tensor(len(vocabulary), graph_dimensions)

    for i, (token, entity) in enumerate(vocabulary):
        word_embeddings[i] = torch.from_numpy(fasttext_model.get_sentence_vector(token))

        if entity:
            x = torch.from_numpy(node2vec_model.wv[entity])
            x = x / x.norm(p=2)
            graph_embeddings[i] = x
        else:
            graph_embeddings[i] = torch.zeros(graph_dimensions)

    # Ensure padding index is zeroed out
    word_embeddings[padding_index] = torch.zeros(word_dimensions)
    graph_embeddings[padding_index] = torch.zeros(graph_dimensions)
    return word_embeddings, graph_embeddings


class Tokenizer:
    def __init__(self, stop=True, min_length=0):
        self.nlp = spacy.load('en')
        self.nlp.tokenizer.infix_finditer = self._get_infix_regex()
        self.stop = stop
        self.min_length = min_length
        self.punctuation = '!"$%&\'()*+,.:;<=>?@[\\]^_`{|}~'

    def __call__(self, sentence):
        sentence = sentence.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()
        document = self.nlp(sentence)
        tokens = []

        for token in document:
            if self._drop_token(token):
                continue

            lemma = self._lemmatize(token)
            lemma = lemma.lower()
            tokens.append(lemma)

        return tokens

    def _drop_token(self, token):
        return (self.stop and token.is_stop) \
               or (token.is_punct or token.is_space) \
               or (len(token.text) < self.min_length)

    def _lemmatize(self, token):
        # Spacy lemmas return -PRON- for `I`
        return token.lemma_ if token.lemma_ != '-PRON-' else token.text

    def _get_infix_regex(self):
        # Customize Spacy tokenization to NOT split words with hyphens
        # Source: https://spacy.io/usage/linguistic-features#native-tokenizers
        return compile_infix_regex(
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # EDIT: commented out regex that splits on hyphens between letters:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        ).finditer


class EntityLinker:
    def __init__(self, graph: KnowledgeGraph, ngrams=5):
        self.graph = graph
        self.ngrams = ngrams

    def __call__(self, sentence):
        num_tokens = len(sentence)
        document = []
        entities = []
        categories = []
        start_token = 0

        while start_token < num_tokens:
            max_end_token = min(start_token + self.ngrams, num_tokens)
            token = sentence[start_token]
            next_start_token = start_token + 1
            entity = None
            category = None

            for end_token in range(start_token, max_end_token + 1):
                ngram = ' '.join(sentence[start_token:end_token])

                if len(ngram) > 0 and ngram in self.graph.term2node:
                    token = ngram
                    next_start_token = end_token
                    entity = self.graph.term2node[ngram]
                    # FIXME: Only first category is recognized
                    entity_categories = self.graph.find_categories(entity)

                    if len(entity_categories) > 0:
                        category = entity_categories[0]

            document.append(token)
            entities.append(entity)
            categories.append(category)
            start_token = next_start_token

        return document, entities, categories


class Vocabulary:
    def __init__(self, max_length=256, categories=[]):
        self.token2index = {}
        self.categories = categories
        self.category2index = self._category2index(categories)
        self.index2category = {}
        self.tokens = ['<pad>']
        self.token_document_frequency = [0]
        self.token_frequency = [0]
        self.num_documents = 0
        self.entities = [None]
        self.max_length = max_length
        self._freeze = False

    def __call__(self, sentence_tokens, sentence_entities, sentence_categories):
        assert len(sentence_tokens) <= self.max_length, f'Maximum sentence length is {self.max_length}, but got: {len(sentence_tokens)}'
        sentence_length = min(len(sentence_tokens), self.max_length)
        embedding = torch.zeros(self.max_length).long()
        # Fill category vector with -1 for invalid category
        categories = torch.zeros(self.max_length).long() - 1

        for i, (token, entity, category) in enumerate(zip(sentence_tokens, sentence_entities, sentence_categories)):
            category_index = self.category2index[category] if category in self.category2index else -1

            if token not in self.token2index:
                assert not self._freeze, f'Vocabulary is frozen, cannot accept new token: {token}, {sentence_tokens}'
                index = len(self.tokens)
                self.tokens.append(token)
                self.entities.append(entity)
                self.token_document_frequency.append(0)
                self.token_frequency.append(0)
                self.token2index[token] = index
                self.index2category[index] = category_index
                assert len(self.tokens) == len(self.entities), 'Critical Error: Tokens and entities out of sync!'

            if i < self.max_length:
                index = self.token2index[token]
                embedding[i] = index
                categories[i] = category_index
                self.token_frequency[index] += 1

        for token in set(sentence_tokens):
            # Count unique token appearances in the document
            self.token_document_frequency[self.token2index[token]] += 1

        # Keep track of documents in vocabulary
        self.num_documents += 1

        return embedding, categories, sentence_length

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return zip(self.tokens.__iter__(), self.entities.__iter__())

    def freeze(self):
        self._freeze = True

    def _category2index(self, categories):
        category2index = {}

        for i, c in enumerate(categories):
            category2index[c] = i

        return category2index

    def category2tokens(self, category):
        return torch.tensor([t for t, c in self.index2category.items() if c == category])

    def index2sentence(self, embedding):
        return [self.tokens[i] for i in embedding], \
               [self.entities[i] for i in embedding]

    def index2idf(self, index):
        if index == 0:
            # Ignore padding
            return 0
        return np.log(self.num_documents / self.token_document_frequency[index])

    def get_idf(self):
        idf = torch.log(self.num_documents / torch.Tensor(self.token_document_frequency))
        idf[0] = 0
        return idf

    def load(self, path: str):
        with open(path, 'r') as f:
            json_load = json.loads(f.read())
            self.categories = json_load['categories']
            self.tokens = json_load['tokens']
            self.entities = json_load['entities']
            self.token_frequency = json_load['token_frequency']
            self.token_document_frequency = json_load['token_document_frequency']
            self.num_documents = json_load['num_documents']
            self.token2index = json_load['token2index']
            self.category2index = json_load['category2index']
            self.index2category = json_load['index2category']

            # JSON turns int keys to string during serialization.
            self.index2category = {int(k): int(v) for k, v in self.index2category.items()}

            self.freeze()

    def save(self, path: str):
        with open(path, 'w') as f:
            json_dump = json.dumps({
                'categories': self.categories,
                'tokens': self.tokens,
                'entities': self.entities,
                'token_frequency': self.token_frequency,
                'token_document_frequency': self.token_document_frequency,
                'num_documents': self.num_documents,
                'token2index': self.token2index,
                'category2index': self.category2index,
                'index2category': self.index2category
            })

            f.write(json_dump)


class TextProcessor:
    def __init__(self, graph, categories, max_length, stop):
        self.tokenizer = Tokenizer(stop)
        self.entity_linker = EntityLinker(graph)
        self.vocabulary = Vocabulary(max_length, categories)

    def __call__(self, sentence):
        tokens = self.tokenizer(sentence)
        tokens, entities, categories = self.entity_linker(tokens)
        indices, categories, length = self.vocabulary(tokens, entities, categories)
        return indices, categories, length

    def save(self, path: str):
        logger.info('Save vocabulary to: %s', path)
        self.vocabulary.save(path)

    def load(self, path: str):
        logger.info('Load vocabulary to: %s', path)
        self.vocabulary.load(path)
