import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from experiments.baseline.Doc2VecEncoder import Doc2VecEncoder
from experiments.baseline.FastTextEncoder import FastTextEncoder
from experiments.baseline.Node2VecEncoder import Node2VecEncoder
from experiments.sentence_similarity.util import to_file, sentence_similarity, pearson_correlation

logger = logging.getLogger(__name__)

""""
MedSTS Benchmark
Authors: Yanshan Wang, Naveed Afzal, Sunyang Fu, Liwei Wang, Feichen Shen, Majid Rastegar-Mojarad, Hongfang Liu
Paper: https://arxiv.org/ftp/arxiv/papers/1808/1808.09397.pdf 
"""


def prepare_train(in_path, out_path):
    frame = pd.read_csv(in_path, sep='\t', header=None)
    frame = frame.reset_index()
    frame.columns = ['pair_id', 'sentence_1', 'sentence_2', 'score']
    to_file(frame, out_path)
    return frame


def prepare_test(in_path, score_path, out_path):
    frame = pd.read_csv(in_path, sep='\t', header=None)
    frame['score'] = pd.read_csv(score_path, sep='\t', header=None)
    frame = frame.reset_index()
    frame.columns = ['pair_id', 'sentence_1', 'sentence_2', 'score']
    to_file(frame, out_path)
    return frame


def train_doc2vec(train_df):
    model = Doc2VecEncoder()
    model.fit(train_df)
    return model


def train_fasttext(config, pretrained=False):
    model = FastTextEncoder(pretrained=pretrained)
    model.fit(config)
    return model


def train_node2vec(config):
    model = Node2VecEncoder()
    model.fit(config)
    return model


def report(model, train_df, test_df, name):
    train_df['similarity'] = sentence_similarity(model, train_df).numpy()
    test_df['similarity'] = sentence_similarity(model, test_df).numpy()

    train_correlation = pearson_correlation(train_df.similarity, train_df.score)
    test_correlation = pearson_correlation(test_df.similarity, test_df.score)

    logger.info('Baseline %s', name)
    logger.info('MedSTS Train correlation %s', train_correlation)
    logger.info('MedSTS Test correlation %s', test_correlation)


@hydra.main('../../config', 'medsts_config.yaml')
def experiment(config: DictConfig):
    train_in_path = hydra.utils.to_absolute_path(config.data.raw_train_path)
    test_in_path = hydra.utils.to_absolute_path(config.data.raw_test_path)
    test_score_path = hydra.utils.to_absolute_path(config.data.raw_test_score_path)
    train_out_path = hydra.utils.to_absolute_path(config.data.train_path)
    test_out_path = hydra.utils.to_absolute_path(config.data.test_path)

    train_df = prepare_train(train_in_path, train_out_path)
    test_df = prepare_test(test_in_path, test_score_path, test_out_path)

    model = train_doc2vec(train_df)
    report(model, train_df, test_df, 'Doc2Vec')

    model = train_fasttext(config)
    report(model, train_df, test_df, 'FastText')

    model = train_fasttext(config, pretrained=True)
    report(model, train_df, test_df, 'FastText-Pretrained')

    model = train_node2vec(config)
    report(model, train_df, test_df, 'Node2Vec')


if __name__ == '__main__':
    experiment()
