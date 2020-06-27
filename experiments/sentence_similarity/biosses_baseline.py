import logging

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from experiments.baseline.Doc2VecEncoder import Doc2VecEncoder
from experiments.baseline.FastTextEncoder import FastTextEncoder
from experiments.baseline.Node2VecEncoder import Node2VecEncoder
from experiments.sentence_similarity.util import to_file, sentence_similarity, pearson_correlation

logger = logging.getLogger(__name__)

""""
BIOSSES Benchmark
Authors: Gizem Soğancıoğlu, Hakime Öztürk, Arzucan Özgür
Paper: https://academic.oup.com/bioinformatics/article/33/14/i49/3953954
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


def train_model(config, train_df, name):
    if name == 'Node2Vec':
        return train_node2vec(config)
    elif name == 'FastText':
        return train_fasttext(config)
    elif name == 'FastText-Pretrained':
        return train_fasttext(config, True)
    elif name == 'Doc2Vec':
        return train_doc2vec(train_df)


def report(config, name):
    df = pd.read_csv(hydra.utils.to_absolute_path(config.data.raw_path))
    df.columns = ['pair_id', 'sentence_1', 'sentence_2', 'a_1', 'a2', 'a3', 'a4', 'a5', 'score']

    df['score_bin'] = df.score.map(lambda x: int(x))
    results = []
    cross_validation = StratifiedKFold(n_splits=10, random_state=config.data.random_state)

    for i, (train_index, test_index) in enumerate(cross_validation.split(df, df.score_bin.values)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        to_file(train_df, hydra.utils.to_absolute_path(config.data.train_path))
        to_file(test_df, hydra.utils.to_absolute_path(config.data.test_path))

        model = train_model(config, train_df, name)

        train_df['similarity'] = sentence_similarity(model, train_df)
        test_df['similarity'] = sentence_similarity(model, test_df)

        train_correlation = pearson_correlation(train_df.similarity, train_df.score)
        test_correlation = pearson_correlation(test_df.similarity, test_df.score)
        logger.info('Baseline %s', name)
        logger.info('Cross Validation Split %s', i)
        logger.info('Train correlation %s', train_correlation)
        logger.info('Test correlation %s', test_correlation)
        results.append({'train': train_correlation, 'test': test_correlation})

    result_df = pd.DataFrame(results)

    logger.info('Baseline %s', name)
    logger.info('BIOSSES Train correlation %s', result_df.train.mean())
    logger.info('BIOSSES Test correlation %s', result_df.test.mean())


@hydra.main('../../config', 'biosses_config.yaml')
def experiment(config: DictConfig):
    report(config, 'Doc2Vec')
    report(config, 'FastText')
    report(config, 'FastText-Pretrained')
    report(config, 'Node2Vec')


if __name__ == '__main__':
    experiment()
