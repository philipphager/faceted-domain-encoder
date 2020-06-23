import pandas as pd
from hydra import experimental
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks

from experiments.sentence_similarity.util import to_file, sentence_similarity, pearson_correlation
from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder

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

def train_model(config):
    use_gpu(3)

    trainer = Trainer(
        gpus=config.trainer.gpus,
        max_epochs=config.trainer.max_epochs,
        early_stop_callback=callbacks.EarlyStopping(),
    )

    model = FacetedDomainEncoder(config)
    trainer.fit(model)
    model.cpu()
    return model


def experiment():
    # experimental.initialize(config_path='../config')
    config = experimental.compose('medicine_config.yaml', overrides=[
        'dataset=medsts',
        'normalizer=corpus'
    ])
    print(config.pretty())

    train_in_path = 'data/experiment/sentence_similarity/medsts/clinicalSTS.train.txt'
    test_in_path = 'data/experiment/sentence_similarity/medsts/clinicalSTS.test.txt'
    test_score_path = 'data/experiment/sentence_similarity/medsts/clinicalSTS.test.gs.sim.txt'
    train_out_path = config.data.train_path
    test_out_path = config.data.test_path

    train_df = prepare_train(train_in_path, train_out_path)
    test_df = prepare_test(test_in_path, test_score_path, test_out_path)

    model = train_model(config)

    train_df['similarity'] = sentence_similarity(model, train_df)
    test_df['similarity'] = sentence_similarity(model, test_df)

    train_correlation = pearson_correlation(train_df.similarity, train_df.score)
    test_correlation = pearson_correlation(test_df.similarity, test_df.score)

    logger.info('Train correlation {}', train_correlation)
    logger.info('Test correlation {}', test_correlation)
    return train_correlation, test_correlation


if __name__ == '__main__':
    experiment()
