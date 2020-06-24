import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from sklearn.model_selection import StratifiedKFold

from experiments.sentence_similarity.util import to_file, sentence_similarity, pearson_correlation
from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder

logger = logging.getLogger(__name__)

""""
BIOSSES Benchmark
Authors: Gizem Soğancıoğlu, Hakime Öztürk, Arzucan Özgür
Paper: https://academic.oup.com/bioinformatics/article/33/14/i49/3953954
"""


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


@hydra.main('../../config', 'biosses_config.yaml')
def experiment(config: DictConfig):
    print(os.getcwd())
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

        model = train_model(config)

        train_df['similarity'] = sentence_similarity(model, train_df)
        test_df['similarity'] = sentence_similarity(model, test_df)

        train_correlation = pearson_correlation(train_df.similarity, train_df.score)
        test_correlation = pearson_correlation(test_df.similarity, test_df.score)
        logger.info('Cross Validation Split %s', i)
        logger.info('Train correlation %s', train_correlation)
        logger.info('Test correlation %s', test_correlation)
        results.append({'train': train_correlation, 'test': test_correlation})

    result_df = pd.DataFrame(results)

    logger.info('Encoder %s', config.model.encoder)
    logger.info('Pooling %s', config.model.pooling)
    logger.info('Normalizer %s', config.model.normalizer)
    logger.info('BIOSSES Train correlation %s', result_df.train.mean())
    logger.info('BIOSSES Test correlation %s', result_df.test.mean())


if __name__ == '__main__':
    experiment()
