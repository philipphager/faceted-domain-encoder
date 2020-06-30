import logging

import hydra

from experiments.ablation.util.ablation_study import ablation_study, sample_documents
from experiments.classification.util.data import OhsumedDataset
from experiments.util.training import train_model

logger = logging.getLogger(__name__)

""""
OHSUMED 20000 Classification
Authors: William Hersh, Chris Buckley, T. J. LeoneDavid Hickam
Dataset: Split of Joachims et al.: http://disi.unitn.eu/moschitti/corpora.htm
"""


def prepare_datasets(config):
    train_df = OhsumedDataset(
        hydra.utils.to_absolute_path(config.data.raw_train_path),
        hydra.utils.to_absolute_path(config.data.train_path)
    ).load()

    test_df = OhsumedDataset(
        hydra.utils.to_absolute_path(config.data.raw_test_path),
        hydra.utils.to_absolute_path(config.data.test_path)
    ).load()

    return train_df, test_df


@hydra.main('../../config', 'ohsumed_ablation_config.yaml')
def experiment(config):
    prepare_datasets(config)
    model = train_model(config)

    logger.info('Test set ablation study')
    logger.info('Sampling %s documents per graph category for ablation study', config.ablation.num_samples)
    df = sample_documents(config, model.test_df, config.ablation.num_samples)
    ablation_df = ablation_study(model, config, df)
    ablation_df.to_csv('ablation.csv')
    logger.info(ablation_df.groupby('ablation_category').mean().to_string())

    logger.info('Test set ablation study (Unique Words)')
    logger.info('Sampling %s documents per graph category for ablation study', config.ablation.num_samples)
    df = sample_documents(config, model.test_df, config.ablation.num_samples)
    ablation_df = ablation_study(model, config, df, unique_tokens=True)
    ablation_df.to_csv('ablation_unique.csv')
    logger.info(ablation_df.groupby('ablation_category').mean().to_string())

    if config.ablation.distance_map:
        logger.info('Distance: Mean Average Precision: %s', ablation_df['distance_map'].mean())
        logger.info('Distance: Median Mean Average Precision: %s', ablation_df['distance_map'].median())
    if config.ablation.attention_map:
        logger.info('Attention: Mean Average Precision: %s', ablation_df['attention_map'].median())
        logger.info('Attention: Median Mean Average Precision: %s', ablation_df['attention_map'].median())


if __name__ == '__main__':
    experiment()
