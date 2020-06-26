import logging

import hydra

from experiments.ablation.util.ablation_study import ablation_study, sample_documents
from experiments.ablation.util.data import EmailDataset
from experiments.util.training import train_model

logger = logging.getLogger(__name__)

""""
Rolls-Royce Email Support Dataset
Authors: -
Dataset: -
"""


def prepare_datasets(config):
    dataset = EmailDataset(
        hydra.utils.to_absolute_path(config.data.raw_path),
        hydra.utils.to_absolute_path(config.data.train_path),
        hydra.utils.to_absolute_path(config.data.test_path)
    )
    train_df, test_df = dataset.load()
    return train_df, test_df


@hydra.main('../../config', 'aviation_email_ablation_config.yaml')
def experiment(config):
    prepare_datasets(config)
    model = train_model(config)

    logger.info('Test Set ablation study')
    logger.info('Sampling %s documents per graph category for ablation study', config.ablation.num_samples)
    df = sample_documents(config, model.test_df, config.ablation.num_samples)
    ablation_df = ablation_study(model, config, df)
    logger.info(ablation_df.groupby('ablation_category').mean().to_string())

    logger.info('Test Set ablation study (Unique Words)')
    logger.info('Sampling %s documents per graph category for ablation study', config.ablation.num_samples)
    df = sample_documents(config, model.test_df, config.ablation.num_samples)
    ablation_df = ablation_study(model, config, df, unique_tokens=True)
    logger.info(ablation_df.groupby('ablation_category').mean().to_string())

    if config.ablation.distance_map:
        logger.info('Distance: Mean Average Precision', ablation_df['distance_map'].mean())
        logger.info('Distance: Median Mean Average Precision', ablation_df['distance_map'].median())
    if config.ablation.attention_map:
        logger.info('Attention: Mean Average Precision', ablation_df['attention_map'].median())
        logger.info('Attention: Median Mean Average Precision', ablation_df['attention_map'].median())

    logger.info('Mean tokens in category:', ablation_df.groupby('ablation_category').num_tokens_in_category.mean())
    logger.info('Median tokens in category:', ablation_df.groupby('ablation_category').num_tokens_in_category.median())
    logger.info('Max tokens in category:', ablation_df.groupby('ablation_category').num_tokens_in_category.max())
    logger.info('Documents in category: %s', ablation_df.groupby('ablation_category').size())


if __name__ == '__main__':
    experiment()
