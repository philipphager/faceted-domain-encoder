import csv
import logging
import os

import hydra
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer

from experiments.classification.util.classifier import Classifier
from experiments.classification.util.util import embed
from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder

logger = logging.getLogger(__name__)

""""
Cancer Hallmarks Classification, 10 Binary Classifiers
Authors: Simon Baker, Ilona Silins, Yufan Guo, Imran Ali, Johan Hogberg, Ulla Stenius, Anna Korhonen
Dataset: https://github.com/cambridgeltl/cancer-hallmark-cnn 
"""


def get_dataset(base_dir, label_dir, dataset):
    class_directory = os.path.join(base_dir, label_dir)

    pos_df = pd.read_csv(os.path.join(class_directory, dataset + '.pos'), header=None, sep='\n')
    pos_df['label'] = 'pos'

    neg_df = pd.read_csv(os.path.join(class_directory, dataset + '.neg'), header=None, sep='\n')
    neg_df['label'] = 'neg'

    # Join positive and negative datasets and shuffle
    df = pd.concat((pos_df, neg_df))
    df = df.sample(len(df), random_state=42)
    df.columns = ['document', 'label']
    return df


def to_txt(df, output_file):
    df[['document']].drop_duplicates().to_csv(
        output_file,
        sep='\t',
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        encoding='utf-8')


def train_model(config):
    use_gpu(2)

    trainer = Trainer(
        gpus=config.trainer.gpus,
        max_epochs=config.trainer.max_epochs,
        early_stop_callback=EarlyStopping(),
    )

    model = FacetedDomainEncoder(config)
    trainer.fit(model)
    model.cpu()
    return model


def classify(model, train_df, val_df, test_df, label):
    encoder = LabelBinarizer()
    encoder.fit(train_df.label.values)

    X_train = embed(model, train_df)
    X_val = embed(model, val_df)
    X_test = embed(model, test_df)
    y_train = torch.from_numpy(encoder.transform(train_df.label.values)).float()
    y_val = torch.from_numpy(encoder.transform(val_df.label.values)).float()
    y_test = torch.from_numpy(encoder.transform(test_df.label.values)).float()

    classifier = Classifier(X_train, y_train, X_val, y_val, X_test, y_test, 512, 128, 1)
    trainer = Trainer(early_stop_callback=EarlyStopping(monitor='val_loss', patience=2))
    trainer.fit(classifier)

    classifier.eval()
    y_predict = classifier(X_test).detach()
    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict < 1] = 0

    logger.info('Hallmarks Classification: %s', label)
    logger.info(classification_report(y_test, y_predict, target_names=encoder.classes_))
    logger.info('F1: %s', f1_score(y_test, y_predict))
    logger.info('ROC-AUC: %s', roc_auc_score(y_test, y_predict))


@hydra.main('../../config', 'hallmarks_config.yaml')
def experiment(config):
    base_dir = hydra.utils.to_absolute_path(config.data.raw_path)
    train_path = hydra.utils.to_absolute_path(config.data.train_path)
    test_path = hydra.utils.to_absolute_path(config.data.test_path)

    labels = sorted(os.listdir(base_dir))

    # Collect all files and train model on all official training files
    train_dfs = []
    test_dfs = []

    for label in labels:
        train_dfs.append(get_dataset(base_dir, label, 'train'))
        train_dfs.append(get_dataset(base_dir, label, 'devel'))
        test_dfs.append(get_dataset(base_dir, label, 'test'))

    train_dfs = pd.concat(train_dfs)
    test_dfs = pd.concat(test_dfs)
    to_txt(train_dfs, train_path)
    to_txt(test_dfs, test_path)

    model = train_model(config)

    # Train binary classifier for each label
    for label in labels:
        logger.info('Classifying: %s', label)

        train_df = get_dataset(base_dir, label, 'train')
        val_df = get_dataset(base_dir, label, 'devel')
        test_df = get_dataset(base_dir, label, 'test')

        classify(model, train_df, val_df, test_df, label)


if __name__ == '__main__':
    experiment()
