import logging

import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from experiments.classification.util.classifier import Classifier
from experiments.classification.util.data import OhsumedDataset, HallmarksDataset
from experiments.classification.util.util import embed
from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder

logger = logging.getLogger(__name__)

""""
Cancer Hallmarks Classification
Authors: Simon Baker, Ilona Silins, Yufan Guo, Imran Ali, Johan Hogberg, Ulla Stenius, Anna Korhonen
Dataset: https://github.com/sb895/Hallmarks-of-Cancer 
"""


def prepare_datasets(config):
    return HallmarksDataset(
        hydra.utils.to_absolute_path(config.data.raw_text_path),
        hydra.utils.to_absolute_path(config.data.raw_label_path),
        hydra.utils.to_absolute_path(config.data.train_path),
        hydra.utils.to_absolute_path(config.data.test_path),
    ).load()


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


def classify(model, train_df, test_df):
    encoder = MultiLabelBinarizer()
    encoder.fit(train_df.label.values)
    num_classes = len(encoder.classes_)

    X_train = embed(model, train_df)
    X_test = embed(model, test_df)
    y_train = torch.from_numpy(encoder.transform(train_df.label.values)).float()
    y_test = torch.from_numpy(encoder.transform(test_df.label.values)).float()

    classifier = Classifier(X_train, y_train, X_test, y_test, 512, 128, num_classes)
    trainer = Trainer(early_stop_callback=EarlyStopping(monitor='val_loss', patience=2))
    trainer.fit(classifier)

    classifier.eval()
    y_predict = classifier(X_test).detach()
    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict < 1] = 0

    logger.info(classification_report(y_test, y_predict, target_names=encoder.classes_))


@hydra.main('../../config', 'hallmarks_config.yaml')
def experiment(config):
    train_df, test_df = prepare_datasets(config)
    model = train_model(config)
    classify(model, train_df, test_df)


if __name__ == '__main__':
    experiment()
