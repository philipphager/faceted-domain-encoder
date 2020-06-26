import logging
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder

logger = logging.getLogger(__name__)


def clear_checkpoint_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for f in os.listdir(path):
            if f.endswith('.ckpt'):
                os.remove(os.path.join(path, f))


def get_last_checkpoint(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.ckpt')]
    file = files[0]

    if len(files) > 0:
        logger.warning('Expected a single model in checkpoint directory! Parallel executions?')

    logger.info('Loading best model: %s', file)
    return os.path.join(path, file)


def train_model(config, gpu=3):
    use_gpu(gpu)

    # Ensure reproducibility
    seed_everything(42)

    checkpoint_dir = os.path.join(os.getcwd(), 'model')
    clear_checkpoint_dir(checkpoint_dir)
    checkpoint = ModelCheckpoint(filepath=checkpoint_dir)
    early_stopping = EarlyStopping(patience=config.trainer.patience)

    trainer = Trainer(
        gpus=config.trainer.gpus,
        max_epochs=config.trainer.max_epochs,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint,
        deterministic=True
    )

    # Train model
    model = FacetedDomainEncoder(config)
    trainer.fit(model)
    train_df = model.train_df
    validation_df = model.validation_df
    test_df = model.test_df

    # Load best model
    path = get_last_checkpoint(checkpoint_dir)
    model = FacetedDomainEncoder.load_from_checkpoint(path)
    # Keep preprocessed documents from original model
    model.train_df = train_df
    model.validation_df = validation_df
    model.test_df = test_df

    model.cpu()
    return model
