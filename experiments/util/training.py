import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from experiments.util.env import use_gpu
from faceted_domain_encoder import FacetedDomainEncoder


def clear_checkpoint_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for f in os.listdir(path):
            if f.endswith('.ckpt'):
                os.remove(os.path.join(path, f))


def get_model_path(path):
    files = os.listdir(path)
    file = [f for f in files if f.endswith('.ckpt')][0]
    return os.path.join(path, file)


def train_model(config):
    use_gpu(3)

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

    # Load best model
    path = get_model_path(checkpoint_dir)
    model = FacetedDomainEncoder.load_from_checkpoint(path)

    model.cpu()
    return model
