from datetime import datetime

import torch
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from source.config import ExperimentConfig


def get_checkpoint_filename(cfg: ExperimentConfig) -> str:
    current_date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.model.generator_type == "classical":
        filename = f"best-checkpoint-classical-{current_date_time}"
    elif cfg.model.use_shadows:
        filename = f"best-checkpoint-quantum-shadows-{current_date_time}"
    else:
        filename = f"best-checkpoint-quantum-no-shadows-{current_date_time}"
    return filename


def prepare_device(cfg: ExperimentConfig) -> torch.device:
    return torch.device(
        "cuda"
        if cfg.general.accelerator == "gpu" and torch.cuda.is_available()
        else "cpu"
    )


def prepare_logger(cfg: ExperimentConfig, model):
    if cfg.general.logging_backend == "mlflow":
        logger = MLFlowLogger(experiment_name=cfg.general.experiment_name)
    elif cfg.general.logging_backend == "wandb":
        logger = WandbLogger(project=cfg.general.experiment_name)
        logger.watch(model, log="all", log_freq=100)
    else:
        logger = None
    return logger


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        loaded = pickle.load(f)
    print(f"Pickle loaded from {filepath}")
    return loaded


def save_pickle(instance, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(instance, f)
    print(f"Pickle saved to {filepath}")
