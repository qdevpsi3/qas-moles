import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class GeneralConfig:
    stage: str = "train"  # "train", "test"
    accelerator: str = "cpu"  # "cpu", "gpu"
    experiment_name: str = "molgan"
    logging_backend: str = "mlflow"  # "mlflow", "wandb"
    save_dir: str = "checkpoints"


@dataclass
class DataConfig:
    data_path: str = "./data/gdb9_molecular_dataset.pkl"
    batch_size: int = 32


@dataclass
class ModelConfig:
    generator_type: str = "molgan"  # "classical", "quantum"
    discriminator_type: str = "molgan"  # "classical", "quantum"
    predictor_type: str = "molgan"  # "classical", "quantum"


@dataclass
class TrainingConfig:
    max_epochs: int = 200
    learning_rate: float = 0.001
    grad_penalty: float = 10.0
    process_method: str = "soft_gumbel"
    agg_method: str = "prod"
    train_predictor_on_fake: bool = False
    n_critic: int = 5
    checkpoint_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_file(config_file: Optional[str]) -> Dict[str, Any]:
        if config_file is None or not os.path.exists(config_file):
            return {}
        if config_file.endswith((".yml", ".yaml")):
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        elif config_file.endswith(".json"):
            with open(config_file, "r") as f:
                return json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON.")

    def update_from_dict(self, updates: Dict[str, Any]):
        # updates is a nested dictionary
        for key, val in updates.items():
            if hasattr(self, key) and dataclasses.is_dataclass(getattr(self, key)):
                nested_obj = getattr(self, key)
                if isinstance(val, dict):
                    for nk, nv in val.items():
                        if hasattr(nested_obj, nk):
                            setattr(nested_obj, nk, nv)
            else:
                setattr(self, key, val)

    def save_to_file(self, filepath: str):
        data = dataclasses.asdict(self)
        if filepath.endswith(".yml") or filepath.endswith(".yaml"):
            with open(filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif filepath.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml.")
