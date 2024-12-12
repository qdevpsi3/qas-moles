from functools import partial

import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .config import ExperimentConfig
from .data import MolecularDataModule, MolecularDataset
from .model import MolGAN
from .nn import Discriminator, Generator, QuantumGenerator
from .utils import get_checkpoint_filename, prepare_device, prepare_logger


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = prepare_device(cfg)

    def setup_models(self):
        dataset = MolecularDataset.load(self.cfg.data.data_path)

        if self.cfg.model.generator_type == "classical":
            G = Generator(dataset, z_dim=self.cfg.model.z_dim)
        else:
            G = QuantumGenerator(
                dataset,
                use_shadows=self.cfg.model.use_shadows,
                z_dim=self.cfg.model.z_dim,
            )

        D = Discriminator(dataset)
        V = Discriminator(dataset)
        G.to(self.device)
        D.to(self.device)
        V.to(self.device)

        model = MolGAN(
            dataset,
            G,
            D,
            V,
            optimizer=partial(torch.optim.RMSprop, lr=self.cfg.training.learning_rate),
            grad_penalty=self.cfg.training.grad_penalty,
            process_method=self.cfg.training.process_method,
            agg_method=self.cfg.training.agg_method,
            train_predictor_on_fake=self.cfg.training.train_predictor_on_fake,
            n_critic=self.cfg.training.n_critic,
        )
        model.to(self.device)

        if self.cfg.training.checkpoint_path is not None:
            ckpt_path = (
                f"{self.cfg.general.save_dir}/{self.cfg.training.checkpoint_path}.ckpt"
            )
            model = MolGAN.load_from_checkpoint(
                ckpt_path,
                dataset=dataset,
                generator=G,
                discriminator=D,
                predictor=V,
                optimizer=partial(
                    torch.optim.RMSprop, lr=self.cfg.training.learning_rate
                ),
            )
            model.to(self.device)

        return dataset, model

    def setup_datamodule(self, dataset):
        datamodule = MolecularDataModule(dataset, batch_size=self.cfg.data.batch_size)
        datamodule.setup()
        return datamodule

    def run_train(self):
        if self.cfg.general.logging_backend == "mlflow":
            mlflow.pytorch.autolog(checkpoint_save_best_only=False)

        dataset, model = self.setup_models()
        datamodule = self.setup_datamodule(dataset)

        checkpoint_filename = get_checkpoint_filename(self.cfg)
        checkpoint_callback = ModelCheckpoint(
            monitor="Aggregated_metric_during_validation",
            save_top_k=1,
            mode="max",
            dirpath=self.cfg.general.save_dir,
            filename=checkpoint_filename,
            save_last=True,
        )

        logger = prepare_logger(self.cfg, model)
        trainer = Trainer(
            max_epochs=self.cfg.training.max_epochs,
            accelerator=self.cfg.general.accelerator,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model=model, datamodule=datamodule)

    def run_test(self):
        dataset, model = self.setup_models()
        datamodule = self.setup_datamodule(dataset)
        if self.cfg.training.checkpoint_path is None:
            raise ValueError("Checkpoint path required for testing.")

        logger = prepare_logger(self.cfg, model)
        trainer = Trainer(
            accelerator=self.cfg.general.accelerator,
            logger=logger,
        )
        trainer.test(model=model, datamodule=datamodule)

    def run_generate(self):
        dataset, model = self.setup_models()
        if self.cfg.training.checkpoint_path is None:
            raise ValueError("Checkpoint path required for generation.")

        num_samples = 10
        generated_molecules = model.generate_samples(
            num_samples=num_samples, device=self.device
        )
        print("Generated molecules:", generated_molecules)

    def run(self):
        if self.cfg.general.stage == "train":
            self.run_train()
        elif self.cfg.general.stage == "test":
            self.run_test()
        elif self.cfg.general.stage == "generate":
            self.run_generate()
        else:
            raise ValueError(f"Unknown stage {self.cfg.general.stage}")
