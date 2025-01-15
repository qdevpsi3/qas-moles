from functools import partial

import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .config import ExperimentConfig
from .data import MolecularDataModule, MolecularDataset
from .model import MolGAN
from .nn import GNNDiscriminator, MolGANDiscriminator, MolGANGenerator, QMolGANGenerator
from .utils import get_checkpoint_filename, prepare_device, prepare_logger

generators = {
    "molgan": MolGANGenerator(
        z_dim=8,
        conv_dims=(64, 128, 256),
    ),
    "qmolgan": QMolGANGenerator(
        num_circuit_qubits=8,
        num_circuit_layers=3,
        conv_dims=(64, 128, 256),
    ),
}

discriminators = {
    "molgan": MolGANDiscriminator(
        conv_dims=((128, 64), 128, (128, 64)),
        with_features=False,
    ),
    "gnn_v0": GNNDiscriminator(),
}

predictors = {"molgan": discriminators["molgan"]}


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = prepare_device(cfg)

    def setup_generator(self):
        generator = generators[self.cfg.model.generator_type]
        generator.build(
            num_vertices=self.num_vertices,
            num_bond_types=self.num_bond_types,
            num_atom_types=self.num_atom_types,
        )
        generator.to(self.device)
        self.generator = generator

    def setup_discriminator(self):
        discriminator = discriminators[self.cfg.model.discriminator_type]
        discriminator.build(
            num_vertices=self.num_vertices,
            num_bond_types=self.num_bond_types,
            num_atom_types=self.num_atom_types,
            num_metrics=1,
        )
        discriminator.to(self.device)
        self.discriminator = discriminator

    def setup_predictor(self):
        predictor = predictors[self.cfg.model.predictor_type]
        predictor.build(
            num_vertices=self.num_vertices,
            num_bond_types=self.num_bond_types,
            num_atom_types=self.num_atom_types,
            num_metrics=1,
        )
        predictor.to(self.device)
        self.predictor = predictor

    def setup_model(self):
        model = MolGAN(
            dataset=self.dataset,
            generator=self.generator,
            discriminator=self.discriminator,
            predictor=self.predictor,
            optimizer=self.optimizer,
            grad_penalty=self.cfg.training.grad_penalty,
            process_method=self.cfg.training.process_method,
            agg_method=self.cfg.training.agg_method,
            train_predictor_on_fake=self.cfg.training.train_predictor_on_fake,
            n_critic=self.cfg.training.n_critic,
        )
        # model.to(self.device)

        if self.cfg.training.checkpoint_path is not None:
            ckpt_path = (
                f"{self.cfg.general.save_dir}/{self.cfg.training.checkpoint_path}.ckpt"
            )
            model = MolGAN.load_from_checkpoint(
                ckpt_path,
                generator=self.generator,
                discriminator=self.discriminator,
                predictor=self.predictor,
                optimizer=self.optimizer,
            )
            model.to(self.device)
        self.model = model
        return model

    def setup_datamodule(self):
        """Setup the data module."""
        self.dataset = MolecularDataset.load(self.cfg.data.data_path)
        self.datamodule = MolecularDataModule(
            self.dataset,
            train_batch_size=self.cfg.data.train_batch_size,
            test_batch_size=self.cfg.data.test_batch_size,
        )
        self.datamodule.setup()
        self.num_vertices = self.dataset.num_vertices
        self.num_bond_types = self.dataset.bond_num_types
        self.num_atom_types = self.dataset.atom_num_types

    def setup_optimizer(self):
        """Setup the optimizer."""
        self.optimizer = partial(
            torch.optim.RMSprop, lr=self.cfg.training.learning_rate
        )

    def setup(
        self,
    ):
        self.setup_datamodule()
        self.setup_generator()
        self.setup_discriminator()
        self.setup_predictor()
        self.setup_optimizer()
        self.setup_model()

    def run_train(self):
        """Train the model."""
        if self.cfg.general.logging_backend == "mlflow":
            mlflow.pytorch.autolog(checkpoint_save_best_only=False)

        self.setup()

        # checkpoint_filename = get_checkpoint_filename(self.cfg)
        # checkpoint_callback = ModelCheckpoint(
        #     monitor="Aggregated_metric_during_validation",
        #     save_top_k=1,
        #     mode="max",
        #     dirpath=self.cfg.general.save_dir,
        #     filename=checkpoint_filename,
        #     save_last=True,
        # )

        # logger = prepare_logger(self.cfg, self.model)
        trainer = Trainer(
            max_epochs=self.cfg.training.max_epochs,
            accelerator=self.cfg.general.accelerator,
            # logger=logger,
            # limit_train_batches=10,
            limit_test_batches=1,
            limit_val_batches=1,
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
        )
        print(self.model)
        trainer.fit(model=self.model, datamodule=self.datamodule)

    def run_test(self):
        """Test the model."""
        if self.cfg.training.checkpoint_path is None:
            raise ValueError("Checkpoint path required for testing.")

        logger = prepare_logger(self.cfg, self.model)
        trainer = Trainer(
            accelerator=self.cfg.general.accelerator,
            logger=logger,
        )
        trainer.test(model=self.model, datamodule=self.datamodule)

    def run_generate(self):
        """Generate samples using the trained model."""
        model, _ = self.setup()
        if self.cfg.training.checkpoint_path is None:
            raise ValueError("Checkpoint path required for generation.")

        num_samples = self.cfg.generation.num_samples
        generated_molecules = model.generate_samples(
            num_samples=num_samples, device=self.device
        )
        print("Generated molecules:", generated_molecules)

    def run(self):
        """Run the experiment based on the stage in the configuration."""
        if self.cfg.general.stage == "train":
            self.run_train()
        elif self.cfg.general.stage == "test":
            self.run_test()
        elif self.cfg.general.stage == "generate":
            self.run_generate()
        else:
            raise ValueError(f"Unknown stage {self.cfg.general.stage}")
