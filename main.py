import argparse
from datetime import datetime
from functools import partial

import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from source.data import MolecularDataModule, MolecularDataset
from source.model import MolGAN
from source.nn import Discriminator, Generator, QuantumGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the MolGAN model on the gdb9 dataset"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        help="Stage to run ('train', 'test')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--generator_type",
        type=str,
        default="classical",
        help="Type of generator to use ('classical', 'quantum')",
    )
    parser.add_argument(
        "--use_shadows",
        type=bool,
        default=False,  # or shadow
        help="Use shadow noise generator for the quantum generator",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/gdb9_molecular_dataset.pkl",
        help="Path to the molecular dataset file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--grad_penalty",
        type=float,
        default=10.0,
        help="Gradient penalty regularization factor",
    )
    parser.add_argument(
        "--process_method",
        type=str,
        default="soft_gumbel",
        help="Method to process the output probabilities ('soft_gumbel', 'hard_gumbel')",
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="prod",
        help="Aggregation method for the rewards.",
    )
    parser.add_argument(
        "--train_predictor_on_fake",
        type=bool,
        default=False,
        help="Train the predictor on fake samples",
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="Number of discriminator updates per generator update",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=8,
        help="Dimension of the latent space",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )

    # Enable MLflow autologging
    mlflow.pytorch.autolog(checkpoint_save_best_only=False)

    # Load data module
    dataset = MolecularDataset.load(args.data_path)
    datamodule = MolecularDataModule(dataset, batch_size=args.batch_size)
    datamodule.setup()

    # Initialize networks
    if args.generator_type == "classical":
        G = Generator(
            dataset,
            z_dim=args.z_dim,
        )
    elif args.generator_type == "quantum":
        G = QuantumGenerator(
            dataset,
            use_shadows=args.use_shadows,
            z_dim=args.z_dim,
        )
    D = Discriminator(dataset)
    V = Discriminator(dataset)
    G.to(device)
    D.to(device)
    V.to(device)
    print("Nets created")

    # Setup the MolGAN model
    model = MolGAN(
        dataset,
        G,
        D,
        V,
        optimizer=partial(torch.optim.RMSprop, lr=args.learning_rate),
        grad_penalty=args.grad_penalty,
        process_method=args.process_method,
        agg_method=args.agg_method,
        train_predictor_on_fake=args.train_predictor_on_fake,
        n_critic=args.n_critic,
    )
    model.to(device)
    if args.checkpoint_path is not None:
        # Load the best model
        model = MolGAN.load_from_checkpoint(
            "checkpoints/" + args.checkpoint_path + ".ckpt",
            dataset=dataset,
            generator=G,
            discriminator=D,
            predictor=V,
            optimizer=partial(torch.optim.RMSprop, lr=args.learning_rate),
        )
        model.to(device)
    # Define the checkpoint callback
    current_date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.generator_type == "classical":
        filename = f"best-checkpoint-classical-{current_date_time}"
    elif args.use_shadows:
        filename = f"best-checkpoint-quantum-shadows-{current_date_time}"
    else:
        filename = f"best-checkpoint-quantum-no-shadows-{current_date_time}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        save_top_k=1,
        mode="max",
        dirpath="checkpoints/",
        filename=filename,
        save_last=True,
    )

    # Setup the trainer with MLflow logging and checkpoint callback
    mlflow_logger = MLFlowLogger(experiment_name="molgan")
    # wandb_logger = WandbLogger(
    #     project="molgan",  # Change project name as required
    #     name=f"molgan-run-{current_date_time}",
    # )
    # wandb_logger.watch(model, log="all", log_freq=100)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=mlflow_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    if args.stage == "train":
        # Start training
        trainer.fit(model=model, datamodule=datamodule)
    elif args.stage == "test":
        # Start testing
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
