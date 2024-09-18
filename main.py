import argparse
from functools import partial

from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim import Adam

from source.datasets import MolecularDataModule, MolecularDataset
from source.model import MolGAN
from source.nets import Discriminator, Generator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the MolGAN model on the gdb9 dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/gdb9_molecular_dataset.pkl",
        help="Path to the molecular dataset file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data module
    dataset = MolecularDataset.load(args.data_path)
    datamodule = MolecularDataModule(dataset, batch_size=args.batch_size)
    datamodule.setup()

    # Initialize nets
    G = Generator(dataset)
    D = Discriminator(dataset)
    V = Discriminator(dataset)
    print(" Nets created")

    # Setup the MolGAN model
    model = MolGAN(
        dataset,
        G,
        D,
        V,
        optimizer=partial(Adam, lr=args.learning_rate),
        grad_penalty=args.grad_penalty,
        process_method=args.process_method,
    )
    # Setup the trainer with MLFlow logging
    mlflow_logger = MLFlowLogger(experiment_name="molgan")
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="cpu",
        logger=mlflow_logger,
    )

    # Start training
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
