import argparse
from datetime import datetime
from functools import partial

import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split

from source.data import TwoGaussianDataset
from source.gmodel import GaussianGAN, MLPDiscriminator, MLPGenerator, MLPPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the GaussianGAN model on a synthetic 2D Gaussians dataset"
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
        help="Path to the checkpoint file (without .ckpt extension)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
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
        "--train_predictor_on_fake",
        type=bool,
        default=False,
        help="Train the predictor also on fake samples",
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
        help="Device to use ('cpu' or 'gpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )

    # Enable MLflow autologging
    mlflow.pytorch.autolog(checkpoint_save_best_only=False)

    # Create the dataset
    dataset = TwoGaussianDataset(
        num_samples=20000,
        dim=2,
        mean0=(-10.0, -10.0),
        mean1=(10.0, 10.0),
        std0=1.0,
        std1=1.0,
    )

    # Split into train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TwoGaussianDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=TwoGaussianDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=TwoGaussianDataset.collate_fn,
    )

    # Initialize networks
    G = MLPGenerator(noise_dim=16, hidden_dim=64, output_dim=2)
    D = MLPDiscriminator(input_dim=2, hidden_dim=64)
    P = MLPPredictor(input_dim=2, hidden_dim=64)
    G.to(device)
    D.to(device)
    P.to(device)
    print("Networks created")

    # Setup the GaussianGAN model
    model = GaussianGAN(
        dataset=dataset,
        generator=G,
        discriminator=D,
        predictor=P,
        optimizer_class=partial(torch.optim.RMSprop, lr=args.learning_rate),
        grad_penalty=args.grad_penalty,
        train_predictor_on_fake=args.train_predictor_on_fake,
        n_critic=args.n_critic,
    )
    model.to(device)

    # Optionally load from a checkpoint
    if args.checkpoint_path is not None:
        model = GaussianGAN.load_from_checkpoint(
            "checkpoints/" + args.checkpoint_path + ".ckpt",
            dataset=dataset,
            generator=G,
            discriminator=D,
            predictor=P,
            optimizer_class=partial(torch.optim.RMSprop, lr=args.learning_rate),
            grad_penalty=args.grad_penalty,
            train_predictor_on_fake=args.train_predictor_on_fake,
            n_critic=args.n_critic,
        )
        model.to(device)

    # Define the checkpoint callback
    current_date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"best-checkpoint-gaussian-{current_date_time}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",  # The metric we log in GaussianGAN
        save_top_k=1,
        mode="max",
        dirpath="checkpoints/",
        filename=filename,
        save_last=True,
    )

    # Setup the trainer with MLflow logging and checkpoint callback
    mlflow_logger = MLFlowLogger(experiment_name="gaussian_gan")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=mlflow_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    if args.stage == "train":
        # Start training
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    elif args.stage == "test":
        # Start testing
        trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
