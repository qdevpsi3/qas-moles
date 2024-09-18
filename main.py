from functools import partial

from lightning import Trainer
from torch.optim import Adam

from source.datasets import MolecularDataModule, MolecularDataset
from source.nets import Discriminator, Generator
from source.trainer import MolGAN

if __name__ == "__main__":
    # Create data module
    dataset = MolecularDataset.load("./data/gdb9_molecular_dataset.pkl")
    datamodule = MolecularDataModule(dataset, batch_size=32)
    datamodule.setup()

    # Create generator
    G = Generator(dataset)
    D = Discriminator(dataset)
    V = Discriminator(dataset)
    print("Model created")

    model = MolGAN(
        dataset,
        G,
        D,
        V,
        optimizer=partial(Adam, lr=0.0001),
        grad_penalty=10.0,
        process_method="soft_gumbel",
    )
    trainer = Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model=model, datamodule=datamodule)
