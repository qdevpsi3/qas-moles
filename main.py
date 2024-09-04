from functools import partial

from lightning import Trainer
from torch.optim import Adam

from source.datasets import SparseMolecularDataModule, SparseMolecularDataset
from source.nets import Discriminator, Generator
from source.trainer import MolGAN

if __name__ == "__main__":
    # Create data module
    datamodule = SparseMolecularDataModule(
        "./data/gdb9.sdf",
        max_atoms=9,
        add_h=False,
    )
    datamodule.setup()

    complexity = "nr"
    if complexity == "nr":
        g_conv_dim = [128, 256, 512]
    elif complexity == "mr":
        g_conv_dim = [128]
    elif complexity == "hr":
        g_conv_dim = [16]
    else:
        raise ValueError(
            "Please enter an valid model complexity from 'mr', 'hr' or 'nr'!"
        )
    z_dim = 8
    num_vertices = datamodule.dataset.num_vertices
    bond_num_types = datamodule.dataset.bond_num_types
    atom_num_types = datamodule.dataset.atom_num_types
    print(
        f"num_vertices: {num_vertices}, bond_num_types: {bond_num_types}, atom_num_types: {atom_num_types}"
    )
    dropout = 0.0

    d_conv_dim = [[128, 64], 128, [128, 64]]
    m_dim = atom_num_types
    b_dim = bond_num_types
    # Create generator
    G = Generator(
        g_conv_dim,
        z_dim,
        num_vertices,
        bond_num_types,
        atom_num_types,
        dropout,
    )
    D = Discriminator(d_conv_dim, m_dim, b_dim - 1, dropout)
    V = Discriminator(d_conv_dim, m_dim, b_dim - 1, dropout)
    print("Model created")

    model = MolGAN(
        G,
        D,
        V,
        optimizer=partial(Adam, lr=0.0001),
        grad_penalty=10.0,
        process_method="soft_gumbel",
    )
    trainer = Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model=model, datamodule=datamodule)
