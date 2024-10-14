import pickle
from typing import NamedTuple

import numpy as np
import torch
from lightning import LightningDataModule
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Disable RDKit logging
RDLogger.DisableLog("rdApp.error")


def extract_molecules_from_file(filename):
    """Extracts molecules from the specified file and filters them based on specified criteria."""
    print("Extracting molecules from {} ...".format(filename))
    # Read molecules from an SDF or SMILES file
    if filename.endswith(".sdf"):
        mols = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
    elif filename.endswith(".smi"):
        mols = [Chem.MolFromSmiles(line) for line in open(filename, "r").readlines()]
    return mols


def process_molecules(mols, add_h=False, max_atoms=None):
    # Optionally add hydrogens to each molecule
    if add_h:
        mols = list(map(Chem.AddHs, mols))
    # Filter molecules by the number of atoms
    if max_atoms is not None:
        filters = lambda x: x.GetNumAtoms() <= max_atoms
        mols = list(filter(filters, mols))
    return mols


def get_atom_encoders_decoders(mols):
    # Atom encoder and decoder creation
    atom_labels = sorted(
        set([atom.GetAtomicNum() for mol in mols for atom in mol.GetAtoms()] + [0])
    )
    atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
    atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
    atom_num_types = len(atom_labels)
    return atom_encoder_m, atom_decoder_m, atom_num_types


def get_bond_encoders_decoders(mols):
    # Bond encoder and decoder creation
    bond_labels = [Chem.rdchem.BondType.ZERO] + list(
        sorted(set(bond.GetBondType() for mol in mols for bond in mol.GetBonds()))
    )

    bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
    bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
    bond_num_types = len(bond_labels)
    return bond_encoder_m, bond_decoder_m, bond_num_types


def get_smiles_encoders_decoders(mols):
    # SMILES encoder and decoder creation
    smiles_labels = ["E"] + list(set(c for mol in mols for c in Chem.MolToSmiles(mol)))
    smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
    smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
    smiles_num_types = len(smiles_labels)
    return smiles_encoder_m, smiles_decoder_m, smiles_num_types


def extact_features(mols, max_length):
    """Computes features and adjacency matrices for the molecules."""
    bond_encoder_m, _, _ = get_bond_encoders_decoders(mols)
    atom_encoder_m, _, _ = get_atom_encoders_decoders(mols)
    smiles_encoder_m, _, _ = get_smiles_encoders_decoders(mols)

    def _genA(mol, connected=True, max_length=None):
        """Generates adjacency matrix for the molecule."""
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [
            b.GetEndAtomIdx() for b in mol.GetBonds()
        ]
        bond_type = [bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(mol, max_length=None):
        """Generates feature matrix for atoms in the molecule."""
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array(
            [atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (max_length - mol.GetNumAtoms()),
            dtype=np.int32,
        )

    def _genS(mol, max_length=None):
        """Generates a sequence of SMILES character encodings for the molecule."""
        max_length = (
            max_length if max_length is not None else len(Chem.MolToSmiles(mol))
        )

        return np.array(
            [smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)]
            + [smiles_encoder_m["E"]] * (max_length - len(Chem.MolToSmiles(mol))),
            dtype=np.int32,
        )

    def _genF(mol, max_length=None):
        """Generates a feature matrix for atoms with various chemical properties."""
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array(
            [
                [
                    *[a.GetDegree() == i for i in range(5)],
                    *[a.GetExplicitValence() == i for i in range(9)],
                    *[int(a.GetHybridization()) == i for i in range(1, 7)],
                    *[a.GetImplicitValence() == i for i in range(9)],
                    a.GetIsAromatic(),
                    a.GetNoImplicit(),
                    *[a.GetNumExplicitHs() == i for i in range(5)],
                    *[a.GetNumImplicitHs() == i for i in range(5)],
                    *[a.GetNumRadicalElectrons() == i for i in range(5)],
                    a.IsInRing(),
                    *[a.IsInRingSize(i) for i in range(2, 9)],
                ]
                for a in mol.GetAtoms()
            ],
            dtype=np.int32,
        )

        return np.vstack(
            (features, np.zeros((max_length - features.shape[0], features.shape[1])))
        )

    valid_mols = []
    data_S = []
    data_A = []
    data_X = []
    data_D = []
    data_F = []
    data_Le = []
    data_Lv = []

    # Determine the maximum number of atoms and SMILES length for padding
    max_length = max(mol.GetNumAtoms() for mol in mols)
    max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in mols)

    for _, mol in tqdm(
        enumerate(mols),
        total=len(mols),
        desc="Extracting Molecules Features",
    ):

        A = _genA(mol, connected=True, max_length=max_length)
        D = np.count_nonzero(A, -1)
        if A is not None:
            valid_mols.append(mol)
            data_S.append(_genS(mol, max_length=max_length_s))
            data_A.append(A)
            data_X.append(_genX(mol, max_length=max_length))
            data_D.append(D)
            data_F.append(_genF(mol, max_length=max_length))

            L = D - A
            Le, Lv = np.linalg.eigh(L)

            data_Le.append(Le)
            data_Lv.append(Lv)

    data_S = np.stack(data_S)
    data_A = np.stack(data_A)
    data_X = np.stack(data_X)
    data_D = np.stack(data_D)
    data_F = np.stack(data_F)
    data_Le = np.stack(data_Le)
    data_Lv = np.stack(data_Lv)
    print(
        "Created {} features and adjacency matrices out of {} molecules!".format(
            len(valid_mols), len(mols)
        )
    )
    features = {
        "S": data_S,
        "A": data_A,
        "X": data_X,
        "D": data_D,
        "F": data_F,
        "Le": data_Le,
        "Lv": data_Lv,
    }
    features = {k: np.array(v, dtype=np.float32) for k, v in features.items()}
    return valid_mols, features


def extract_smiles(mols):
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles


class MolecularSample(NamedTuple):
    mol: Chem.Mol
    smiles: str
    features: dict
    is_batched: bool = False


class MolecularDataset(Dataset):

    def __init__(self, mols, smiles, features):
        self.mols = mols
        self.smiles = smiles
        self.features = features

        self.atom_encoder_m, self.atom_decoder_m, self.atom_num_types = (
            get_atom_encoders_decoders(self.mols)
        )
        self.bond_encoder_m, self.bond_decoder_m, self.bond_num_types = (
            get_bond_encoders_decoders(self.mols)
        )
        self.smiles_encoder_m, self.smiles_decoder_m, self.smiles_num_types = (
            get_smiles_encoders_decoders(self.mols)
        )
        self.num_vertices = self.features["F"].shape[-2]
        self.num_features = self.features["F"].shape[-1]

    def __len__(self):
        """Returns the number of valid molecules in the dataset."""
        return len(self.mols)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset by index."""
        mol = self.mols[idx]
        smiles = self.smiles[idx]
        features = dict((k, v[idx]) for k, v in self.features.items())
        return MolecularSample(mol, smiles, features)

    def matrices2mol(self, node_labels, edge_labels, strict=False):
        """Converts node and edge labels back into a molecule object."""
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(
                    int(start), int(end), self.bond_decoder_m[edge_labels[start, end]]
                )

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def save(self, filename):
        """Saves the dataset to a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {filename}.")

    @classmethod
    def load(cls, filename):
        """Loads the dataset from a pickle file."""
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {filename}.")
        return dataset

    def seq2mol(self, seq, strict=False):
        """Converts a sequence of SMILES character encodings back into a molecule object."""
        mol = Chem.MolFromSmiles(
            "".join([self.smiles_decoder_m[e] for e in seq if e != 0])
        )

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol


def collate_fn(batch):
    """Custom collate function for MolecularDataset to handle batching of features."""

    # Unzip the batch to separate mols, smiles, and features
    batch = [(sample.mol, sample.smiles, sample.features) for sample in batch]
    mols, smiles_list, features_list = zip(*batch)

    # Initialize a dictionary to hold batched features
    batched_features = {}

    # For each key in the features dictionary
    for key in features_list[0].keys():
        # Stack the arrays from each sample to create a batch tensor
        batched_array = np.stack([features[key] for features in features_list], axis=0)
        # Convert the numpy array to a PyTorch tensor
        batched_features[key] = torch.tensor(batched_array, dtype=torch.float32)

    # Return the batched mols, smiles, and features
    samples = MolecularSample(mols, smiles_list, batched_features, is_batched=True)
    return samples


class MolecularDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        batch_size=32,
        train_test_val_split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        # Save the batch size as a hyperparameter
        self.save_hyperparameters("batch_size")
        # Save the dataset
        self.dataset = dataset
        self.train_test_val_split = train_test_val_split
        # Initialize the train, validation, and test datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This method should only run on 1 GPU/TPU in distributed settings,
        # thus we do not need to set anything related to the dataset itself here,
        # since it's done in setup() which is called on every GPU/TPU.
        pass

    def setup(self, stage=None):
        # Calculate split sizes based on the provided tuple ratios
        train_size = int(self.train_test_val_split[0] * len(self.dataset))
        val_size = int(self.train_test_val_split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        # Returns the training dataloader.
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        # Returns the validation dataloader.
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Returns the testing dataloader.
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
        )
