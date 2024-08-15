import numpy as np
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset
from tqdm import tqdm

# Set DEBUG to False for production environment where detailed logging is not required
DEBUG = False

# Disable RDKit logging when not in debug mode
if not DEBUG:
    RDLogger.DisableLog("rdApp.error")


class SparseMolecularDataset(Dataset):
    """Dataset class for handling sparse molecular data.

    This class processes molecular data files, extracts molecules, and computes various
    features necessary for molecular generation tasks using neural networks.

    Attributes:
        filename (str): Path to the molecular data file (.sdf or .smi).
        add_h (bool): Flag to determine whether hydrogen atoms should be added to the molecules.
        max_atoms (int): Maximum number of atoms per molecule for inclusion in the dataset.
    """

    def __init__(self, filename, *, add_h=False, max_atoms=None):
        """Initialize the dataset with a file path and optional transformations."""
        if not filename.endswith((".sdf", ".smi")):
            raise ValueError(
                "Invalid file format! Please provide an SDF or SMILES file."
            )
        self.filename = filename
        self.add_h = add_h
        self.max_atoms = max_atoms

        self._extract_molecules()
        self._create_encoders_decoders()
        self._compute_features()

    def __len__(self):
        """Returns the number of valid molecules in the dataset."""
        return len(self.valid_mols)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset by index."""
        sample = {
            "mol": self.valid_mols[idx],
            "smiles": self.smiles[idx],
            "S": self.data_S[idx],
            "A": self.data_A[idx],
            "X": self.data_X[idx],
            "D": self.data_D[idx],
            "F": self.data_F[idx],
            "Le": self.data_Le[idx],
            "Lv": self.data_Lv[idx],
        }
        return sample

    def _extract_molecules(self):
        """Extracts molecules from the specified file and filters them based on specified criteria."""
        print("Extracting molecules from {} ...".format(self.filename))
        # Read molecules from an SDF or SMILES file
        if self.filename.endswith(".sdf"):
            mols = list(
                filter(lambda x: x is not None, Chem.SDMolSupplier(self.filename))
            )
        elif self.filename.endswith(".smi"):
            mols = [
                Chem.MolFromSmiles(line)
                for line in open(self.filename, "r").readlines()
            ]
        # Optionally add hydrogens to each molecule
        if self.add_h:
            mols = list(map(Chem.AddHs, mols))
        # Filter molecules by the number of atoms
        if self.max_atoms is not None:
            filters = lambda x: x.GetNumAtoms() <= self.max_atoms
            mols = list(filter(filters, mols))
        print(
            "Extracted {} out of {} molecules {}adding Hydrogen!".format(
                len(mols),
                len(Chem.SDMolSupplier(self.filename)),
                "" if self.add_h else "not ",
            )
        )
        self.mols = mols

    def _create_encoders_decoders(self):
        """Creates mappings for encoding and decoding atom and bond types, as well as SMILES characters."""
        # Atom encoder and decoder creation
        print("Creating atoms encoder and decoder..")
        atom_labels = sorted(
            set(
                [atom.GetAtomicNum() for mol in self.mols for atom in mol.GetAtoms()]
                + [0]
            )
        )
        atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        atom_num_types = len(atom_labels)
        print(
            "Created atoms encoder and decoder with {} atom types and 1 PAD symbol!".format(
                atom_num_types - 1
            )
        )

        # Bond encoder and decoder creation
        print("Creating bonds encoder and decoder..")
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(
            sorted(
                set(bond.GetBondType() for mol in self.mols for bond in mol.GetBonds())
            )
        )

        bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        bond_num_types = len(bond_labels)
        print(
            "Created bonds encoder and decoder with {} bond types and 1 PAD symbol!".format(
                bond_num_types - 1
            )
        )

        # SMILES encoder and decoder creation
        print("Creating SMILES encoder and decoder..")
        smiles_labels = ["E"] + list(
            set(c for mol in self.mols for c in Chem.MolToSmiles(mol))
        )
        smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        smiles_num_types = len(smiles_labels)
        print(
            "Created SMILES encoder and decoder with {} types and 1 PAD symbol!".format(
                smiles_num_types - 1
            )
        )

        self.atom_encoder_m = atom_encoder_m
        self.atom_decoder_m = atom_decoder_m
        self.atom_num_types = atom_num_types

        self.bond_encoder_m = bond_encoder_m
        self.bond_decoder_m = bond_decoder_m
        self.bond_num_types = bond_num_types

        self.smiles_encoder_m = smiles_encoder_m
        self.smiles_decoder_m = smiles_decoder_m
        self.smiles_num_types = smiles_num_types

    def _compute_features(self):
        """Computes features and adjacency matrices for the molecules."""
        valid_mols = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        # Determine the maximum number of atoms and SMILES length for padding
        max_length = max(mol.GetNumAtoms() for mol in self.mols)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.mols)

        for _, mol in tqdm(
            enumerate(self.mols), total=len(self.mols), desc="Processing Molecules"
        ):

            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                valid_mols.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = D - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

        print(
            "Created {} features and adjacency matrices out of {} molecules!".format(
                len(valid_mols), len(self.mols)
            )
        )

        self.valid_mols = np.array(valid_mols)
        self.smiles = np.array(smiles)
        self.data_S = np.stack(data_S)
        self.data_A = np.stack(data_A)
        self.data_X = np.stack(data_X)
        self.data_D = np.stack(data_D)
        self.data_F = np.stack(data_F)
        self.data_Le = np.stack(data_Le)
        self.data_Lv = np.stack(data_Lv)

        self.num_vertices = self.data_F.shape[-2]
        self.num_features = self.data_F.shape[-1]

    def _genA(self, mol, connected=True, max_length=None):
        """Generates adjacency matrix for the molecule."""
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [
            b.GetEndAtomIdx() for b in mol.GetBonds()
        ]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):
        """Generates feature matrix for atoms in the molecule."""
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array(
            [self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (max_length - mol.GetNumAtoms()),
            dtype=np.int32,
        )

    def _genS(self, mol, max_length=None):
        """Generates a sequence of SMILES character encodings for the molecule."""
        max_length = (
            max_length if max_length is not None else len(Chem.MolToSmiles(mol))
        )

        return np.array(
            [self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)]
            + [self.smiles_encoder_m["E"]] * (max_length - len(Chem.MolToSmiles(mol))),
            dtype=np.int32,
        )

    def _genF(self, mol, max_length=None):
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


if __name__ == "__main__":
    dataset = SparseMolecularDataset(
        "./data/gdb9.sdf",
        max_atoms=9,
        add_h=False,
    )
