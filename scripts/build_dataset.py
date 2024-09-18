#!/usr/bin/env python

# build_dataset.py

import argparse
import os
import sys

from rdkit import RDLogger

# Add the source directory to the Python path
sys.path.append("..")

# Import your module
from source.datasets import (
    MolecularDataset,
    extact_features,
    extract_molecules_from_file,
    extract_smiles,
    process_molecules,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build and save MolecularDataset")
    parser.add_argument(
        "--input", type=str, required=True, help="Input file (SDF or SMILES)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file (pickle)"
    )
    parser.add_argument(
        "--max_atoms", type=int, default=9, help="Maximum number of atoms"
    )
    parser.add_argument(
        "--add_h", action="store_true", help="Add hydrogens to molecules"
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.error")

    print("Extracting molecules from the dataset file ...")
    mols = extract_molecules_from_file(args.input)
    mols = process_molecules(mols, add_h=args.add_h, max_atoms=args.max_atoms)
    mols, features = extact_features(mols, max_length=args.max_atoms)
    smiles = extract_smiles(mols)
    dataset = MolecularDataset(mols, smiles, features)
    dataset.save(args.output)
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
