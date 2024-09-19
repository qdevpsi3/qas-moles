#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Navigate to the data directory
cd data

# Download the dataset
echo "Downloading gdb9 dataset..."
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz 

# Extract the dataset
echo "Extracting gdb9 dataset..."
tar xvzf gdb9.tar.gz

# Remove the compressed file
rm gdb9.tar.gz

# Download additional resources
echo "Downloading NP_score.pkl.gz..."
wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz

echo "Downloading SA_score.pkl.gz..."
wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz

# Navigate back to the root directory
cd ..

# Run the Python script to build the dataset
echo "Building the dataset..."
python -m scripts.build_dataset --input data/gdb9.sdf --output data/gdb9_molecular_dataset_small.pkl --max_atoms 9 --max_mols 1000
python -m scripts.build_dataset --input data/gdb9.sdf --output data/gdb9_molecular_dataset.pkl --max_atoms 9

echo "Setup completed successfully."
