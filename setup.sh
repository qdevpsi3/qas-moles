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

# Experiment variables
DATA_PATH="./data/gdb9_molecular_dataset.pkl"
EPOCHS=300
BATCH_SIZE=32
Z_DIM=4

# Run experiments
echo "Starting experiments..."

# Classical generator
echo "Training with classical generator..."
python main.py --stage train --generator_type classical --data_path $DATA_PATH --max_epochs $EPOCHS --batch_size $BATCH_SIZE --z_dim $Z_DIM

# Quantum generator without shadows
echo "Training with quantum generator (no shadows)..."
python main.py --stage train --generator_type quantum --use_shadows False --data_path $DATA_PATH --max_epochs $EPOCHS --batch_size $BATCH_SIZE --z_dim $Z_DIM

# Quantum generator with shadows
echo "Training with quantum generator (with shadows)..."
python main.py --stage train --generator_type quantum --use_shadows True --data_path $DATA_PATH --max_epochs $EPOCHS --batch_size $BATCH_SIZE --z_dim $Z_DIM

echo "All experiments completed successfully."