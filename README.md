# MolGAN Model Training and Testing Guide

This guide provides instructions on how to train and test the MolGAN model using classical or quantum generators, with or without shadow noise. It explains the command-line arguments, describes how checkpoints are saved with date-time stamps in the `checkpoints` folder, and guides you through running the script in various configurations.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Command-Line Arguments](#command-line-arguments)
- [Training the Model](#training-the-model)
  - [Using the Classical Generator](#using-the-classical-generator)
  - [Using the Quantum Generator Without Shadows](#using-the-quantum-generator-without-shadows)
  - [Using the Quantum Generator With Shadows](#using-the-quantum-generator-with-shadows)
- [Testing the Model](#testing-the-model)
- [Checkpoints and Output](#checkpoints-and-output)
- [Examples](#examples)
- [Additional Notes](#additional-notes)

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/qdevpsi3/qas-moles.git
   cd qas-moles
   ```

2. **Install the requirements:**

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the necessary Python dependencies for the project.

3. **Run the setup script:**

   ```bash
   bash setup.sh
   ```

   - **What it does:**
     - Downloads the necessary datasets (e.g., `gdb9.sdf` and additional files).
     - Extracts the dataset files.
     - Builds the molecular dataset using the provided `build_dataset.py` script and saves it as `gdb9_molecular_dataset.pkl`.

---

## Command-Line Arguments

The main script `main.py` accepts several command-line arguments to control the training and testing process:

- **`--stage`**: The stage to run. Options are `'train'` or `'test'`. Default is `'train'`.

- **`--checkpoint_path`**: Path to the checkpoint file (without the `.ckpt` extension) for testing or resuming training. Default is `None`.

- **`--generator_type`**: Type of generator to use. Options are `'classical'` or `'quantum'`. Default is `'classical'`.

- **`--use_shadows`**: Boolean flag (`True` or `False`) to use shadow noise generator for the quantum generator. Default is `False`.

- **`--data_path`**: Path to the molecular dataset file. Default is `'./data/gdb9_molecular_dataset_small.pkl'`.

- **`--batch_size`**: Batch size for training and testing. Default is `32`.

- **`--max_epochs`**: Maximum number of epochs to train. Default is `5`.

- **`--learning_rate`**: Learning rate for the optimizer. Default is `0.001`.

- **`--grad_penalty`**: Gradient penalty regularization factor. Default is `10.0`.

- **`--process_method`**: Method to process the output probabilities. Options are `'soft_gumbel'` or `'hard_gumbel'`. Default is `'soft_gumbel'`.

- **`--agg_method`**: Aggregation method for the rewards. Options are `'prod'` or `'mean'`. Default is `'prod'`.

- **`--accelerator`**: Device to use for computation. Options are `'cpu'`, `'gpu'`, etc. Default is `'cpu'`.

---

## Training the Model

To train the MolGAN model, use the `--stage train` argument (default is `'train'`). The model supports both classical and quantum generators. You can also specify whether to use shadow noise with the quantum generator.

### Using the Classical Generator

To train the model using the **classical generator**:

```bash
python main.py --stage train --generator_type classical --max_epochs 5
```

- **Explanation:**
  - `--generator_type classical`: Uses the classical generator.
  - `--max_epochs 5`: Sets the maximum number of training epochs to 5.

### Using the Quantum Generator Without Shadows

To train the model using the **quantum generator without shadow noise**:

```bash
python main.py --stage train --generator_type quantum --use_shadows False --max_epochs 5
```

- **Explanation:**
  - `--generator_type quantum`: Uses the quantum generator.
  - `--use_shadows False`: Disables shadow noise in the quantum generator.

### Using the Quantum Generator With Shadows

To train the model using the **quantum generator with shadow noise**:

```bash
python main.py --stage train --generator_type quantum --use_shadows True --max_epochs 5
```

- **Explanation:**
  - `--use_shadows True`: Enables shadow noise in the quantum generator.

---

## Testing the Model

To test a trained model, use the `--stage test` argument and provide the path to the checkpoint file using `--checkpoint_path`.

**Example:**

Assuming you have a checkpoint file named `best-checkpoint-classical-20240930-115624.ckpt` in the `checkpoints/` directory, you can test the model as follows:

```bash
python main.py --stage test --checkpoint_path best-checkpoint-classical-20240930-115624 --generator_type classical
```

- **Explanation:**
  - `--stage test`: Runs the testing stage.
  - `--checkpoint_path best-checkpoint-classical-20240930-115624`: Specifies the checkpoint file (without the `.ckpt` extension).
  - `--generator_type classical`: Must match the generator type used during training.

**Note:** The script will generate molecules, print their SMILES strings, and save them to a file.

---

## Checkpoints and Output

- **Checkpoints:**
  - The script saves model checkpoints in the `checkpoints/` directory.
  - Checkpoint filenames include the generator type and a date-time stamp, e.g., `best-checkpoint-classical-20240930-115624.ckpt`.
  - The date-time stamp is in the format `YYYYMMDD-HHMMSS`.
  - **Important:** The checkpoint files are saved under these unique names to prevent overwriting and to keep track of different training runs.

- **Generated SMILES:**
  - During testing, the script generates molecules and converts them to SMILES strings.
  - The SMILES strings are printed to the console and saved to a file named `generated_smiles_{YYYYMMDD-HHMMSS}.txt`, where `{YYYYMMDD-HHMMSS}` is the current date and time.
  - The output file is saved in the current working directory.

---

## Examples

### Training with the Classical Generator

```bash
python main.py --stage train --generator_type classical --max_epochs 10
```

- Trains the model using the classical generator for 10 epochs.

### Training with the Quantum Generator Without Shadows

```bash
python main.py --stage train --generator_type quantum --use_shadows False --max_epochs 10
```

- Trains the model using the quantum generator without shadow noise for 10 epochs.

### Training with the Quantum Generator With Shadows

```bash
python main.py --stage train --generator_type quantum --use_shadows True --max_epochs 10
```

- Trains the model using the quantum generator with shadow noise for 10 epochs.

### Testing a Trained Model

First, list the checkpoint files in the `checkpoints/` directory:

```bash
ls checkpoints/
```

Suppose you have the checkpoint file `best-checkpoint-classical-20240930-115624.ckpt`. To test this model:

```bash
python main.py --stage test --checkpoint_path best-checkpoint-classical-20240930-115624 --generator_type classical
```

- **Note:** Ensure that `--generator_type` matches the generator type used during training.

---

## Additional Notes

- **Data Path:**
  - By default, the script uses the dataset at `./data/gdb9_molecular_dataset_small.pkl`.
  - If you have a different dataset file, specify it using the `--data_path` argument.

- **Accelerator:**
  - To use a GPU for training or testing, set `--accelerator gpu`.
  - Ensure that your environment has GPU support and the necessary drivers installed.

- **Batch Size and Learning Rate:**
  - Adjust the `--batch_size` and `--learning_rate` according to your computational resources and requirements.

- **Process Method and Aggregation Method:**
  - `--process_method`: Choose how to process the output probabilities from the generator. Options are `'soft_gumbel'` or `'hard_gumbel'`.
  - `--agg_method`: Choose the aggregation method for the rewards during training. Options are `'prod'` or `'mean'`.

- **Monitoring and Checkpointing:**
  - The script uses `val_metric` to monitor performance during validation.
  - The best model checkpoint is saved based on the highest `val_metric`.
  - Checkpoints are saved with date-time stamps to avoid overwriting and to keep track of different runs.

- **SMILES Output:**
  - Generated SMILES strings during testing are saved to a file with a name like `generated_smiles_{YYYYMMDD-HHMMSS}.txt`.
  - These files are useful for analyzing the molecules generated by the model.

- **Logging:**
  - The script uses MLflow for logging metrics and parameters.
  - Logs can be viewed using the MLflow UI if desired.

- **RDKit Dependency:**
  - The script requires RDKit for molecule handling and SMILES conversion.
  - Install RDKit via Conda:
    ```bash
    conda install -c rdkit rdkit
    ```
  - Or via pip:
    ```bash
    pip install rdkit-pypi
    ```

---

## Conclusion

This guide provides a comprehensive overview of how to run the MolGAN model using different generator configurations and stages. By following these instructions, you can train and test the model, generate molecules, and analyze the results.

If you have any questions or encounter issues, please refer to the code comments or open an issue in the repository.

---

**Example Command to Train and Test with Classical Generator:**

```bash
# Train
python main.py --stage train --generator_type classical --max_epochs 5

# Test
python main.py --stage test --generator_type classical --checkpoint_path best-checkpoint-classical-YYYYMMDD-HHMMSS
```

**Example Command to Train and Test with Quantum Generator With Shadows:**

```bash
# Train
python main.py --stage train --generator_type quantum --use_shadows True --max_epochs 5

# Test
python main.py --stage test --generator_type quantum --use_shadows True --checkpoint_path best-checkpoint-quantum-shadows-YYYYMMDD-HHMMSS
```

**Note:** Replace `YYYYMMDD-HHMMSS` with the actual date-time stamp from your checkpoint file.

