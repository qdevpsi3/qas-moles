# qas-moles

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/qas-moles.git
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
```
