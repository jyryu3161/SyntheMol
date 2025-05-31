#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Synthemol installation for Ubuntu (Conda based)..."

# 0. Check if Conda is available
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found."
    echo "Please install Miniconda or Anaconda first, and ensure 'conda' is in your PATH."
    echo "Installation instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

# 1. Update package index (for system dependencies like build-essential)
echo "Updating package index..."
sudo apt-get update

# 2. Install essential system dependencies (build-essential for compiling packages if needed)
echo "Installing essential system dependencies (build-essential)..."
sudo apt-get install -y build-essential

# 3. Create Conda environment
ENV_NAME="synthemol_env"
PYTHON_VERSION="3.10"
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    echo "If you want to recreate it, please remove it first with: conda env remove -n $ENV_NAME"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "Conda environment '$ENV_NAME' created/verified."

# 4. Install key packages with Conda
echo "Installing RDKit, PyTorch (cpuonly), NumPy, Pandas with Conda..."
# Note: Specific versions from requirements.txt are not strictly enforced here;
# Conda will pick compatible versions. This is generally more robust for these complex packages.
conda install -n "$ENV_NAME" -c conda-forge rdkit numpy pandas -y
conda install -n "$ENV_NAME" -c pytorch pytorch torchvision torchaudio cpuonly -y
# As of recent PyTorch versions, numpy might be better from pytorch channel too, but conda-forge is usually fine.

# 5. Install remaining Python packages using pip within the Conda environment
echo "Installing remaining Python dependencies from requirements files using pip..."

# Ensure pip itself is up-to-date within the Conda environment
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

# Install packages from requirements_mcts.txt
if [ -f requirements_mcts.txt ]; then
    echo "Installing packages from requirements_mcts.txt..."
    conda run -n "$ENV_NAME" python -m pip install -r requirements_mcts.txt
else
    echo "Warning: requirements_mcts.txt not found. Skipping."
fi

# Install packages from requirements_rl.txt
# This will also handle upgrades for packages already installed from requirements_mcts.txt if versions differ
if [ -f requirements_rl.txt ]; then
    echo "Installing packages from requirements_rl.txt..."
    conda run -n "$ENV_NAME" python -m pip install -r requirements_rl.txt
else
    echo "Warning: requirements_rl.txt not found. Skipping."
fi

echo ""
echo "---------------------------------------------------------------------"
echo "Conda-based installation of dependencies complete."
echo "To activate the Conda environment, run the following command:"
echo "  conda activate $ENV_NAME"
echo "After activation, you can run the project scripts."
echo "---------------------------------------------------------------------"
