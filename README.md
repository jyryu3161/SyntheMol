# Similarity-Guided Synthesis Pathway Generation

This project, adapted from the original [SyntheMol](https://github.com/swansonk14/SyntheMol) framework, focuses on generating synthesizable analogs and their reaction pathways for a user-provided chemical compound. Instead of optimizing for predicted bioactivity as in the original SyntheMol, this version utilizes Tanimoto similarity to a target molecule to guide the exploration of chemical space using Monte Carlo Tree Search (MCTS) or Reinforcement Learning (RL).

## Core Functionality

Given a SMILES string of a target molecule, this tool will:
1.  Search a defined chemical space (composed of building blocks and known reactions).
2.  Identify molecules that are similar to the input molecule (based on Tanimoto similarity using Morgan fingerprints).
3.  For these similar molecules, provide the sequence of reactions and building blocks required for their synthesis.

This allows for the exploration of feasible synthetic routes to analogs of a compound of interest.

## Key Features

*   **Targeted Analog Generation**: Provide an input SMILES and find similar, synthesizable molecules.
*   **Guided Search**: Employs MCTS or RL to navigate large chemical reaction spaces.
*   **Similarity Metric**: Uses Tanimoto similarity with Morgan fingerprints as the primary score to optimize.
*   **Synthesis Pathway Output**: Generates detailed step-by-step reaction pathways for candidate molecules.
*   **Chemical Spaces**: Can utilize predefined chemical reaction spaces (e.g., Enamine REAL space, WuXi GalaXi) or custom-defined spaces.

## Installation

This project uses Conda for environment management.

1.  **Prerequisites**:
    *   An Ubuntu-like environment.
    *   Conda (Miniconda or Anaconda) installed and available in your PATH.
    *   Git for cloning the repository.

2.  **Setup**:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory>
    bash install.sh
    ```
    The `install.sh` script will:
    *   Check for Conda.
    *   Create a Conda environment named `synthemol_env` with necessary dependencies (including RDKit, PyTorch).
    *   Install Python packages listed in `requirements_mcts.txt` and `requirements_rl.txt`.

3.  **Activate Environment**:
    After the script completes, activate the Conda environment:
    ```bash
    conda activate synthemol_env
    ```

## Quick Start / Usage Example

To generate molecules similar to a given input SMILES, use the `scripts/generate.py` script.

**Example Command:**

```bash
python scripts/generate.py \
    --input_smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \ # Example: Aspirin
    --search_type "mcts" \
    --chemical_spaces "real" \
    --building_blocks_paths "synthemol/resources/real/building_blocks.csv" \
    --reaction_to_building_blocks_paths "synthemol/resources/real/reaction_to_building_blocks.pkl" \
    --score_types "tanimoto_to_input" \
    --score_weights 1.0 \
    --score_signs 1 \
    --score_fingerprint_types "morgan" \
    --score_names "tanimoto_similarity" \
    --n_rollout 100 \
    --max_reactions 2 \
    --save_dir "output_similar_to_aspirin"
```

**Explanation of Key Arguments:**

*   `--input_smiles`: SMILES string of the molecule you want to find analogs for.
*   `--search_type`: `mcts` or `rl`.
*   `--chemical_spaces`: Defines the reaction space and building blocks (e.g., "real").
*   `--building_blocks_paths`, `--reaction_to_building_blocks_paths`: Paths to chemical space definition files.
*   `--score_types "tanimoto_to_input"`: Specifies that Tanimoto similarity to the `--input_smiles` should be used as the score.
*   `--score_weights 1.0`: Weight for the Tanimoto score (if it's the only score).
*   `--score_signs 1`: Maximize the Tanimoto score.
*   `--score_fingerprint_types "morgan"`: Fingerprint type for Tanimoto calculation.
*   `--score_names`: Name for the score column in the output CSV.
*   `--n_rollout`: Number of MCTS/RL rollouts to perform. More rollouts can lead to better results but take longer.
*   `--max_reactions`: Maximum number of reaction steps to build a molecule.
*   `--save_dir`: Directory where results (CSV files, logs) will be saved.

For a runnable example, see the `run_example.sh` script in the repository root. You can adapt it for your own inputs.

## Output Description

The primary output is a CSV file (e.g., `molecules.csv`) located in the specified `--save_dir`. This file contains:
*   `smiles`: The SMILES string of the generated similar molecule.
*   `node_id`, `rollout_num`, etc.: Search process information.
*   `tanimoto_similarity` (or your specified `--score_names`): The Tanimoto similarity score to your input molecule.
*   `num_reactions`: Number of reactions in the synthesis.
*   `reaction_X_chemical_space`, `reaction_X_id`: Information about each reaction step.
*   `building_block_X_Y_id`, `building_block_X_Y_smiles`: Details of each building block used in each reaction.

This provides a full synthesis pathway for each generated analog.

## Original Project

This work is based on the SyntheMol project by Swanson et al., originally aimed at designing novel bioactive compounds. For details on the original framework and its applications in antibiotic discovery, please refer to:
*   **GitHub**: [swansonk14/SyntheMol](https://github.com/swansonk14/SyntheMol)
*   **Paper**: Swanson, K., Liu, G., Catacutan, D. B., Arnold, A., Zou, J., Stokes, J. M. Generative AI for designing and validating easily synthesizable and structurally novel antibiotics. _Nature Machine Intelligence_, 2024.

## License
This project likely retains the original license from the SyntheMol repository (e.g., MIT License). Please check the `LICENSE` file for details.
