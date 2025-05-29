#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the SMILES for aspirin
INPUT_SMILES="CC(=O)OC1=CC=CC=C1C(=O)O"

# Define the output directory
SAVE_DIR="test_output_aspirin"

# Create the output directory if it doesn't exist
mkdir -p "$SAVE_DIR"

echo "Running example generation for SMILES: $INPUT_SMILES"
echo "Output will be saved in: $SAVE_DIR"

python scripts/generate.py \
    --input_smiles "$INPUT_SMILES" \
    --search_type "mcts" \
    --chemical_spaces "real" \
    --building_blocks_paths "synthemol/resources/real/building_blocks.csv" \
    --reaction_to_building_blocks_paths "synthemol/resources/real/reaction_to_building_blocks.pkl" \
    --score_types "tanimoto_to_input" \
    --score_weights 1.0 \
    --score_signs 1 \
    --score_fingerprint_types "morgan" \
    --score_names "tanimoto_aspirin_similarity" \
    --n_rollout 10 \
    --max_reactions 1 \
    --num_expand_nodes 10 \
    --explore_weight 1.0 \
    --save_dir "$SAVE_DIR"

echo "Example generation finished."
echo "Check $SAVE_DIR for results."
