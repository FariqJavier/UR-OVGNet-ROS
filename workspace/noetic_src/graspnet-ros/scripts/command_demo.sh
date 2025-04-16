#!/bin/bash

# Define the project root relative to this script's location
# SCRIPT_DIR is .../graspnet/graspnet/doc
# PROJECT_ROOT should be .../graspnet/graspnet/ (the parent of doc)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")   # Go up ONE level from 'doc'

# --- Example usage (assuming you still run from PROJECT_ROOT) ---

# Define the checkpoint path relative to the new project root
# If log is inside this new PROJECT_ROOT, it might be '../log/...' or just 'log/...'
# Assuming 'log' is SIBLING to 'graspnet/graspnet', the path from the OLD root was 'log/...'
# From the NEW root 'graspnet/graspnet/', the path to reach the SIBLING 'log' is '../log/...'
CHECKPOINT_PATH="logs/log_rs/checkpoint.tar"

# Change directory to the project root before executing python
cd "$PROJECT_ROOT" || exit

echo "Running from: $(pwd)"
# The module path will now be relative to the NEW PROJECT_ROOT.
# If PROJECT_ROOT is '.../graspnet/graspnet/', the module is simply 'doc.demo'
echo "Executing: python -m doc.demo --checkpoint_path $CHECKPOINT_PATH"

# Execute python as a module from the project root
# Note: the module path changes because the root context for '-m' changed!
CUDA_VISIBLE_DEVICES=0 python -m doc.demo --checkpoint_path "$CHECKPOINT_PATH"
