#!/bin/bash
#SBATCH --job-name=text2sql-eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Check if a config file was provided
if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <path_to_experiment_config>"
    exit 1
fi

CONFIG_FILE=$1
EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml)

# Dynamically set the job name based on the config file
# Note: This is an illustrative comment; the job name is usually set from the initial #SBATCH directives.
# To dynamically set it, you might need advanced SLURM features or a wrapper script.

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "--- JOB INFO ---"
echo "Starting experiment: ${EXPERIMENT_NAME}"
echo "Configuration file: ${CONFIG_FILE}"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "----------------"

# --- Environment Setup ---
# IMPORTANT: You MUST uncomment and modify this section for your cluster.
#
# Example for loading modules:
# echo "Loading modules..."
# module load cuda/12.1
# module load python/3.10
#
# Example for activating a Conda environment:
# echo "Activating Conda environment..."
# source /path/to/your/miniconda3/etc/profile.d/conda.sh
# conda activate your_text2sql_env
# -------------------------

# Set CUDA environment variables for performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --- Dependency and Environment Verification ---
echo "Verifying Python environment..."
python3 --version
echo "Verifying key packages..."
pip3 freeze | grep -E "torch|vllm|transformers|sqlparse"

# Install/update the local text2sql package
echo "Installing/updating local package in editable mode..."
pip install -e .

# --- Run Evaluation ---
echo "Starting evaluation script: run_eval.py"
python scripts/run_eval.py --config "${CONFIG_FILE}"

# --- Post-run Check ---
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully for ${EXPERIMENT_NAME}!"
else
    echo "❌ Evaluation FAILED for ${EXPERIMENT_NAME}!"
    exit 1
fi

echo "Job completed at $(date)" 