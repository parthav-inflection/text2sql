#!/bin/bash
#SBATCH --job-name=text2sql_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Create logs directory
mkdir -p logs

# Print job info
echo "Starting text2sql evaluation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load environment (adjust as needed)
# source /mnt/vast/home/your_username/miniconda3/etc/profile.d/conda.sh
# conda activate text2sql

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
echo "Starting evaluation..."
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml --log-level INFO

echo "Job completed at $(date)" 