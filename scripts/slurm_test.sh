#!/bin/bash
#SBATCH --job-name=text2sql_test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Create logs directory
mkdir -p logs

# Print job info
echo "========================================"
echo "Text2SQL Evaluation Setup Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Install dependencies
echo "Installing minimal dependencies for testing..."
pip install pyyaml torch tqdm requests pandas numpy vllm

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0

# Run setup test
echo "Running setup test..."
python scripts/test_setup.py

# Capture exit code
TEST_EXIT_CODE=$?

echo "========================================"
echo "Test completed at $(date)"
echo "Exit code: $TEST_EXIT_CODE"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Ready for full evaluation!"
else
    echo "❌ TESTS FAILED - Check logs before running full evaluation"
fi
echo "========================================"

exit $TEST_EXIT_CODE 