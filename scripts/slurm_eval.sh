#!/bin/bash
#SBATCH --job-name=bird_agent_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Create logs directory
mkdir -p logs

# Print job info
echo "Starting BIRD Agent Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load environment (adjust as needed for your cluster)
# source /mnt/vast/home/your_username/miniconda3/etc/profile.d/conda.sh
# conda activate text2sql

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0

# Check Python environment
echo "Checking Python environment..."
python3 --version
echo "Python path: $(which python3)"

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'✅ torch {torch.__version__} available')" || echo "❌ torch missing"
python3 -c "import numpy; print(f'✅ numpy available')" || echo "❌ numpy missing"
python3 -c "import transformers; print(f'✅ transformers available')" || echo "❌ transformers missing"
python3 -c "import vllm; print(f'✅ vllm available')" || echo "❌ vllm missing"

# Install/update dependencies if needed
echo "Installing/updating dependencies..."
pip install --user --upgrade vllm transformers huggingface-hub sqlparse tqdm pyyaml

# Install the text2sql package in development mode
echo "Installing text2sql package..."
pip install --user -e .

# Verify installation
echo "Verifying installation..."
python3 -c "from src.agents.factory import AgentFactory; print('✅ Agent factory available')" || echo "❌ Agent factory not available"
python3 -c "from src.evaluation.evaluator import Evaluator; print('✅ Evaluator available')" || echo "❌ Evaluator not available"

# Run BIRD evaluation with agent-based architecture
echo "Starting BIRD evaluation with agent-based architecture..."
python scripts/run_eval.py \
    --config configs/experiments/bird_eval.yaml \
    --agent-config configs/agents/standard_agent.yaml \
    --log-level INFO

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
else
    echo "❌ Evaluation failed!"
    exit 1
fi

echo "Job completed at $(date)" 