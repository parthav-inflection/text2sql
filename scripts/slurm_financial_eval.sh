#!/bin/bash
#SBATCH --job-name=financial_xiyanSQL_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/slurm_financial_%j.out
#SBATCH --error=logs/slurm_financial_%j.err

# Create logs directory
mkdir -p logs

# Print job info
echo "Starting Financial Dataset Evaluation with XiYanSQL-QwenCoder-7B-2504"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load environment (adjust as needed for your cluster)
# Uncomment and modify these lines for your cluster setup:
# module load cuda/12.1
# module load python/3.9
# source /path/to/your/venv/bin/activate
# or
# source /mnt/vast/home/your_username/miniconda3/etc/profile.d/conda.sh
# conda activate text2sql

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0

# Environment variables for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Check Python environment
echo "Checking Python environment..."
python3 --version
echo "Python path: $(which python3)"
nvidia-smi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'✅ torch {torch.__version__} available, CUDA: {torch.cuda.is_available()}')" || echo "❌ torch missing"
python3 -c "import vllm; print(f'✅ vllm available')" || echo "❌ vllm missing"
python3 -c "import transformers; print(f'✅ transformers available')" || echo "❌ transformers missing"

# Install/update dependencies if needed
echo "Installing/updating dependencies..."
pip install --user --upgrade vllm transformers huggingface-hub sqlparse tqdm pyyaml

# Install the text2sql package in development mode
echo "Installing text2sql package..."
pip install --user -e .

# Verify installation
echo "Verifying installation..."
python3 -c "from src.datasets.financial import FinancialDataset; print('✅ Financial dataset available')" || echo "❌ Financial dataset not available"
python3 -c "from src.evaluation.evaluator import Evaluator; print('✅ Evaluator available')" || echo "❌ Evaluator not available"

# Check if financial dataset is properly organized
echo "Checking financial dataset..."
if [ -f "./data/financial/processed/financial_data.json" ]; then
    echo "✅ Financial dataset processed file found"
    python3 -c "
import json
with open('./data/financial/processed/financial_data.json', 'r') as f:
    data = json.load(f)
print(f'✅ Dataset contains {len(data)} examples')
print(f'✅ Example keys: {list(data[0].keys()) if data else \"No examples\"}')"
else
    echo "❌ Financial dataset processed file not found at ./data/financial/processed/financial_data.json"
    echo "Please ensure your dataset is properly organized!"
    exit 1
fi

# Run financial evaluation with XiYanSQL
echo "Starting Financial dataset evaluation with XiYanSQL-QwenCoder-7B-2504..."
python scripts/run_eval.py \
    --config configs/experiments/financial_xiyanSQL_eval.yaml \
    --agent-config configs/agents/standard_agent.yaml \
    --log-level INFO

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Financial evaluation completed successfully!"
    echo "Results saved to: ./results/financial_xiyanSQL_eval/"
    
    # Show quick summary if results exist
    if [ -d "./results/financial_xiyanSQL_eval" ]; then
        echo "📊 Quick Results Summary:"
        find ./results/financial_xiyanSQL_eval -name "*.json" -type f | head -3
    fi
else
    echo "❌ Financial evaluation failed!"
    exit 1
fi

echo "Job completed at $(date)" 