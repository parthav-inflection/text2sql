# Text2SQL Evaluation Scaffolding

A minimal, scalable framework for evaluating text2sql models and agent architectures on different benchmarks.

## Features

- **Model Support**: Currently supports vLLM-based models (Qwen3-8B, OmniSQL-7B)
- **Dataset Support**: Bird benchmark with automatic setup
- **GPU Optimized**: Designed for efficient inference on NVIDIA GPUs
- **Slurm Compatible**: Ready for cluster deployment
- **Extensible**: Easy to add new models, datasets, and metrics

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data results logs
```

### 2. Run Evaluation

```bash
# Run locally
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml

# Or submit to Slurm
sbatch scripts/slurm_eval.sh
```

## Configuration

### Model Configuration

Models are configured in `configs/models/`:

```yaml
# configs/models/qwen3.yaml
name: "Qwen3-8B"
model_path: "Qwen/Qwen3-8B"
model_type: "vllm"
generation_config:
  temperature: 0.0
  max_tokens: 512
vllm_config:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
```

### Experiment Configuration

Experiments are configured in `configs/experiments/`:

```yaml
# configs/experiments/bird_eval.yaml
experiment_name: "bird_bench_eval"
dataset:
  name: "bird"
  subset_size: 100
models:
  - "configs/models/qwen3.yaml"
  - "configs/models/omnisql.yaml"
evaluation:
  metrics:
    - "execution_accuracy"
```

## Project Structure

```
text2sql/
├── configs/                 # Configuration files
│   ├── models/             # Model configurations
│   └── experiments/        # Experiment configurations
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── datasets/          # Dataset implementations
│   ├── evaluation/        # Evaluation logic
│   └── utils/             # Utilities
├── scripts/               # Execution scripts
├── data/                  # Dataset storage
├── results/               # Evaluation results
└── logs/                  # Log files
```

## Usage on Slurm Cluster

### 1. Connect to Cluster

```bash
ssh your_username@slurm-head-1
cd /mnt/vast/home/your_username/text2sql
```

### 2. Submit Job

```bash
# Edit slurm script if needed
vim scripts/slurm_eval.sh

# Submit job
sbatch scripts/slurm_eval.sh

# Check status
squeue
```

### 3. Monitor Results

```bash
# Check logs
tail -f logs/slurm_*.out

# View results
ls -la results/
```

## Results

Results are saved in the `results/` directory with timestamps:

- `summary_YYYYMMDD_HHMMSS.json`: Metrics summary
- `full_results_YYYYMMDD_HHMMSS.json`: Detailed results
- `{model_name}_YYYYMMDD_HHMMSS.json`: Per-model results

Example summary:
```json
{
  "experiment_name": "bird_bench_eval",
  "models": {
    "Qwen3-8B": {
      "metrics": {
        "execution_accuracy": 0.850
      }
    },
    "OmniSQL-7B": {
      "metrics": {
        "execution_accuracy": 0.920
      }
    }
  }
}
```

## Extending the Framework

### Adding New Models

1. Create model configuration in `configs/models/`
2. If using different inference backend, extend `BaseModel` class
3. Add model to experiment configuration

### Adding New Datasets

1. Extend `BaseDataset` class in `src/datasets/`
2. Implement required methods: `download_and_setup()`, `load_data()`, etc.
3. Register in `Evaluator._create_dataset()`

### Adding New Metrics

1. Add metric function to `src/evaluation/metrics.py`
2. Register in `calculate_metrics()` function
3. Add to experiment configuration

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce `gpu_memory_utilization` in model config
2. **Model Loading Errors**: Check `trust_remote_code: true` in vLLM config
3. **Dataset Not Found**: Run `download_and_setup()` manually

### Debugging

```bash
# Run with debug logging
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml --log-level DEBUG

# Check GPU usage
nvidia-smi

# Monitor Slurm job
tail -f logs/slurm_*.err
```

## Performance Tips

- Use `tensor_parallel_size > 1` for multi-GPU inference
- Adjust `batch_size` based on GPU memory
- Set `gpu_memory_utilization: 0.9` for maximum utilization
- Use subset evaluation for quick testing

## License

MIT License 