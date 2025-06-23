# Text2SQL Evaluation Framework

A scalable framework for evaluating text2sql models on the **BIRD Mini-Dev benchmark** with improved prompting strategies.

## Features

- **BIRD Mini-Dev Dataset**: Full integration with the official BIRD Mini-Dev dataset (500 examples)
- **Model Support**: vLLM-based models (Qwen3-8B, OmniSQL-7B) with optimized prompting
- **Improved Prompting**: Clean SQL output format without explanations or markdown
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
# Test with small subset (5 examples)
python scripts/run_eval.py --config configs/experiments/test_setup.yaml

# Full Mini-Dev evaluation (500 examples)
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml

# Or submit to Slurm
sbatch scripts/slurm_eval.sh
```

## BIRD Mini-Dev Dataset

The framework now uses the official **BIRD Mini-Dev dataset** with:
- **500 high-quality text2SQL pairs** from 11 databases
- **Difficulty distribution**: 30% simple, 50% moderate, 20% challenging
- **Automatic download** from official sources
- **SQLite database execution** for evaluation

### Dataset Statistics
- Total examples: 500
- Databases: 11 (debit_card_specializing, financial, formula_1, etc.)
- Fields: question_id, db_id, question, evidence, SQL, difficulty

## Improved Prompting Strategy

Updated prompting to ensure models output clean SQL queries:

```
You are a SQL expert. Given the database schema and question, generate a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}

Instructions:
- Return only the SQL query, no explanations or additional text
- Use proper SQL syntax for SQLite
- Do not include markdown formatting or code blocks
- End the query with a semicolon

SQL Query:
```

## Configuration

### Model Configuration

Models are configured in `configs/models/`:

```yaml
# configs/models/omnisql.yaml
name: "OmniSQL-7B"
model_path: "seeklhy/OmniSQL-7B"
model_type: "vllm"
generation_config:
  temperature: 0.0
  max_tokens: 512
vllm_config:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
```

### Experiment Configuration

```yaml
# configs/experiments/bird_eval.yaml
experiment_name: "bird_minidev_eval"
dataset:
  name: "bird"
  subset_size: null  # Use full Mini-Dev dataset (500 examples)
models:
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
│   ├── datasets/          # BIRD Mini-Dev dataset implementation
│   ├── evaluation/        # Evaluation with improved SQL extraction
│   └── utils/             # Utilities
├── scripts/               # Execution scripts
├── data/                  # BIRD Mini-Dev dataset storage
│   └── bird/
│       └── mini_dev_data/ # Downloaded dataset files
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

## Results

Results are saved in the `results/` directory with detailed metrics:

- **Execution Accuracy**: Compares SQL query results against ground truth
- **Enhanced SQL Extraction**: Robust parsing of model outputs
- **Per-example Analysis**: Detailed success/failure tracking

Example summary:
```json
{
  "experiment_name": "bird_minidev_eval",
  "dataset": "bird",
  "num_examples": 500,
  "models": {
    "OmniSQL-7B": {
      "metrics": {
        "execution_accuracy": 0.824
      }
    }
  }
}
```

## Key Improvements

### 1. Real Dataset Integration
- Replaced placeholder data with official BIRD Mini-Dev dataset
- Automatic download and extraction from official sources
- Full 500-example evaluation capability

### 2. Enhanced Prompting
- Clear instructions for SQL-only output
- Removal of explanatory text and formatting
- Optimized for model understanding

### 3. Robust SQL Extraction
- Multiple parsing strategies for different model outputs
- Handles various response formats (code blocks, plain text)
- Improved accuracy in SQL query extraction

### 4. Production Ready
- No more placeholder implementations
- Full pipeline testing
- Ready for large-scale evaluation

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

## Performance Tips

- Use `tensor_parallel_size > 1` for multi-GPU inference
- Adjust `batch_size` based on GPU memory
- Set `gpu_memory_utilization: 0.9` for maximum utilization
- Use subset evaluation for quick testing (`subset_size: 10`)

## License

MIT License 