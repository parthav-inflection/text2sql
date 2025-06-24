# Text2SQL Evaluation Framework

A comprehensive framework for evaluating and improving text2sql models through **modular agent architectures** and **ablation testing** on the BIRD Mini-Dev benchmark.

## 🚀 Key Features

### Core Evaluation
- **BIRD Mini-Dev Dataset**: Full integration with official BIRD Mini-Dev dataset (500 examples)
- **Model Support**: vLLM-based models (Qwen3-8B, OmniSQL-7B) with optimized prompting
- **GPU Optimized**: Efficient inference on NVIDIA GPUs with Slurm compatibility

### 🧠 Agent Architecture (NEW)
- **Modular Agents**: Extensible pipeline with pluggable improvement modules
- **Ablation Testing**: Compare techniques individually and in combination
- **Configuration-Driven**: No code changes needed for experiments
- **Intermediate Analysis**: Capture module outputs for detailed insights

### 📊 Implemented Techniques
- **M-Schema Representation** ✅: Compact schema format (XiYan-SQL)
- **Schema Processing Pipeline** ✅: Extensible schema transformation
- **Enhanced Prompting** ✅: Clean SQL output without explanations

### 🔬 Planned Techniques
- **Schema Compression & Linking**: Relevant table/column identification
- **Multi-Generator Candidates**: Divide & conquer, execution plan, skeleton ICL
- **Self-Refinement**: Iterative SQL correction with execution feedback
- **Fine-tuned Selection**: DPO-trained candidate selection model
- **Column Exploration**: Dynamic database sampling for ambiguity resolution

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data results logs
```

### 2. Basic Evaluation

```bash
# Test with small subset (5 examples)
python scripts/run_eval.py --config configs/experiments/test_setup.yaml

# Full Mini-Dev evaluation (500 examples)
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml
```

### 3. Agent-Based Evaluation (Recommended)

```bash
# Run ablation test with M-Schema module
python scripts/run_agent_eval.py --config configs/experiments/ablation_mschema.yaml

# Test agent architecture
python scripts/test_agent_architecture.py
```

### 4. Cluster Deployment

```bash
# Submit to Slurm
sbatch scripts/slurm_eval.sh
```

## 🏗️ Architecture Overview

### Agent-Based Pipeline

The framework now features a **modular agent architecture** that allows systematic testing of text2sql improvements:

```
User Query + Database → AgentContext → StandardAgent → Module Pipeline → Enhanced SQL
                                           ↓
                                    [Schema Processing]
                                           ↓
                                    [Candidate Generation] 
                                           ↓
                                    [Selection & Refinement]
                                           ↓
                                      Base Model → Final SQL
```

### Core Components

- **Agents** (`src/agents/`): Orchestrate the text2sql process using configurable modules
- **Modules** (`src/modules/`): Implement specific improvement techniques
- **Pipeline**: Processes queries through prioritized module sequence
- **Configuration**: YAML-based setup for easy experimentation

## 📊 BIRD Mini-Dev Dataset

The framework uses the official **BIRD Mini-Dev dataset** with:
- **500 high-quality text2SQL pairs** from 11 databases
- **Difficulty distribution**: 30% simple, 50% moderate, 20% challenging
- **Automatic download** from official sources
- **SQLite database execution** for evaluation

### Dataset Statistics
- Total examples: 500
- Databases: 11 (debit_card_specializing, financial, formula_1, etc.)
- Fields: question_id, db_id, question, evidence, SQL, difficulty

## 📋 Module System

### Implemented Modules

#### M-Schema Module ✅
Converts database schemas to M-Schema format (from XiYan-SQL paper):

**Input (CREATE TABLE):**
```sql
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE
);
```

**Output (M-Schema):**
```
(DB_ID) ecommerce_db

# Table customers (3 columns)
  (id, INTEGER, "unique identifier", PK)
  (name, TEXT, "name field", )
  (email, TEXT, "email address", )
```

### Module Categories

- **Schema Processing** (`src/modules/schema/`): Transform and enhance schema representation
- **Candidate Generation** (`src/modules/generation/`): Generate multiple SQL candidates
- **Selection** (`src/modules/selection/`): Choose best candidate from multiple options
- **Refinement** (`src/modules/refinement/`): Iteratively improve SQL queries

### Adding New Modules

1. Inherit from appropriate base class (`SchemaProcessingModule`, `CandidateGenerationModule`, etc.)
2. Implement required abstract methods
3. Add to agent factory method
4. Configure in YAML files

```python
class SchemaLinkingModule(SchemaProcessingModule):
    def process_schema(self, schema: str, context: AgentContext) -> str:
        # Implement schema linking logic
        return filtered_schema
```

## ⚙️ Configuration System

### Agent Configuration

Configure agents and modules in `configs/agents/`:

```yaml
# configs/agents/standard_agent.yaml
name: "StandardAgent"
description: "Standard Text2SQL agent with configurable modules"

modules:
  mschema:
    enabled: true
    critical: false
    priority: 10
    module_config:
      include_samples: true
      include_descriptions: true
      max_sample_values: 3
  
  # Future modules:
  # schema_linking:
  #   enabled: false
  #   priority: 20
  #   module_config:
  #     similarity_threshold: 0.7
```

### Ablation Experiment Configuration

Configure ablation tests in `configs/experiments/`:

```yaml
# configs/experiments/ablation_mschema.yaml
experiment_name: "ablation_mschema_test"

dataset:
  name: "bird"
  subset_size: 100  # Small subset for quick testing

models:
  - "configs/models/omnisql.yaml"

# Agent configurations for ablation
agents:
  - name: "baseline"
    config: null  # Use raw model without agent
    
  - name: "with_mschema"
    config: "configs/agents/standard_agent.yaml"
    module_overrides:
      mschema:
        enabled: true

evaluation:
  metrics: ["execution_accuracy", "exact_match", "valid_sql"]
  save_intermediate_outputs: true  # Capture module outputs
  create_comparison_report: true   # Generate ablation report
```

### Model Configuration

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

## 📁 Project Structure

```
text2sql/
├── configs/                 # Configuration files
│   ├── agents/             # Agent configurations
│   ├── models/             # Model configurations
│   └── experiments/        # Experiment configurations (basic + ablation)
├── src/                    # Source code
│   ├── agents/            # 🆕 Agent orchestration (BaseAgent, StandardAgent)
│   ├── modules/           # 🆕 Pluggable improvement modules
│   │   ├── schema/        # Schema processing (M-Schema, etc.)
│   │   ├── generation/    # SQL candidate generation
│   │   ├── selection/     # Candidate selection
│   │   └── refinement/    # SQL refinement
│   ├── models/            # Model implementations (vLLM)
│   ├── datasets/          # BIRD Mini-Dev dataset implementation
│   ├── evaluation/        # Evaluation (basic + agent-based)
│   └── utils/             # Utilities
├── scripts/               # Execution scripts
│   ├── run_eval.py        # Basic evaluation
│   ├── run_agent_eval.py  # 🆕 Agent-based evaluation
│   └── test_agent_architecture.py  # 🆕 Architecture tests
├── data/                  # BIRD Mini-Dev dataset storage
├── results/               # Evaluation results + ablation reports
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

## 📊 Results & Ablation Testing

### Basic Evaluation Results

Results are saved in `results/` with detailed metrics:

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

### Ablation Testing Results

Agent-based evaluation generates **comparison reports** showing technique impact:

```markdown
# Ablation Study Report
Experiment: ablation_mschema_test

## Model: OmniSQL-7B

| Agent | execution_accuracy | exact_match | valid_sql |
|-------|-------------------|-------------|-----------|
| baseline | 0.650 | 0.420 | 0.890 |
| with_mschema | 0.680 | 0.450 | 0.895 |

```

### Intermediate Analysis

With `save_intermediate_outputs: true`, capture detailed module information:
- Schema transformations (original → M-Schema)
- SQL candidates generated by each technique
- Module execution timing and success rates
- Step-by-step pipeline analysis

## 🚀 Development Roadmap

### Implemented ✅
1. **Agent Architecture**: Modular pipeline with configurable modules
2. **M-Schema Representation**: XiYan-SQL schema format implementation
3. **Ablation Testing**: Compare techniques individually and in combination
4. **BIRD Dataset Integration**: Official Mini-Dev dataset with 500 examples
5. **Enhanced Evaluation**: Intermediate output capture and analysis

### Next Phase 🔄
1. **Schema Compression & Linking** (Priority 20)
   - Relevant table/column identification 
   - Vector similarity search for schema filtering
   - Semantic matching between questions and schema elements

2. **Multi-Generator Candidate Creation** (Priority 30-40)
   - **Divide & Conquer CoT**: Break complex queries into sub-problems
   - **Execution Plan CoT**: Database engine reasoning simulation
   - **Skeleton-Based ICL**: Structure-focused few-shot examples

3. **Self-Refinement & Selection** (Priority 40-50)
   - **Iterative Correction**: Execute SQL, fix errors, retry
   - **Fine-tuned Selection**: DPO-trained model for candidate ranking
   - **Column Exploration**: Dynamic sampling for ambiguity resolution

### Future Enhancements 🔮
- **Production Optimization**: Multi-GPU scheduling and serving
- **Custom Base Models**: Fine-tuned models for specific SQL dialects
- **Advanced Evaluation**: Semantic equivalence beyond execution accuracy

## 🔧 Extending the Framework

### Adding New Modules

1. **Choose Base Class**: Inherit from appropriate module type
   ```python
   from src.modules.base import SchemaProcessingModule
   
   class SchemaLinkingModule(SchemaProcessingModule):
       def process_schema(self, schema: str, context: AgentContext) -> str:
           # Implement your technique here
           return enhanced_schema
   ```

2. **Add to Agent Factory**: Register in `StandardAgent._create_module()`
   ```python
   elif module_name == "schema_linking":
       return SchemaLinkingModule(module_config)
   ```

3. **Configure**: Add to agent YAML configuration
   ```yaml
   modules:
     schema_linking:
       enabled: true
       priority: 20
       module_config:
         similarity_threshold: 0.7
   ```

### Module Execution Order

Modules execute by **priority** (lower numbers first):
- **10-20**: Schema processing (M-Schema, Schema Linking)
- **30-40**: Candidate generation (Divide & Conquer, Execution Plan)
- **50+**: Selection and refinement (Fine-tuned Selection, Self-Refinement)

### Adding New Models

1. Create model configuration in `configs/models/`
2. If using different inference backend, extend `BaseModel` class
3. Add model to experiment configuration

### Adding New Datasets

1. Extend `BaseDataset` class in `src/datasets/`
2. Implement required methods: `download_and_setup()`, `load_data()`, etc.
3. Register in `Evaluator._create_dataset()`

## Performance Tips

- Use `tensor_parallel_size > 1` for multi-GPU inference
- Adjust `batch_size` based on GPU memory
- Set `gpu_memory_utilization: 0.9` for maximum utilization
- Use subset evaluation for quick testing (`subset_size: 10`)

## License

MIT License 