# Text2SQL Agent Framework

A comprehensive framework for evaluating and improving text2sql models through **modular agent architectures**, **systematic ablation testing**, and **regularized dataset management** across multiple benchmarks.

## 🚀 Key Features

### Modular Agent Architecture
- **Factory Pattern**: Clean dependency injection for models, modules, and agents
- **Pipeline System**: Configurable module processing with priority ordering
- **Ablation Support**: Multi-agent comparison studies with comprehensive metrics
- **Configuration-Driven**: No code changes needed for experiments

### Production-Ready Evaluation
- **Multi-Dataset Support**: BIRD Mini-Dev + Financial dataset with standardized structure
- **Multi-Model Support**: vLLM-based models (XiYanSQL, Qwen3, OmniSQL) with CPU/GPU optimization
- **SLURM Integration**: HPC cluster deployment with job management
- **Comprehensive Metrics**: Execution accuracy, exact match, SQL validity with detailed logging

### Implemented Techniques & Testing
- **M-Schema Representation** ✅: Compact schema format with ablation studies
- **CPU Testing** ✅: Local development with Qwen3-0.6B (ultra-fast iteration)
- **GPU Evaluation** ✅: XiYanSQL-QwenCoder-7B production deployment
- **Ablation Framework** ✅: Compare agent configurations (with/without modules)
- **Dataset Standardization** ✅: 5-minute dataset addition with automated structure

## 🏗️ Architecture Overview

### Directory Structure

```
src/
├── agents/          # Agent implementations and factory
│   ├── base.py         # BaseAgent, AgentContext, StandardModulePipeline
│   ├── factory.py      # AgentFactory, ModelFactory, ModuleFactory
│   └── standard_agent.py  # StandardAgent implementation
├── models/          # Model implementations
│   ├── base.py         # BaseModel interface
│   └── vllm_model.py   # vLLM implementation with CPU/GPU support
├── modules/         # Modular improvement techniques
│   ├── base.py         # Base module classes
│   ├── schema/         # Schema processing modules
│   ├── generation/     # SQL candidate generation
│   ├── selection/      # Candidate selection
│   └── refinement/     # SQL refinement
├── datasets/        # Dataset implementations
│   ├── base.py         # BaseDataset interface
│   └── bird.py         # BIRD Mini-Dev dataset
├── evaluation/      # Evaluation framework
└── utils/          # Configuration utilities
```

### Agent Processing Pipeline

```
Question + Schema → AgentContext → Agent.process() → Final SQL
                         ↓
                  [Module Pipeline]
                         ↓
              ┌─ Schema Processing ─┐
              │   (M-Schema, etc.)  │
              └─────────┬───────────┘
                        ↓
              ┌─ Candidate Generation ─┐
              │  (Multi-strategies)     │
              └─────────┬───────────────┘
                        ↓
              ┌─ Selection & Refinement ─┐
              │   (Best candidate)       │
              └─────────┬─────────────────┘
                        ↓
                  Base Model Generation
```

## 🛠️ Quick Start

### 1. Local CPU Testing (Recommended First Step)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Test the full pipeline locally
python scripts/test_cpu_setup.py

# Run small evaluation (5 examples, ultra-fast)
python scripts/run_eval.py --config configs/experiments/test_cpu.yaml

# M-Schema ablation test (CPU, 5 examples)
python scripts/run_eval.py --config configs/experiments/financial_mschema_cpu_test.yaml
```

### 2. Production GPU Evaluation

```bash
# Financial dataset with XiYanSQL (full dataset)
python scripts/run_eval.py --config configs/experiments/financial_xiyanSQL_eval.yaml

# M-Schema ablation study (3-way comparison)
python scripts/run_eval.py --config configs/experiments/financial_xiyanSQL_mschema_ablation.yaml

# BIRD Mini-Dev evaluation (500 examples)
python scripts/run_eval.py --config configs/experiments/bird_eval.yaml

# SLURM cluster deployment
sbatch scripts/slurm_financial_eval.sh configs/experiments/financial_xiyanSQL_eval.yaml
```

## 🔧 Extending the Framework

### Adding New Models

1. **Create model configuration** in `configs/models/`:

```yaml
# configs/models/your_model.yaml
name: "YourModel"
model_path: "org/model-name"
model_type: "vllm"  # Currently only vLLM supported
generation_config:
  temperature: 0.0
  max_tokens: 256
  top_p: 1.0
vllm_config:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 8192
  trust_remote_code: true
  # For CPU: add device: "cpu", enforce_eager: true
```

**Available Models**:
- **XiYanSQL-QwenCoder-7B**: Production-ready model for complex financial queries
- **Qwen3-0.6B**: Ultra-fast CPU testing and development
- **OmniSQL**: General-purpose text2sql model

2. **For non-vLLM models**, extend `BaseModel`:

```python
# src/models/your_model.py
from src.models.base import BaseModel

class YourModel(BaseModel):
    def generate(self, prompts: List[str]) -> List[str]:
        # Implement your model's generation logic
        pass
    
    def format_prompt(self, question: str, schema: str) -> str:
        # Implement your model's prompt format
        pass
```

3. **Register in factory** (`src/models/base.py` or extend `ModelFactory`):

```python
# In ModelFactory.create_model()
elif model_type == 'your_type':
    return YourModel(config)
```

### Adding New Modules

1. **Choose module type** and create in appropriate subdirectory:

```python
# src/modules/schema/your_schema_module.py
from src.modules.base import SchemaProcessingModule

class YourSchemaModule(SchemaProcessingModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your module
        
    def process_schema(self, schema: str, context: AgentContext) -> str:
        """Transform the schema."""
        # Your schema processing logic
        processed_schema = self._your_processing(schema)
        
        # Store results in context
        context.set_module_output(self.name, {
            "original": schema,
            "processed": processed_schema
        })
        
        return processed_schema
```

2. **For other module types**:

```python
# Candidate Generation
class YourGenerationModule(CandidateGenerationModule):
    def generate_candidates(self, context: AgentContext) -> List[str]:
        # Generate multiple SQL candidates
        pass

# Selection
class YourSelectionModule(CandidateSelectionModule):
    def select_candidate(self, candidates: List[str], context: AgentContext) -> str:
        # Select best candidate
        pass

# Refinement  
class YourRefinementModule(SelfRefinementModule):
    def refine_sql(self, sql: str, context: AgentContext) -> str:
        # Iteratively improve SQL
        pass
```

3. **Register in ModuleFactory** (`src/agents/factory.py`):

```python
# In ModuleFactory.create_module()
module_registry = {
    'mschema': MSchemaModule,
    'your_module': YourModule,  # Add this line
    # ... other modules
}
```

4. **Configure in agent config**:

```yaml
# configs/agents/standard_agent.yaml
modules:
  your_module:
    enabled: true
    critical: false
    priority: 15  # Order in pipeline
    module_config:
      your_param: "value"
```

### Adding New Agents

1. **Create agent class**:

```python
# src/agents/your_agent.py
from src.agents.base import BaseAgent

class YourAgent(BaseAgent):
    def __init__(self, name: str, model: BaseModel, pipeline: ModulePipeline, config: Dict[str, Any]):
        super().__init__(name, model, pipeline, config)
        
    def process(self, context: AgentContext) -> str:
        """Override the main processing logic if needed."""
        # Custom processing logic
        # Or call super().process(context) for standard pipeline
        return super().process(context)
```

2. **Register in AgentFactory**:

```python
# In AgentFactory.create_agent()
elif agent_type == 'your_agent':
    return YourAgent(name, model, pipeline, agent_config)
```

3. **Create agent configuration**:

```yaml
# configs/agents/your_agent.yaml
name: "YourAgent"
type: "your_agent"  # Maps to factory
description: "Your custom agent"
modules:
  # Configure modules for your agent
```

### Adding New Datasets

**🚀 Quick Start**: Use the automated dataset creation script:

```bash
# Create a new dataset with automatic structure setup
python scripts/create_dataset.py --name spider \
  --main_url "https://yale-lily.github.io/spider" \
  --data_url "https://drive.google.com/file/d/1TqleXec_OykOYFREKKtschzY29dUcVAQ/view"

# This creates:
# - src/datasets/spider.py (implementation template)
# - configs/experiments/spider_test.yaml (test experiment)
# - data/spider/ (standardized directory structure)
# - Automatic registration in evaluator
```

**Current Datasets**:
- **BIRD Mini-Dev**: 500-example benchmark with automatic download
- **Financial**: Czech bank financial dataset (850+ examples) for complex analytical queries

**📁 Standardized Directory Structure**: Every dataset follows this pattern:

```
data/
└── {dataset_name}/
    ├── raw/              # Downloaded files (archives, etc.)
    │   ├── dataset.zip   # Original download
    │   └── extracted/    # Extracted contents
    ├── processed/        # Standardized dataset files
    │   └── {name}_data.json  # Main dataset in standard format
    └── databases/        # Organized database files
        ├── {db_id}/
        │   └── {db_id}.sqlite
        └── ...
```

**🛠️ Manual Implementation**: If you need custom processing:

```python
# src/datasets/your_dataset.py
from datasets.base import BaseDataset

class YourDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.examples = None
        
        # URLs can be in config or hardcoded here
        if not self.urls:
            self.urls = {
                "main": "https://your-dataset-website.com",
                "data": "https://download-url.com/data.zip"
            }
        
    def download_and_setup(self):
        """Download and setup using standardized utilities."""
        self._ensure_directories()  # Creates raw/, processed/, databases/
        
        # Check if already processed
        processed_file = os.path.join(self.processed_dir, "your_data.json")
        if os.path.exists(processed_file):
            return
            
        # Download dataset
        archive_path = os.path.join(self.raw_dir, "dataset.zip")
        if not os.path.exists(archive_path):
            self._download_file(self.urls["data"], archive_path, "Downloading dataset")
        
        # Extract dataset  
        extract_dir = os.path.join(self.raw_dir, "extracted")
        self._extract_archive(archive_path, extract_dir)
        
        # Process dataset files (implement your custom logic)
        main_file = self._find_file(extract_dir, "train.json")
        if main_file:
            self._process_dataset_file(main_file, processed_file)
        
        # Organize databases into standard structure
        self._organize_databases(extract_dir)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data using standard pattern."""
        if self.examples is not None:
            return self.examples
            
        self.download_and_setup()
        
        # Load processed data
        processed_file = os.path.join(self.processed_dir, "your_data.json")
        with open(processed_file, 'r') as f:
            all_examples = json.load(f)
        
        # Apply subset and add schemas (handled by base class)
        if self.subset_size and self.subset_size < len(all_examples):
            all_examples = all_examples[:self.subset_size]
            
        for example in all_examples:
            if "schema" not in example:
                example["schema"] = self.get_schema(example)
        
        self.examples = all_examples
        return self.examples
    
    # get_schema() and execute_sql() are provided by base class
    # They use the standard databases/{db_id}/{db_id}.sqlite structure
```

**🔧 Configuration**: The script automatically creates all configs, or you can create them manually:

```yaml
# configs/datasets/your_dataset.yaml
name: "your_dataset"
subset_size: null
data_dir: "./data/your_dataset"
urls:
  main: "https://your-dataset-website.com"
  data: "https://download-url.com/data.zip"
```

```yaml  
# configs/experiments/your_dataset_test.yaml
experiment_name: "your_dataset_eval"
dataset:
  name: "your_dataset"
  subset_size: 10  # Start small for testing
  data_dir: "./data/your_dataset"
models:
  - "configs/models/qwen3_0.6b_cpu.yaml"
evaluation:
  metrics: ["execution_accuracy"]
  batch_size: 1
output:
  results_dir: "./results"
  save_predictions: true
  save_detailed_logs: true
```

**✅ Testing Your Dataset**:

```bash
# Test with small subset first
python scripts/run_eval.py --config configs/experiments/your_dataset_test.yaml

# Scale up once working
# Edit config to remove subset_size or increase it
```

## 📊 Configuration System

### Standard Experiment Configuration

```yaml
# configs/experiments/your_experiment.yaml
experiment_name: "your_experiment"

dataset:
  name: "financial"  # or "bird"
  subset_size: 100  # null for full dataset
  data_dir: "./data/financial"

models:
  - "configs/models/xiyanSQL_qwencoder_7b.yaml"

evaluation:
  metrics: ["execution_accuracy", "exact_match", "valid_sql"]
  batch_size: 4
  timeout_seconds: 60

output:
  results_dir: "./results"
  save_predictions: true
  save_detailed_logs: true
```

### Ablation Study Configuration

```yaml
# configs/experiments/ablation_experiment.yaml
experiment_name: "mschema_ablation"

dataset:
  name: "financial"
  data_dir: "./data/financial"

models:
  - "configs/models/xiyanSQL_qwencoder_7b.yaml"

# Multiple agent configurations for comparison
agents:
  - name: "baseline_no_agent"
    config: null  # Raw model without agent processing
  - name: "without_mschema"
    config: "configs/agents/agent_without_mschema.yaml"
  - name: "with_mschema"
    config: "configs/agents/agent_with_mschema.yaml"

evaluation:
  metrics: ["execution_accuracy", "exact_match", "valid_sql"]
  batch_size: 4

output:
  results_dir: "./results"
  save_intermediate_outputs: true
  create_comparison_report: true
```

### Module Priorities

Modules execute in priority order (lower numbers first):
- **10-19**: Schema Processing (M-Schema, linking, compression)
- **20-29**: Candidate Generation (divide & conquer, skeleton ICL)
- **30-39**: Refinement (self-correction, execution feedback)
- **40-49**: Selection (ranking, confidence scoring)

## 🧪 Testing & Validation

### Development Workflow

1. **Ultra-Fast CPU Testing**: 5 examples in ~2-3 minutes with Qwen3-0.6B
2. **Ablation Studies**: Compare agent variants (with/without modules)
3. **Small Scale GPU**: 10-50 examples for configuration validation
4. **Production Evaluation**: Full datasets with comprehensive metrics

### Testing Commands

```bash
# Ultra-fast CPU development (5 examples, ~2 minutes)
python scripts/run_eval.py --config configs/experiments/financial_mschema_cpu_test.yaml

# Quick GPU validation (subset testing)
python scripts/run_eval.py --config configs/experiments/financial_test.yaml

# Full ablation study (production)
python scripts/run_eval.py --config configs/experiments/financial_xiyanSQL_mschema_ablation.yaml

# Component testing
python scripts/test_components.py
python scripts/test_cpu_setup.py
```

### Debugging Tools

```bash
# Test agent creation
python -c "from src.agents.factory import AgentFactory; agent = AgentFactory.create_agent('configs/agents/standard_agent.yaml', 'configs/models/qwen3_0.6b_cpu.yaml')"

# Validate configurations  
python -c "from src.utils.config import load_experiment_config; print(load_experiment_config('configs/experiments/test_cpu.yaml'))"
```

## 📈 Results & Analysis

The framework outputs comprehensive results with ablation support:

### Standard Evaluation Results:
- **Summary files**: High-level metrics and model comparison
- **Detailed logs**: Per-example predictions and intermediate outputs  
- **Module outputs**: Results from each pipeline stage
- **Execution logs**: SQL execution results and errors

### Ablation Study Results:
- **Comparison reports**: Side-by-side agent performance analysis
- **Module impact**: Quantified effect of each technique (e.g., M-Schema)
- **Statistical significance**: Confidence intervals and significance tests
- **Intermediate outputs**: Debug information from each agent configuration

### Example Output Structure:
```
results/financial_xiyanSQL_mschema_ablation_2024-06-24_19-30-45/
├── baseline_no_agent_results.json          # Raw model performance
├── without_mschema_results.json            # Agent without M-Schema
├── with_mschema_results.json               # Agent with M-Schema
├── comparison_report.json                  # Side-by-side analysis
├── summary_20240624_193045.json            # High-level metrics
└── full_results_20240624_193045.json       # Complete detailed results
```

Results are saved in timestamped directories under `./results/` for easy comparison and analysis.

## 📦 Regularized Dataset Structure

The framework now uses a **standardized structure** that makes adding new datasets trivial:

### 🎯 Standard Workflow

```bash
# 1. Create dataset structure automatically
python scripts/create_dataset.py --name mydataset \
  --data_url "https://example.com/data.zip"

# 2. Edit generated file for dataset-specific processing  
# src/datasets/mydataset.py

# 3. Test immediately
python scripts/run_eval.py --config configs/experiments/mydataset_test.yaml

# 4. Scale up once working
```

### 📁 Every Dataset Gets:

```
data/mydataset/
├── raw/                    # Downloads go here automatically
│   ├── dataset.zip         # Original download  
│   └── extracted/          # Auto-extracted contents
├── processed/              # Standardized format
│   └── mydataset_data.json # Standard schema
└── databases/              # Auto-organized databases  
    ├── db1/db1.sqlite      # Standard path resolution
    └── db2/db2.sqlite
```

### 🛠️ Built-in Utilities:
- **`_download_file()`**: Progress bars, resume support
- **`_extract_archive()`**: ZIP/TAR support  
- **`_organize_databases()`**: Auto-move to standard paths
- **`_find_file()`**: Recursive file search
- **`get_schema()` & `execute_sql()`**: Work automatically

### ✨ Benefits:
- **5-minute dataset addition** with script
- **Consistent structure** across all datasets
- **Automatic download/organization** 
- **Standard path resolution**
- **No manual database organization**

## 🔮 Roadmap

### Completed ✅
- **M-Schema Module**: Compact schema representation with ablation testing
- **Multi-Dataset Support**: BIRD + Financial datasets with standardized structure
- **Ablation Framework**: Multi-agent comparison studies
- **CPU Development**: Ultra-fast iteration with Qwen3-0.6B
- **GPU Production**: XiYanSQL deployment with SLURM integration
- **Dataset Standardization**: 5-minute dataset addition workflow

### In Progress 🚧
- **Schema Linking**: Table/column relevance scoring
- **Multi-Generator**: Divide & conquer, execution plans, skeleton ICL
- **Self-Refinement**: Iterative improvement with execution feedback
- **Selection Models**: Fine-tuned candidate ranking

### Architecture Improvements
- **Async Processing**: Parallel module execution
- **Caching**: Schema and intermediate result caching
- **Streaming**: Real-time evaluation progress
- **Web Interface**: Browser-based experiment management

## 🤝 Contributing

The modular architecture makes it easy to contribute:

1. **Fork** and create feature branch
2. **Add** your module/model/agent following the patterns above
3. **Test** with the CPU setup and small examples
4. **Document** your component in configs and docstrings
5. **Submit** PR with test results

Each component is self-contained and follows clear interfaces, making the codebase easy to extend and maintain.

## 🎯 Available Experiments

### CPU Development & Testing
- **`test_cpu.yaml`**: Basic CPU testing with Qwen3-0.6B (5 examples)
- **`financial_mschema_cpu_test.yaml`**: M-Schema ablation on CPU (5 examples, 3 agents)

### GPU Production Evaluation  
- **`bird_eval.yaml`**: BIRD Mini-Dev evaluation (500 examples)
- **`financial_test.yaml`**: Financial dataset quick test (subset)
- **`financial_xiyanSQL_eval.yaml`**: Full financial evaluation with XiYanSQL
- **`financial_xiyanSQL_mschema_ablation.yaml`**: Complete M-Schema ablation study

### Quick Reference Commands
```bash
# Development (CPU, ~2-3 minutes)
python scripts/run_eval.py --config configs/experiments/financial_mschema_cpu_test.yaml

# Production (GPU, ~30-60 minutes)  
python scripts/run_eval.py --config configs/experiments/financial_xiyanSQL_mschema_ablation.yaml

# SLURM Deployment
sbatch scripts/slurm_financial_eval.sh configs/experiments/financial_xiyanSQL_eval.yaml
```

The framework is designed for **rapid iteration** (CPU) → **validation** (GPU subset) → **production** (full datasets) workflow. 