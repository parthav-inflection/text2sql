#!/usr/bin/env python3
"""
Helper script to create a new dataset class for manually organized data.
Usage: python scripts/create_dataset.py --name financial
"""

import os
import argparse
import yaml
from typing import Dict, Any


def create_dataset_template(name: str) -> str:
    """Create a simple dataset Python file for pre-organized data."""
    
    class_name = f"{name.title()}Dataset"
    
    template = f'''import os
import json
import logging
from typing import List, Dict, Any, Tuple
from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class {class_name}(BaseDataset):
    """Implementation for the {name.upper()} text2sql dataset with manually organized data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.examples = None
        
    def download_and_setup(self):
        """Setup for manually organized dataset - just ensure directories exist."""
        self._ensure_directories()
        
        # Check if processed data exists
        processed_file = os.path.join(self.processed_dir, "{name}_data.json")
        if not os.path.exists(processed_file):
            raise FileNotFoundError(
                f"Processed dataset file not found: {{processed_file}}\\n"
                f"Please manually place your dataset file at this location."
            )
        
        logger.info("Dataset already organized manually - ready to use")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the manually organized {name.upper()} dataset."""
        if self.examples is not None:
            return self.examples
            
        # Ensure setup is complete
        self.download_and_setup()
        
        # Load processed data
        processed_file = os.path.join(self.processed_dir, "{name}_data.json")
        
        logger.info("Loading {name.upper()} dataset from manually organized files...")
        with open(processed_file, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)
        
        # Apply subset if specified
        if self.subset_size and self.subset_size < len(all_examples):
            all_examples = all_examples[:self.subset_size]
            logger.info(f"Using subset of {{self.subset_size}} examples")
        
        # Add schema information to each example using base class method
        for example in all_examples:
            if "schema" not in example:
                example["schema"] = self.get_schema(example)
        
        self.examples = all_examples
        logger.info(f"Loaded {{len(self.examples)}} {name.upper()} examples")
        return self.examples
    
    # get_schema() and execute_sql() are inherited from BaseDataset
    # They automatically use the standardized databases/{{db_id}}/{{db_id}}.sqlite structure
    
    def cleanup(self):
        """Clean up any resources."""
        # Base class handles cleanup automatically
        pass
'''
    
    return template


def create_dataset_config(name: str) -> Dict[str, Any]:
    """Create configuration template for manually organized dataset."""
    
    config = {
        "name": name,
        "subset_size": None,  # Use full dataset
        "data_dir": f"./data/{name}",
    }
    
    return config


def create_experiment_config(name: str) -> Dict[str, Any]:
    """Create experiment configuration template."""
    
    config = {
        "experiment_name": f"{name}_eval",
        "dataset": {
            "name": name,
            "subset_size": 10,  # Start with small subset for testing
            "data_dir": f"./data/{name}",
        },
        "models": [
            "configs/models/qwen3_0.6b_cpu.yaml"  # Start with CPU model for testing
        ],
        "evaluation": {
            "metrics": ["execution_accuracy"],
            "batch_size": 1,
        },
        "output": {
            "results_dir": "./results",
            "save_predictions": True,
            "save_detailed_logs": True,
        }
    }
    
    return config


def register_dataset_in_evaluator(name: str):
    """Add the new dataset to the evaluator registry."""
    evaluator_path = "src/evaluation/evaluator.py"
    
    # Read current file
    with open(evaluator_path, 'r') as f:
        content = f.read()
    
    # Add import
    import_line = f"from src.datasets.{name} import {name.title()}Dataset"
    if import_line not in content:
        # Find existing dataset imports
        lines = content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if line.startswith('from src.datasets.') and 'import' in line:
                insert_idx = i + 1
        
        if insert_idx:
            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
    
    # Add to registry in _create_dataset method
    registry_addition = f'''        elif dataset_name == "{name}":
            return {name.title()}Dataset(dataset_config)'''
    
    if registry_addition not in content:
        # Find the _create_dataset method
        if 'elif dataset_name == "bird":' in content:
            content = content.replace(
                'elif dataset_name == "bird":\n            return BirdDataset(dataset_config)',
                f'elif dataset_name == "bird":\n            return BirdDataset(dataset_config)\n{registry_addition}'
            )
    
    # Write back
    with open(evaluator_path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Create a new dataset class for manually organized data")
    parser.add_argument("--name", required=True, help="Dataset name (e.g., 'financial')")
    parser.add_argument("--skip_registration", action="store_true", help="Skip automatic registration in evaluator")
    
    args = parser.parse_args()
    
    name = args.name.lower()
    
    print(f"Creating dataset class for manually organized data: {name}")
    
    # Check if data directory exists and is organized
    data_dir = f"data/{name}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        print(f"Please create the directory structure first:")
        print(f"  {data_dir}/")
        print(f"  ├── processed/{name}_data.json")
        print(f"  ├── databases/{{db_id}}/{{db_id}}.sqlite")
        print(f"  └── raw/ (optional)")
        return
    
    processed_file = f"{data_dir}/processed/{name}_data.json"
    if not os.path.exists(processed_file):
        print(f"⚠️  Processed file not found: {processed_file}")
        print(f"The generated class will expect this file to exist.")
    
    # 1. Create dataset Python file
    dataset_file = f"src/datasets/{name}.py"
    if not os.path.exists(dataset_file):
        template = create_dataset_template(name)
        with open(dataset_file, 'w') as f:
            f.write(template)
        print(f"✅ Created dataset implementation: {dataset_file}")
    else:
        print(f"⚠️  Dataset file already exists: {dataset_file}")
    
    # 2. Create experiment config
    experiment_file = f"configs/experiments/{name}_test.yaml"
    if not os.path.exists(experiment_file):
        experiment_config = create_experiment_config(name)
        with open(experiment_file, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created experiment config: {experiment_file}")
    else:
        print(f"⚠️  Experiment config already exists: {experiment_file}")
    
    # 3. Register in evaluator
    if not args.skip_registration:
        try:
            register_dataset_in_evaluator(name)
            print(f"✅ Registered dataset in evaluator")
        except Exception as e:
            print(f"⚠️  Could not auto-register in evaluator: {e}")
            print("You'll need to manually add the import and registration")
    
    print(f"""
🎉 Dataset class '{name}' created successfully!

Your manually organized structure should be:
{data_dir}/
├── processed/
│   └── {name}_data.json          # Your dataset in standard format
├── databases/
│   ├── {{db_id1}}/
│   │   └── {{db_id1}}.sqlite      # Database files
│   └── {{db_id2}}/
│       └── {{db_id2}}.sqlite
└── raw/                          # (optional for manual setup)

Next steps:
1. Ensure your data is organized in the structure above
2. Test with: python scripts/run_eval.py --config {experiment_file}
3. Once working, scale up by removing subset_size in the experiment config

The dataset class expects:
- Standard JSON format in processed/{name}_data.json
- Each example should have: {{"question": "...", "db_id": "...", "SQL": "..."}}
- Databases organized as databases/{{db_id}}/{{db_id}}.sqlite
""")


if __name__ == "__main__":
    main() 