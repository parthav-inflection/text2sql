import yaml
import os
from typing import Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    return load_yaml(config_path)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    return load_yaml(config_path)


def create_output_dir(results_dir: str, experiment_name: str) -> str:
    """Create output directory for experiment results."""
    output_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """Load any configuration file (alias for load_yaml)."""
    return load_yaml(config_path) 