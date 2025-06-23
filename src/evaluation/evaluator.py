import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

from models.vllm_model import VLLMModel
from datasets.bird import BirdDataset
from utils.config import load_model_config
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluation engine for text2sql models."""
    
    def __init__(self, experiment_config: Dict[str, Any]):
        self.config = experiment_config
        self.experiment_name = experiment_config["experiment_name"]
        self.output_dir = None
        
        # Initialize dataset
        self.dataset = self._create_dataset(experiment_config["dataset"])
        
        # Load models
        self.models = self._load_models(experiment_config["models"])
        
    def _create_dataset(self, dataset_config: Dict[str, Any]) -> Any:
        """Create dataset instance based on configuration."""
        dataset_name = dataset_config["name"].lower()
        
        if dataset_name == "bird":
            return BirdDataset(dataset_config)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def _load_models(self, model_configs: List[str]) -> List[Any]:
        """Load model instances from configuration files."""
        models = []
        
        for config_path in model_configs:
            model_config = load_model_config(config_path)
            
            if model_config["model_type"] == "vllm":
                model = VLLMModel(model_config)
                models.append(model)
            else:
                raise ValueError(f"Unsupported model type: {model_config['model_type']}")
        
        logger.info(f"Loaded {len(models)} models")
        return models
    
    def run_evaluation(self, output_dir: str) -> Dict[str, Any]:
        """Run evaluation on all models."""
        self.output_dir = output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup dataset
        self.dataset.download_and_setup()
        examples = self.dataset.load_data()
        
        logger.info(f"Starting evaluation on {len(examples)} examples")
        
        all_results = {}
        
        for model in self.models:
            logger.info(f"Evaluating model: {model.name}")
            
            # Generate predictions
            predictions = self._generate_predictions(model, examples)
            
            # Calculate metrics
            metrics = calculate_metrics(
                predictions, 
                examples, 
                self.dataset,
                self.config["evaluation"]["metrics"]
            )
            
            # Store results
            model_results = {
                "model_name": model.name,
                "model_path": model.model_path,
                "metrics": metrics,
                "num_examples": len(examples),
                "timestamp": timestamp
            }
            
            if self.config["output"]["save_predictions"]:
                model_results["predictions"] = predictions
                model_results["examples"] = examples
            
            all_results[model.name] = model_results
            
            # Save individual model results
            self._save_model_results(model.name, model_results, timestamp)
            
            # Cleanup model to free GPU memory
            model.cleanup()
        
        # Save combined results
        self._save_combined_results(all_results, timestamp)
        
        logger.info("Evaluation completed")
        return all_results
    
    def _generate_predictions(self, model: Any, examples: List[Dict[str, Any]]) -> List[str]:
        """Generate predictions for all examples."""
        predictions = []
        batch_size = self.config["evaluation"]["batch_size"]
        
        # Process in batches
        for i in tqdm(range(0, len(examples), batch_size), desc=f"Generating with {model.name}"):
            batch_examples = examples[i:i + batch_size]
            
            # Format prompts
            prompts = []
            for example in batch_examples:
                schema = self.dataset.get_schema(example)
                prompt = model.format_prompt(example["question"], schema)
                prompts.append(prompt)
            
            # Generate batch predictions
            batch_predictions = model.generate(prompts)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _save_model_results(self, model_name: str, results: Dict[str, Any], timestamp: str):
        """Save results for individual model."""
        filename = f"{model_name.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results for {model_name} to {filepath}")
    
    def _save_combined_results(self, all_results: Dict[str, Any], timestamp: str):
        """Save combined results across all models."""
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "dataset": self.config["dataset"]["name"],
            "num_examples": len(self.dataset.examples) if self.dataset.examples else 0,
            "models": {}
        }
        
        for model_name, results in all_results.items():
            summary["models"][model_name] = {
                "model_path": results["model_path"],
                "metrics": results["metrics"]
            }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save full results
        full_results_file = os.path.join(self.output_dir, f"full_results_{timestamp}.json")
        with open(full_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved combined results to {self.output_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        for model_name, results in all_results.items():
            print(f"\n{model_name}:")
            for metric, value in results["metrics"].items():
                print(f"  {metric}: {value:.3f}")
        print("="*50) 