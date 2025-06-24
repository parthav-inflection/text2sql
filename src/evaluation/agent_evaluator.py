import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from src.models.vllm_model import VLLMModel
from src.datasets.bird import BirdDataset
from src.utils.config import load_model_config, load_yaml
from src.agents.base import AgentContext
from src.agents.standard_agent import StandardAgent
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Evaluator for agent-based text2sql systems with ablation testing support."""
    
    def __init__(self, experiment_config: Dict[str, Any]):
        self.config = experiment_config
        self.experiment_name = experiment_config["experiment_name"]
        self.output_dir = None
        
        # Initialize dataset
        self.dataset = self._create_dataset(experiment_config["dataset"])
        
        # Load models
        self.models = self._load_models(experiment_config["models"])
        
        # Load agent configurations
        self.agent_configs = experiment_config.get("agents", [])
        
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
    
    def _create_agent(self, agent_config: Dict[str, Any], model: Any) -> Optional[Any]:
        """Create an agent instance from configuration."""
        if agent_config.get("config") is None:
            # No agent config means use raw model
            return None
            
        # Load agent configuration
        agent_config_path = agent_config["config"]
        base_config = load_yaml(agent_config_path)
        
        # Apply module overrides for ablation testing
        if "module_overrides" in agent_config:
            self._apply_module_overrides(base_config, agent_config["module_overrides"])
        
        # Create agent (currently only StandardAgent is supported)
        agent = StandardAgent(base_config, model)
        
        return agent
    
    def _apply_module_overrides(self, base_config: Dict[str, Any], overrides: Dict[str, Any]):
        """Apply module configuration overrides for ablation testing."""
        if "modules" not in base_config:
            base_config["modules"] = {}
            
        for module_name, override_config in overrides.items():
            if module_name not in base_config["modules"]:
                base_config["modules"][module_name] = {}
            
            # Update module configuration
            base_config["modules"][module_name].update(override_config)
    
    def run_evaluation(self, output_dir: str) -> Dict[str, Any]:
        """Run evaluation with ablation testing."""
        self.output_dir = output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup dataset
        self.dataset.download_and_setup()
        examples = self.dataset.load_data()
        
        logger.info(f"Starting evaluation on {len(examples)} examples")
        
        all_results = {}
        
        for model in self.models:
            model_results = {}
            
            for agent_config in self.agent_configs:
                agent_name = agent_config["name"]
                logger.info(f"Evaluating {model.name} with agent: {agent_name}")
                
                # Create agent (or None for baseline)
                agent = self._create_agent(agent_config, model)
                
                # Generate predictions
                predictions, intermediate_outputs = self._generate_predictions_with_agent(
                    model, agent, examples
                )
                
                # Calculate metrics
                metrics = calculate_metrics(
                    predictions, 
                    examples, 
                    self.dataset,
                    self.config["evaluation"]["metrics"]
                )
                
                # Store results
                agent_results = {
                    "model_name": model.name,
                    "agent_name": agent_name,
                    "agent_config": agent_config,
                    "metrics": metrics,
                    "num_examples": len(examples),
                    "timestamp": timestamp
                }
                
                if self.config["output"]["save_predictions"]:
                    agent_results["predictions"] = predictions
                
                if self.config["output"].get("save_intermediate_outputs", False):
                    agent_results["intermediate_outputs"] = intermediate_outputs
                
                model_results[agent_name] = agent_results
                
                # Save individual results
                self._save_agent_results(model.name, agent_name, agent_results, timestamp)
            
            all_results[model.name] = model_results
            
            # Cleanup model to free GPU memory
            model.cleanup()
        
        # Save combined results and create comparison report
        self._save_combined_results(all_results, timestamp)
        
        if self.config["output"].get("create_comparison_report", False):
            self._create_comparison_report(all_results, timestamp)
        
        logger.info("Evaluation completed")
        return all_results
    
    def _generate_predictions_with_agent(
        self, 
        model: Any, 
        agent: Optional[Any], 
        examples: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Generate predictions using agent or raw model."""
        predictions = []
        intermediate_outputs = []
        batch_size = self.config["evaluation"]["batch_size"]
        
        # Process in batches
        for i in tqdm(range(0, len(examples), batch_size), 
                      desc=f"Generating with {model.name}"):
            batch_examples = examples[i:i + batch_size]
            
            if agent is None:
                # Use raw model
                batch_predictions, batch_intermediates = self._generate_with_model(
                    model, batch_examples
                )
            else:
                # Use agent
                batch_predictions, batch_intermediates = self._generate_with_agent(
                    agent, batch_examples
                )
            
            predictions.extend(batch_predictions)
            intermediate_outputs.extend(batch_intermediates)
        
        return predictions, intermediate_outputs
    
    def _generate_with_model(
        self, 
        model: Any, 
        examples: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Generate predictions using raw model."""
        prompts = []
        for example in examples:
            schema = self.dataset.get_schema(example)
            prompt = model.format_prompt(example["question"], schema)
            prompts.append(prompt)
        
        predictions = model.generate(prompts)
        
        # No intermediate outputs for raw model
        intermediates = [{"type": "raw_model"} for _ in predictions]
        
        return predictions, intermediates
    
    def _generate_with_agent(
        self, 
        agent: Any, 
        examples: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Generate predictions using agent."""
        predictions = []
        intermediates = []
        
        for example in examples:
            # Create context for this example
            context = AgentContext(
                question=example["question"],
                example=example,
                dataset=self.dataset
            )
            
            # Process with agent
            prediction = agent.process(context)
            predictions.append(prediction)
            
            # Collect intermediate outputs
            intermediate = {
                "type": "agent",
                "agent_name": agent.name,
                "module_outputs": context.module_outputs,
                "sql_candidates": context.sql_candidates,
                "processed_schema": context.processed_schema
            }
            intermediates.append(intermediate)
        
        return predictions, intermediates
    
    def _save_agent_results(
        self, 
        model_name: str, 
        agent_name: str, 
        results: Dict[str, Any], 
        timestamp: str
    ):
        """Save results for individual agent configuration."""
        filename = f"{model_name.replace('/', '_')}_{agent_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results for {model_name}+{agent_name} to {filepath}")
    
    def _save_combined_results(self, all_results: Dict[str, Any], timestamp: str):
        """Save combined results across all configurations."""
        # Save full results
        full_results_file = os.path.join(self.output_dir, f"ablation_results_{timestamp}.json")
        with open(full_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved combined results to {full_results_file}")
    
    def _create_comparison_report(self, all_results: Dict[str, Any], timestamp: str):
        """Create a comparison report for ablation study."""
        report_lines = []
        report_lines.append("# Ablation Study Report")
        report_lines.append(f"Experiment: {self.experiment_name}")
        report_lines.append(f"Generated: {timestamp}")
        report_lines.append("")
        
        for model_name, model_results in all_results.items():
            report_lines.append(f"## Model: {model_name}")
            report_lines.append("")
            
            # Create comparison table
            agent_names = list(model_results.keys())
            if not agent_names:
                continue
                
            # Get metric names from first agent
            first_agent = model_results[agent_names[0]]
            metric_names = list(first_agent["metrics"].keys())
            
            # Table header
            header = "| Agent | " + " | ".join(metric_names) + " |"
            separator = "|-------|" + "|".join(["-------"] * len(metric_names)) + "|"
            
            report_lines.append(header)
            report_lines.append(separator)
            
            # Table rows
            for agent_name in agent_names:
                agent_results = model_results[agent_name]
                metrics = agent_results["metrics"]
                
                row = f"| {agent_name} |"
                for metric_name in metric_names:
                    value = metrics.get(metric_name, 0.0)
                    row += f" {value:.3f} |"
                
                report_lines.append(row)
            
            report_lines.append("")
        
        # Save report
        report_file = os.path.join(self.output_dir, f"comparison_report_{timestamp}.md")
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Created comparison report: {report_file}")
        
        # Also print to console
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        for line in report_lines[3:]:  # Skip markdown header
            if line.startswith("|") or line.startswith("#"):
                print(line)
        print("="*60) 