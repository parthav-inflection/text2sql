import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

from agents.factory import AgentFactory, ModelFactory
from src.datasets.bird import BirdDataset
from src.datasets.financial import FinancialDataset
from utils.config import load_model_config
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluation engine for text2sql agents."""
    
    def __init__(self, experiment_config: Dict[str, Any]):
        self.config = experiment_config
        self.experiment_name = experiment_config["experiment_name"]
        self.output_dir = None
        
        # Initialize dataset
        self.dataset = self._create_dataset(experiment_config["dataset"])
        
        # Load agents (new architecture)
        self.agents = self._load_agents(experiment_config)
        
    def _create_dataset(self, dataset_config: Dict[str, Any]) -> Any:
        """Create dataset instance based on configuration."""
        dataset_name = dataset_config["name"].lower()
        
        if dataset_name == "bird":
            return BirdDataset(dataset_config)
        elif dataset_name == "financial":
            return FinancialDataset(dataset_config)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def _load_agents(self, experiment_config: Dict[str, Any]) -> List[Any]:
        """Load agent instances from configuration."""
        agents = []
        
        # Check if this is an ablation study with multiple agent configs
        if "agents" in experiment_config:
            # New ablation format: multiple agents with different configurations
            model_configs = experiment_config.get("models", [])
            if not model_configs:
                raise ValueError("No model configurations specified in experiment config")
            
            agent_configs = experiment_config["agents"]
            
            for model_config_path in model_configs:
                for agent_def in agent_configs:
                    agent_name = agent_def["name"]
                    agent_config_path = agent_def.get("config")
                    
                    try:
                        if agent_config_path is None:
                            # Baseline: use model directly without agent
                            model = ModelFactory.create_model(model_config_path)
                            
                            # Create a simple wrapper that acts like an agent
                            class BaselineAgent:
                                def __init__(self, name, model):
                                    self.name = name
                                    self.model = model
                                
                                def generate_sql(self, question: str, schema: str) -> str:
                                    prompt = self.model.format_prompt(question, schema)
                                    response = self.model.generate([prompt])[0]
                                    return response
                                
                                def cleanup(self):
                                    if hasattr(self.model, 'cleanup'):
                                        self.model.cleanup()
                            
                            agent = BaselineAgent(agent_name, model)
                        else:
                            # Create agent using factory
                            agent = AgentFactory.create_agent(
                                agent_config_path=agent_config_path,
                                model_config_path=model_config_path
                            )
                            # Override the name to include the agent configuration
                            agent.name = agent_name
                        
                        agents.append(agent)
                        logger.info(f"Created agent '{agent_name}' with model: {model_config_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create agent '{agent_name}' with model {model_config_path}: {e}")
                        raise
        else:
            # Legacy format: single agent_config with multiple models
            agent_config_path = experiment_config.get("agent_config", "configs/agents/standard_agent.yaml")
            model_configs = experiment_config.get("models", [])
            
            if not model_configs:
                raise ValueError("No model configurations specified in experiment config")
            
            for model_config_path in model_configs:
                try:
                    # Create agent using factory
                    agent = AgentFactory.create_agent(
                        agent_config_path=agent_config_path,
                        model_config_path=model_config_path
                    )
                    agents.append(agent)
                    logger.info(f"Created agent with model: {model_config_path}")
                except Exception as e:
                    logger.error(f"Failed to create agent with model {model_config_path}: {e}")
                    raise
        
        logger.info(f"Loaded {len(agents)} agents")
        return agents
    
    def run_evaluation(self, output_dir: str) -> Dict[str, Any]:
        """Run evaluation on all agents."""
        self.output_dir = output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup dataset
        logger.info("Setting up dataset...")
        self.dataset.download_and_setup()
        examples = self.dataset.load_data()
        
        logger.info(f"Starting evaluation on {len(examples)} examples")
        
        all_results = {}
        
        for i, agent in enumerate(self.agents, 1):
            logger.info(f"Evaluating agent {i}/{len(self.agents)}: {agent.name}")
            
            try:
                # Generate predictions using agent
                predictions = self._generate_predictions_with_agent(agent, examples)
                
                # Calculate metrics and get detailed execution results
                metrics, execution_results = calculate_metrics(
                    predictions, 
                    examples, 
                    self.dataset,
                    self.config["evaluation"]["metrics"]
                )
                
                # Store results
                agent_results = {
                    "agent_name": agent.name,
                    "model_name": agent.model.name if hasattr(agent.model, 'name') else 'unknown',
                    "model_path": getattr(agent.model, 'model_path', 'unknown'),
                    "metrics": metrics,
                    "num_examples": len(examples),
                    "timestamp": timestamp,
                    "modules": self._get_agent_module_info(agent)
                }
                
                if self.config["output"]["save_predictions"]:
                    agent_results["predictions"] = predictions
                
                if self.config["output"].get("save_detailed_logs", False):
                    # Clean up examples in execution_results to avoid duplication
                    for res in execution_results:
                        res.pop("example", None)
                    agent_results["detailed_results"] = execution_results
                
                all_results[agent.name] = agent_results
                
                # Save individual agent results
                self._save_agent_results(agent.name, agent_results, timestamp)
                
                logger.info(f"Completed evaluation for {agent.name}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate agent {agent.name}: {e}", exc_info=True)
                # Continue with other agents
                continue
            
            finally:
                # Cleanup agent to free GPU memory
                try:
                    agent.cleanup()
                except:
                    pass
        
        if not all_results:
            raise RuntimeError("No agents were successfully evaluated")
        
        # Save combined results
        self._save_combined_results(all_results, timestamp)
        
        logger.info("Evaluation completed")
        return all_results
    
    def _generate_predictions_with_agent(self, agent: Any, examples: List[Dict[str, Any]]) -> List[str]:
        """Generate predictions using an agent."""
        predictions = []
        batch_size = self.config["evaluation"]["batch_size"]
        
        # Process in batches
        for i in tqdm(range(0, len(examples), batch_size), desc=f"Generating with {agent.name}"):
            batch_examples = examples[i:i + batch_size]
            
            # Use agent to generate predictions
            batch_predictions = []
            for example in batch_examples:
                try:
                    # Get database schema
                    schema = self.dataset.get_schema(example)
                    
                    # Use agent to generate SQL
                    prediction = agent.generate_sql(example["question"], schema)
                    batch_predictions.append(prediction)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for example {i}: {e}")
                    batch_predictions.append("")  # Empty prediction on failure
            
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _get_agent_module_info(self, agent: Any) -> Dict[str, Any]:
        """Get information about agent's modules."""
        try:
            module_info = {}
            if hasattr(agent, 'pipeline') and hasattr(agent.pipeline, 'modules'):
                for module in agent.pipeline.modules:
                    module_info[module.name] = {
                        "enabled": getattr(module, 'enabled', True),
                        "type": type(module).__name__
                    }
            return module_info
        except Exception as e:
            logger.warning(f"Could not get module info: {e}")
            return {}
    
    def _save_agent_results(self, agent_name: str, results: Dict[str, Any], timestamp: str):
        """Save results for individual agent."""
        filename = f"{agent_name.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results for {agent_name} to {filepath}")
    
    def _save_combined_results(self, all_results: Dict[str, Any], timestamp: str):
        """Save combined results across all agents."""
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "dataset": self.config["dataset"]["name"],
            "num_examples": len(self.dataset.examples) if hasattr(self.dataset, 'examples') and self.dataset.examples else 0,
            "agents": {}
        }
        
        for agent_name, results in all_results.items():
            summary["agents"][agent_name] = {
                "model_name": results.get("model_name", "unknown"),
                "model_path": results.get("model_path", "unknown"),
                "metrics": results["metrics"],
                "modules": results.get("modules", {})
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
        for agent_name, results in all_results.items():
            print(f"\n{agent_name}:")
            print(f"  Model: {results.get('model_name', 'unknown')}")
            for metric, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
            if results.get("modules"):
                print(f"  Active modules: {', '.join(results['modules'].keys())}")
        print("="*50) 