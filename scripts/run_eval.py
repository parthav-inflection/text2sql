#!/usr/bin/env python3
"""
Text2SQL Evaluation Script

Usage:
    python scripts/run_eval.py --config configs/experiments/bird_eval.yaml
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_experiment_config, create_output_dir
from evaluation.evaluator import Evaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_file = f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Log file location
    logger = logging.getLogger(__name__)
    logger.info(f"Detailed logs saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Text2SQL Evaluation")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default="configs/agents/standard_agent.yaml",
        help="Path to agent configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load experiment configuration
        logger.info(f"Loading experiment config from: {args.config}")
        experiment_config = load_experiment_config(args.config)
        
        # Add agent config to experiment config
        experiment_config["agent_config"] = args.agent_config
        
        # Create output directory
        output_dir = create_output_dir(
            experiment_config["output"]["results_dir"],
            experiment_config["experiment_name"]
        )
        logger.info(f"Results will be saved to: {output_dir}")
        
        # Initialize evaluator
        logger.info("Initializing evaluator with agent-based architecture")
        evaluator = Evaluator(experiment_config)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation(output_dir)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print quick summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment: {experiment_config['experiment_name']}")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 