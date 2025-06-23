#!/usr/bin/env python3
"""
Full Pipeline Test Script

This script runs a complete end-to-end test with actual model loading
but uses minimal examples and resources to verify everything works.

Usage:
    python scripts/test_full_pipeline.py
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_experiment_config, create_output_dir
from evaluation.evaluator import Evaluator

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_full_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def main():
    """Run complete pipeline test."""
    print("🚀 Text2SQL Full Pipeline Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load test configuration
        logger.info("Loading test configuration...")
        config = load_experiment_config("configs/experiments/test_setup.yaml")
        
        # Create output directory
        output_dir = create_output_dir(
            config["output"]["results_dir"],
            f"test_{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info(f"Results will be saved to: {output_dir}")
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = Evaluator(config)
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.run_evaluation(output_dir)
        
        # Print results
        print("\n" + "=" * 60)
        print("🎉 FULL PIPELINE TEST COMPLETED!")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            for metric, value in model_results["metrics"].items():
                print(f"  {metric}: {value:.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("✅ Ready to run full evaluation on cluster!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        print("Check logs for detailed error information")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 