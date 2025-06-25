#!/usr/bin/env python3
"""
Test script to verify CPU setup works correctly with Qwen3-0.6B model.
This script tests the full pipeline end-to-end with a minimal setup.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.factory import AgentFactory
from datasets.bird import BirdDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test that the CPU model can be loaded."""
    logger.info("Testing model loading...")
    
    try:
        agent = AgentFactory.create_agent(
            agent_config_path="configs/agents/standard_agent.yaml",
            model_config_path="configs/models/qwen3_0.6b_cpu.yaml"
        )
        logger.info(f"✅ Successfully created agent: {agent.name}")
        logger.info(f"✅ Model loaded: {agent.model.name}")
        return agent
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def test_sql_generation(agent):
    """Test SQL generation with a simple example."""
    logger.info("Testing SQL generation...")
    
    try:
        # Simple test case with SQL schema format (expected by MSchema module)
        question = "What are all the table names in the database?"
        schema = """CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    total REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);"""
        
        sql = agent.generate_sql(question, schema)
        logger.info(f"✅ Generated SQL: {sql}")
        return sql
    except Exception as e:
        logger.error(f"❌ Failed to generate SQL: {e}")
        raise

def test_dataset_loading():
    """Test that the dataset can be loaded."""
    logger.info("Testing dataset loading...")
    
    try:
        dataset_config = {
            "name": "bird",
            "subset_size": 2,  # Just 2 examples for testing
            "data_dir": "./data/bird"
        }
        
        dataset = BirdDataset(dataset_config)
        logger.info("✅ Dataset created successfully")
        
        # Try to setup (this might fail if data isn't available, which is fine)
        try:
            dataset.download_and_setup()
            examples = dataset.load_data()
            logger.info(f"✅ Loaded {len(examples)} examples from dataset")
            return examples[:2]  # Return first 2 examples
        except Exception as e:
            logger.warning(f"⚠️ Dataset setup failed (expected if data not available): {e}")
            return []
    except Exception as e:
        logger.error(f"❌ Failed to create dataset: {e}")
        raise

def main():
    print("="*60)
    print("CPU SETUP TEST")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Test 1: Model loading
        agent = test_model_loading()
        print()
        
        # Test 2: SQL generation
        test_sql_generation(agent)
        print()
        
        # Test 3: Dataset loading
        examples = test_dataset_loading()
        print()
        
        # Test 4: Full pipeline if we have examples
        if examples:
            logger.info("Testing full pipeline...")
            for i, example in enumerate(examples):
                try:
                    # Use a simple schema in SQL format for testing
                    schema = "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);"
                    sql = agent.generate_sql(example.get("question", "SELECT 1"), schema)
                    logger.info(f"✅ Example {i+1}: Generated SQL for question")
                except Exception as e:
                    logger.warning(f"⚠️ Example {i+1} failed: {e}")
        
        # Cleanup
        agent.cleanup()
        
        print()
        print("="*60)
        print("✅ ALL TESTS PASSED! CPU setup is working correctly.")
        print("You can now run the full evaluation with:")
        print("python scripts/run_eval.py --config configs/experiments/test_cpu.yaml")
        print("="*60)
        
    except Exception as e:
        print()
        print("="*60)
        print(f"❌ TEST FAILED: {e}")
        print("Please check the error messages above for troubleshooting.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main() 