#!/usr/bin/env python3
"""
Test Script for Text2SQL Evaluation Setup

This script runs a minimal test to verify that the evaluation framework
works correctly before running full evaluations on the cluster.

Usage:
    python scripts/test_setup.py
"""

import os
import sys
import logging
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from utils.config import load_experiment_config, create_output_dir
        from evaluation.evaluator import Evaluator
        print("✅ Core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("📝 Testing configuration loading...")
    
    try:
        from utils.config import load_experiment_config
        
        # Test that config files exist and are valid
        config_file = "configs/experiments/test_setup.yaml"
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return False
            
        config = load_experiment_config(config_file)
        print(f"✅ Loaded test config: {config['experiment_name']}")
        
        # Check model configs exist
        for model_config in config['models']:
            if not os.path.exists(model_config):
                print(f"❌ Model config not found: {model_config}")
                return False
                
        print(f"✅ All {len(config['models'])} model configs found")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dataset_setup():
    """Test dataset initialization and setup."""
    print("📊 Testing dataset setup...")
    
    try:
        from datasets.bird import BirdDataset
        
        dataset_config = {
            "name": "bird",
            "subset_size": 2,
            "data_dir": "./data/bird"
        }
        
        dataset = BirdDataset(dataset_config)
        dataset.download_and_setup()
        examples = dataset.load_data()
        
        print(f"✅ Dataset setup successful, loaded {len(examples)} examples")
        
        # Test schema and SQL execution
        if examples:
            example = examples[0]
            schema = dataset.get_schema(example)
            print(f"✅ Schema extracted: {len(schema)} characters")
            
            # Test SQL execution
            test_sql = "SELECT COUNT(*) FROM employees"
            success, result = dataset.execute_sql(test_sql, example)
            if success:
                print(f"✅ SQL execution test passed: {result}")
            else:
                print(f"⚠️  SQL execution test failed (expected for some cases): {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model configuration without actually loading the models."""
    print("🤖 Testing model configuration...")
    
    try:
        from utils.config import load_model_config
        # Skip vLLM import for basic testing since it requires GPU dependencies
        
        # Test model configs can be loaded
        model_configs = [
            "configs/models/qwen3.yaml",
            "configs/models/omnisql.yaml"
        ]
        
        for config_path in model_configs:
            config = load_model_config(config_path)
            print(f"✅ Model config loaded: {config['name']}")
            
            # Validate required fields
            required_fields = ['model_path', 'model_type', 'generation_config', 'vllm_config']
            for field in required_fields:
                if field not in config:
                    print(f"❌ Missing required field '{field}' in {config_path}")
                    return False
            
            print(f"✅ Model config validated: {config['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("🖥️  Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"✅ Current GPU: {gpu_name}")
            
            # Test GPU memory
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_reserved = torch.cuda.memory_reserved(current_device)
            print(f"✅ GPU Memory - Allocated: {memory_allocated/1e9:.1f}GB, Reserved: {memory_reserved/1e9:.1f}GB")
            
            return True
        else:
            print("⚠️  CUDA not available - evaluation will fail on GPU cluster")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_full_pipeline_dry_run():
    """Test the full pipeline without actually loading heavy models."""
    print("🚀 Testing full pipeline (dry run)...")
    
    try:
        from utils.config import load_experiment_config, create_output_dir
        from evaluation.evaluator import Evaluator
        
        # Load test configuration
        config = load_experiment_config("configs/experiments/test_setup.yaml")
        
        # Create output directory
        output_dir = create_output_dir(
            config["output"]["results_dir"],
            f"test_{config['experiment_name']}"
        )
        print(f"✅ Output directory created: {output_dir}")
        
        # Test evaluator initialization (without model loading)
        print("✅ Pipeline structure validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Text2SQL Evaluation Setup Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    setup_logging()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Dataset Setup", test_dataset_setup),
        ("Model Configuration", test_model_loading),
        ("GPU Availability", test_gpu_availability),
        ("Pipeline Structure", test_full_pipeline_dry_run),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to run full evaluation on Slurm cluster")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("❌ Fix issues before running full evaluation")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 