#!/usr/bin/env python3
"""
Test script to verify the agent architecture works correctly.

Usage:
    python scripts/test_agent_architecture.py
"""

import os
import sys
import logging

# Add project root to Python path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.base import AgentContext
from src.agents.standard_agent import StandardAgent
from src.modules.schema.mschema import MSchemaModule
from src.utils.config import load_yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.name = "MockModel"
        self.model_path = "test/mock"
    
    def format_prompt(self, question: str, schema: str) -> str:
        return f"Question: {question}\nSchema: {schema}\nSQL:"
    
    def generate(self, prompts: list) -> list:
        return ["SELECT * FROM test_table;" for _ in prompts]


class MockDataset:
    """Mock dataset for testing."""
    
    def get_schema(self, example: dict) -> str:
        return """CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    created_date DATE
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);"""
    
    def execute_sql(self, sql: str, example: dict):
        return True, [{"count": 1}]


def test_mschema_module():
    """Test MSchema module functionality."""
    logger.info("Testing MSchema module...")
    
    # Create mock objects
    mock_dataset = MockDataset()
    example = {"db_id": "test_db", "question": "How many customers are there?"}
    
    # Create context
    context = AgentContext(
        question=example["question"],
        example=example,
        dataset=mock_dataset
    )
    
    # Create and configure module
    module_config = {
        "name": "mschema",
        "enabled": True,
        "critical": False,
        "priority": 10,
        "module_config": {
            "include_samples": True,
            "include_descriptions": True,
            "max_sample_values": 3
        }
    }
    
    module = MSchemaModule(module_config)
    
    # Test schema processing
    original_schema = mock_dataset.get_schema(example)
    context.processed_schema = original_schema
    
    # Process with module
    module.process(context)
    
    # Check results
    assert context.processed_schema != original_schema
    assert "(DB_ID) test_db" in context.processed_schema
    assert "# Table customers" in context.processed_schema
    assert "# Table orders" in context.processed_schema
    
    # Check module output
    module_output = context.get_module_output("mschema")
    assert module_output is not None
    assert module_output["format"] == "m-schema"
    
    logger.info("✓ MSchema module test passed")
    return True


def test_standard_agent():
    """Test StandardAgent functionality."""
    logger.info("Testing StandardAgent...")
    
    # Create agent config
    agent_config = {
        "name": "TestAgent",
        "modules": {
            "mschema": {
                "enabled": True,
                "critical": False,
                "priority": 10,
                "module_config": {
                    "include_samples": True,
                    "include_descriptions": True,
                    "max_sample_values": 3
                }
            }
        }
    }
    
    # Create mock model and agent
    mock_model = MockModel()
    agent = StandardAgent(agent_config, mock_model)
    
    # Check module initialization
    assert len(agent.modules) == 1
    assert agent.modules[0].name == "mschema"
    assert agent.modules[0].is_enabled()
    
    # Test processing
    mock_dataset = MockDataset()
    example = {"db_id": "test_db", "question": "How many customers are there?"}
    
    context = AgentContext(
        question=example["question"],
        example=example,
        dataset=mock_dataset
    )
    
    # Process with agent
    result = agent.process(context)
    
    # Check results
    assert result == "SELECT * FROM test_table;"
    assert context.processed_schema != context.original_schema
    assert "(DB_ID) test_db" in context.processed_schema
    
    logger.info("✓ StandardAgent test passed")
    return True


def test_module_priority():
    """Test module priority ordering."""
    logger.info("Testing module priority...")
    
    # Create agent config with multiple modules (only mschema is implemented)
    agent_config = {
        "name": "TestAgent",
        "modules": {
            "mschema": {
                "enabled": True,
                "priority": 10,
                "module_config": {}
            }
        }
    }
    
    mock_model = MockModel()
    agent = StandardAgent(agent_config, mock_model)
    
    # Check modules are sorted by priority
    assert len(agent.modules) == 1
    assert agent.modules[0].priority == 10
    
    logger.info("✓ Module priority test passed")
    return True


def test_module_enable_disable():
    """Test module enable/disable functionality."""
    logger.info("Testing module enable/disable...")
    
    # Create agent config
    agent_config = {
        "name": "TestAgent",
        "modules": {
            "mschema": {
                "enabled": False,  # Start disabled
                "priority": 10,
                "module_config": {}
            }
        }
    }
    
    mock_model = MockModel()
    agent = StandardAgent(agent_config, mock_model)
    
    # Should have no modules since mschema is disabled
    assert len(agent.modules) == 0
    
    logger.info("✓ Module enable/disable test passed")
    return True


def main():
    """Run all tests."""
    logger.info("Starting agent architecture tests...")
    
    tests = [
        test_mschema_module,
        test_standard_agent,
        test_module_priority,
        test_module_enable_disable
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Agent architecture is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 