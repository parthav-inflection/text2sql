#!/usr/bin/env python3
"""
Text2SQL Components Test Script

Tests all major components of the text2sql system without expensive operations:
- Configuration loading and validation
- Factory pattern instantiation
- Mock model creation and basic functionality
- Module system architecture
- Agent pipeline construction
- Dataset interface validation

Usage:
    python scripts/test_components.py [--verbose]
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test components
from src.utils.config import load_yaml, load_experiment_config, load_model_config, create_output_dir
from src.agents.base import BaseAgent, AgentContext, StandardModulePipeline
from src.agents.factory import AgentFactory, ModelFactory, ModuleFactory, PipelineFactory, AgentBuilder
from src.models.base import BaseModel
from src.datasets.base import BaseDataset
from src.modules.base import BaseModule


class MockModel(BaseModel):
    """Mock model for testing without actual model loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loaded = True
        
    def generate(self, prompts: List[str]) -> List[str]:
        """Mock generation returns simple SQL for testing."""
        return [f"SELECT * FROM test_table WHERE question LIKE '%{i}%';" for i in range(len(prompts))]
    
    def cleanup(self):
        """Mock cleanup."""
        self.loaded = False


class MockDataset(BaseDataset):
    """Mock dataset for testing without actual data loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loaded = True
        
    def download_and_setup(self):
        """Mock setup."""
        pass
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Return mock data for testing."""
        return [
            {
                "question": "How many customers are there?",
                "sql": "SELECT COUNT(*) FROM customers;",
                "db_id": "test_db"
            },
            {
                "question": "What is the average age?",
                "sql": "SELECT AVG(age) FROM users;",
                "db_id": "test_db"
            }
        ]
    
    def get_schema(self, example: Dict[str, Any]) -> str:
        """Return mock schema."""
        return """CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
);

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT,
    age INTEGER
);"""
    
    def execute_sql(self, sql: str, example: Dict[str, Any]):
        """Mock SQL execution."""
        return True, [{"count": 42}]


class MockModule(BaseModule):
    """Mock module for testing pipeline functionality."""
    
    def process(self, context: AgentContext):
        """Mock processing that modifies context."""
        context.set_module_output(self.name, f"processed_by_{self.name}")
        context.metadata[self.name] = True
        # Simulate some processing
        if "processed_schema" in context.__dict__:
            context.processed_schema += f"\n-- Processed by {self.name}"


class ComponentTester:
    """Main testing class for all text2sql components."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.temp_dir = None
        
    def setup_logging(self):
        """Setup logging for tests."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_all_tests(self) -> bool:
        """Run all component tests."""
        self.logger.info("Starting Text2SQL Component Tests")
        
        try:
            # Setup temporary directory for tests
            self.temp_dir = tempfile.mkdtemp(prefix="text2sql_test_")
            self.logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Run tests in order
            tests = [
                ("Config System", self.test_config_system),
                ("Model Interface", self.test_model_interface),
                ("Dataset Interface", self.test_dataset_interface),
                ("Module System", self.test_module_system),
                ("Pipeline System", self.test_pipeline_system),
                ("Factory Pattern", self.test_factory_pattern),
                ("Agent Architecture", self.test_agent_architecture),
                ("Integration Flow", self.test_integration_flow)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                self.logger.info(f"Running test: {test_name}")
                try:
                    result = test_func()
                    self.test_results[test_name] = result
                    if result:
                        self.logger.info(f"✅ {test_name} PASSED")
                        passed += 1
                    else:
                        self.logger.error(f"❌ {test_name} FAILED")
                except Exception as e:
                    self.logger.error(f"❌ {test_name} ERROR: {e}")
                    self.test_results[test_name] = False
            
            # Summary
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
            self.logger.info(f"{'='*50}")
            
            # Detailed results
            for test_name, result in self.test_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                self.logger.info(f"{status} - {test_name}")
            
            return passed == total
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            return False
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up temporary directory")
    
    def test_config_system(self) -> bool:
        """Test configuration loading and validation."""
        try:
            # Test YAML loading
            test_config = {"test": "value", "nested": {"key": "value"}}
            config_file = os.path.join(self.temp_dir, "test.yaml")
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f)
            
            loaded_config = load_yaml(config_file)
            assert loaded_config == test_config, "YAML loading failed"
            
            # Test output directory creation
            test_dir = create_output_dir(self.temp_dir, "test_experiment")
            assert os.path.exists(test_dir), "Output directory creation failed"
            
            # Test existing config files
            configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
            
            # Test experiment config loading
            exp_config_path = os.path.join(configs_dir, 'experiments', 'test_setup.yaml')
            if os.path.exists(exp_config_path):
                exp_config = load_experiment_config(exp_config_path)
                assert 'experiment_name' in exp_config, "Experiment config invalid"
            
            # Test model config loading
            model_config_path = os.path.join(configs_dir, 'models', 'qwen3.yaml')
            if os.path.exists(model_config_path):
                model_config = load_model_config(model_config_path)
                assert 'model_path' in model_config, "Model config invalid"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config system test failed: {e}")
            return False
    
    def test_model_interface(self) -> bool:
        """Test model interface and mock implementation."""
        try:
            # Test mock model creation
            config = {
                "name": "TestModel",
                "model_path": "test/path",
                "model_type": "mock"
            }
            
            model = MockModel(config)
            assert model.name == "TestModel", "Model name not set correctly"
            assert model.model_path == "test/path", "Model path not set correctly"
            
            # Test prompt formatting
            question = "How many users are there?"
            schema = "CREATE TABLE users (id INT, name TEXT);"
            prompt = model.format_prompt(question, schema)
            
            assert question in prompt, "Question not in formatted prompt"
            assert "users" in prompt, "Schema not properly included"
            
            # Test generation
            prompts = [prompt]
            results = model.generate(prompts)
            assert len(results) == 1, "Generate didn't return correct number of results"
            assert isinstance(results[0], str), "Generated result is not a string"
            
            # Test cleanup
            model.cleanup()
            assert not model.loaded, "Cleanup didn't work properly"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model interface test failed: {e}")
            return False
    
    def test_dataset_interface(self) -> bool:
        """Test dataset interface and mock implementation."""
        try:
            # Test mock dataset creation
            config = {
                "name": "TestDataset",
                "data_dir": "./test_data",
                "subset_size": 10
            }
            
            dataset = MockDataset(config)
            assert dataset.name == "TestDataset", "Dataset name not set correctly"
            assert dataset.data_dir == "./test_data", "Data dir not set correctly"
            
            # Test data loading
            data = dataset.load_data()
            assert isinstance(data, list), "Data should be a list"
            assert len(data) > 0, "Dataset should return some data"
            assert 'question' in data[0], "Data items should have questions"
            assert 'sql' in data[0], "Data items should have SQL"
            
            # Test schema retrieval
            example = data[0]
            schema = dataset.get_schema(example)
            assert isinstance(schema, str), "Schema should be a string"
            assert "CREATE TABLE" in schema.upper(), "Schema should contain table definitions"
            
            # Test SQL execution
            sql = "SELECT COUNT(*) FROM test_table;"
            success, result = dataset.execute_sql(sql, example)
            assert success, "SQL execution should succeed"
            assert result is not None, "SQL execution should return a result"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset interface test failed: {e}")
            return False
    
    def test_module_system(self) -> bool:
        """Test module system architecture."""
        try:
            # Test module creation
            config = {
                "name": "TestModule",
                "enabled": True,
                "critical": False,
                "priority": 10
            }
            
            module = MockModule(config)
            assert module.name == "TestModule", "Module name not set correctly"
            assert module.is_enabled(), "Module should be enabled"
            assert not module.is_critical(), "Module should not be critical"
            assert module.priority == 10, "Module priority not set correctly"
            
            # Test module state changes
            module.disable()
            assert not module.is_enabled(), "Module should be disabled"
            module.enable()
            assert module.is_enabled(), "Module should be enabled again"
            
            # Test module processing
            mock_dataset = MockDataset({"name": "test"})
            context = AgentContext(
                question="Test question",
                example={"test": "data"},
                dataset=mock_dataset
            )
            
            module.process(context)
            assert module.name in context.metadata, "Module should mark processing in metadata"
            assert context.get_module_output(module.name) is not None, "Module should set output"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Module system test failed: {e}")
            return False
    
    def test_pipeline_system(self) -> bool:
        """Test pipeline system functionality."""
        try:
            # Create test modules with different priorities
            modules = [
                MockModule({"name": "Module1", "priority": 30}),
                MockModule({"name": "Module2", "priority": 10}),
                MockModule({"name": "Module3", "priority": 20})
            ]
            
            # Test pipeline creation
            pipeline = StandardModulePipeline(modules)
            assert len(pipeline.modules) == 3, "Pipeline should have 3 modules"
            
            # Test module ordering by priority
            assert pipeline.modules[0].name == "Module2", "First module should be Module2 (priority 10)"
            assert pipeline.modules[1].name == "Module3", "Second module should be Module3 (priority 20)"
            assert pipeline.modules[2].name == "Module1", "Third module should be Module1 (priority 30)"
            
            # Test pipeline execution
            mock_dataset = MockDataset({"name": "test"})
            context = AgentContext(
                question="Test question",
                example={"test": "data"},
                dataset=mock_dataset
            )
            
            pipeline.execute(context)
            
            # Check that all modules processed the context
            for module in modules:
                assert module.name in context.metadata, f"Module {module.name} should have processed context"
            
            # Test module retrieval
            module2 = pipeline.get_module_by_name("Module2")
            assert module2 is not None, "Should be able to retrieve module by name"
            assert module2.name == "Module2", "Retrieved module should be correct"
            
            # Test module enable/disable
            pipeline.disable_module("Module2")
            assert not pipeline.get_module_by_name("Module2").is_enabled(), "Module should be disabled"
            
            pipeline.enable_module("Module2")
            assert pipeline.get_module_by_name("Module2").is_enabled(), "Module should be enabled"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline system test failed: {e}")
            return False
    
    def test_factory_pattern(self) -> bool:
        """Test factory pattern implementation."""
        try:
            # Test module factory
            module_config = {
                "name": "TestModule",
                "enabled": True,
                "priority": 10
            }
            
            # We can't test real modules without dependencies, but we can test the structure
            modules_config = {
                "test_module": module_config
            }
            
            # Test pipeline factory (create modules first, then pipeline)
            test_modules = [MockModule({"name": f"Module{i}", "priority": i*10}) for i in range(3)]
            pipeline = PipelineFactory.create_pipeline(test_modules)
            
            # Debug: Print actual type
            self.logger.debug(f"Pipeline type: {type(pipeline)}")
            self.logger.debug(f"StandardModulePipeline type: {StandardModulePipeline}")
            self.logger.debug(f"Is instance check: {isinstance(pipeline, StandardModulePipeline)}")
            
            assert isinstance(pipeline, StandardModulePipeline), f"Should create StandardModulePipeline, got {type(pipeline)}"
            assert len(pipeline.modules) == 3, "Pipeline should have correct number of modules"
            
            # Test agent builder pattern
            model = MockModel({"name": "TestModel", "model_path": "test"})
            builder = AgentBuilder()
            
            agent = (builder
                    .with_name("TestAgent")
                    .with_model(model)
                    .with_modules(test_modules)
                    .with_config({"test": "config"})
                    .build())
            
            assert agent.name == "TestAgent", "Agent name should be set correctly"
            assert agent.model == model, "Agent should use provided model"
            assert len(agent.pipeline.modules) == 3, "Agent should have correct modules"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Factory pattern test failed: {e}")
            return False
    
    def test_agent_architecture(self) -> bool:
        """Test agent architecture and processing."""
        try:
            # Create test components
            model = MockModel({"name": "TestModel", "model_path": "test"})
            modules = [MockModule({"name": f"Module{i}", "priority": i*10}) for i in range(2)]
            pipeline = StandardModulePipeline(modules)
            
            # Create agent using builder
            agent = (AgentBuilder()
                    .with_name("TestAgent")
                    .with_model(model)
                    .with_modules(modules)
                    .build())
            
            # Debug: Print actual type
            self.logger.debug(f"Agent type: {type(agent)}")
            self.logger.debug(f"BaseAgent type: {BaseAgent}")
            self.logger.debug(f"Is instance check: {isinstance(agent, BaseAgent)}")
            
            assert isinstance(agent, BaseAgent), f"Should create BaseAgent instance, got {type(agent)}"
            assert agent.name == "TestAgent", "Agent name should be correct"
            
            # Test agent processing
            mock_dataset = MockDataset({"name": "test"})
            context = AgentContext(
                question="How many users are there?",
                example={"db_id": "test_db"},
                dataset=mock_dataset
            )
            
            # Process with agent
            result_sql = agent.process(context)
            
            assert isinstance(result_sql, str), "Agent should return SQL string"
            assert len(result_sql) > 0, "Agent should return non-empty SQL"
            assert context.original_schema != "", "Context should have original schema"
            assert context.processed_schema != "", "Context should have processed schema"
            
            # Check that modules were executed
            for module in modules:
                assert module.name in context.metadata, f"Module {module.name} should have been executed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Agent architecture test failed: {e}")
            return False
    
    def test_integration_flow(self) -> bool:
        """Test complete integration flow without heavy dependencies."""
        try:
            # Create a complete but lightweight pipeline
            model = MockModel({
                "name": "IntegrationTestModel",
                "model_path": "test/path"
            })
            
            dataset = MockDataset({
                "name": "IntegrationTestDataset",
                "data_dir": "./test_data"
            })
            
            # Create modules with different functionalities
            modules = [
                MockModule({
                    "name": "SchemaProcessor",
                    "priority": 10,
                    "enabled": True,
                    "critical": False
                }),
                MockModule({
                    "name": "CandidateGenerator", 
                    "priority": 20,
                    "enabled": True,
                    "critical": False
                })
            ]
            
            # Build agent
            agent = (AgentBuilder()
                    .with_name("IntegrationTestAgent")
                    .with_model(model)
                    .with_modules(modules)
                    .with_config({"integration_test": True})
                    .build())
            
            # Load test data
            test_data = dataset.load_data()
            
            # Process multiple examples
            results = []
            for example in test_data[:2]:  # Test with first 2 examples
                context = AgentContext(
                    question=example["question"],
                    example=example,
                    dataset=dataset
                )
                
                sql_result = agent.process(context)
                results.append({
                    "question": example["question"],
                    "predicted_sql": sql_result,
                    "context_metadata": context.metadata.copy()
                })
            
            # Validate results
            assert len(results) == 2, "Should process 2 examples"
            
            for result in results:
                assert "predicted_sql" in result, "Should have predicted SQL"
                assert len(result["predicted_sql"]) > 0, "SQL should not be empty"
                assert "SchemaProcessor" in result["context_metadata"], "Schema processor should run"
                assert "CandidateGenerator" in result["context_metadata"], "Candidate generator should run"
            
            # Test model cleanup
            model.cleanup()
            
            self.logger.info(f"Integration test processed {len(results)} examples successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Integration flow test failed: {e}")
            return False


def main():
    """Main entry point for component testing."""
    parser = argparse.ArgumentParser(description="Test Text2SQL Components")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = ComponentTester(verbose=args.verbose)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()