"""
Factory classes for creating agents, models, and modules with dependency injection.
"""

from typing import Dict, Any, List, Optional
import logging
import yaml
from pathlib import Path

from src.agents.base import BaseAgent, StandardModulePipeline
from src.agents.standard_agent import StandardAgent
from src.models.base import BaseModel
from src.models.vllm_model import VLLMModel
from src.modules.base import BaseModule
from src.modules.schema.mschema import MSchemaModule
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(config_path: str) -> BaseModel:
        """Create a model from configuration file."""
        config = load_config(config_path)
        model_type = config.get('model_type', 'vllm')
        
        if model_type == 'vllm':
            return VLLMModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ModuleFactory:
    """Factory for creating module instances."""
    
    @staticmethod
    def create_module(module_name: str, module_config: Dict[str, Any]) -> BaseModule:
        """Create a module by name and configuration."""
        # Ensure module name is in config
        module_config = dict(module_config)
        module_config["name"] = module_name
        
        # Registry of available modules
        module_registry = {
            'mschema': MSchemaModule,
            # Future modules can be added here:
            # 'schema_linking': SchemaLinkingModule,
            # 'candidate_generation': CandidateGenerationModule,
            # 'self_refinement': SelfRefinementModule,
            # 'candidate_selection': CandidateSelectionModule,
        }
        
        module_class = module_registry.get(module_name)
        if not module_class:
            raise ValueError(f"Unknown module type: {module_name}")
        
        return module_class(module_config)
    
    @staticmethod
    def create_modules_from_config(modules_config: Dict[str, Dict[str, Any]]) -> List[BaseModule]:
        """Create multiple modules from configuration dictionary."""
        modules = []
        
        for module_name, module_config in modules_config.items():
            if not module_config.get("enabled", True):
                logger.info(f"Skipping disabled module: {module_name}")
                continue
                
            try:
                module = ModuleFactory.create_module(module_name, module_config)
                modules.append(module)
                logger.info(f"Created module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to create module {module_name}: {e}")
                if module_config.get("critical", False):
                    raise
        
        return modules


class PipelineFactory:
    """Factory for creating module pipelines."""
    
    @staticmethod
    def create_pipeline(modules: List[BaseModule]) -> StandardModulePipeline:
        """Create a standard module pipeline."""
        return StandardModulePipeline(modules)
    
    @staticmethod
    def create_pipeline_from_config(modules_config: Dict[str, Dict[str, Any]]) -> StandardModulePipeline:
        """Create a pipeline from module configuration."""
        modules = ModuleFactory.create_modules_from_config(modules_config)
        return PipelineFactory.create_pipeline(modules)


class AgentFactory:
    """Factory for creating agent instances."""
    
    @staticmethod
    def create_agent(
        agent_config_path: str,
        model_config_path: str,
        agent_name: Optional[str] = None
    ) -> BaseAgent:
        """Create an agent from configuration files."""
        # Load configurations
        agent_config = load_config(agent_config_path)
        model = ModelFactory.create_model(model_config_path)
        
        # Create pipeline
        modules_config = agent_config.get('modules', {})
        pipeline = PipelineFactory.create_pipeline_from_config(modules_config)
        
        # Determine agent name
        name = agent_name or agent_config.get('name', 'StandardAgent')
        
        # Create agent (currently only StandardAgent is supported)
        agent_type = agent_config.get('type', 'standard')
        if agent_type == 'standard':
            return StandardAgent(name, model, pipeline, agent_config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def create_agent_from_components(
        name: str,
        model: BaseModel,
        modules: List[BaseModule],
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """Create an agent from individual components."""
        pipeline = PipelineFactory.create_pipeline(modules)
        return StandardAgent(name, model, pipeline, config)


class AgentBuilder:
    """Builder pattern for creating agents with fluent interface."""
    
    def __init__(self):
        self._name = "Agent"
        self._model = None
        self._modules = []
        self._config = {}
    
    def with_name(self, name: str) -> 'AgentBuilder':
        """Set agent name."""
        self._name = name
        return self
    
    def with_model(self, model: BaseModel) -> 'AgentBuilder':
        """Set the base model."""
        self._model = model
        return self
    
    def with_model_from_config(self, config_path: str) -> 'AgentBuilder':
        """Set the model from configuration file."""
        self._model = ModelFactory.create_model(config_path)
        return self
    
    def with_module(self, module: BaseModule) -> 'AgentBuilder':
        """Add a module."""
        self._modules.append(module)
        return self
    
    def with_modules(self, modules: List[BaseModule]) -> 'AgentBuilder':
        """Add multiple modules."""
        self._modules.extend(modules)
        return self
    
    def with_modules_from_config(self, modules_config: Dict[str, Dict[str, Any]]) -> 'AgentBuilder':
        """Add modules from configuration."""
        modules = ModuleFactory.create_modules_from_config(modules_config)
        self._modules.extend(modules)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """Set agent configuration."""
        self._config = config
        return self
    
    def build(self) -> BaseAgent:
        """Build the agent."""
        if not self._model:
            raise ValueError("Model is required to build agent")
        
        return AgentFactory.create_agent_from_components(
            self._name, self._model, self._modules, self._config
        ) 