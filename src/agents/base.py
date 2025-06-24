from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from src.models.base import BaseModel
from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class AgentContext:
    """Context object that flows through the agent pipeline."""
    
    def __init__(self, question: str, example: Dict[str, Any], dataset: BaseDataset):
        self.question = question
        self.example = example
        self.dataset = dataset
        
        # Pipeline state
        self.original_schema = ""
        self.processed_schema = ""
        self.relevant_tables = []
        self.relevant_columns = []
        self.metadata = {}
        self.sql_candidates = []
        self.execution_results = []
        self.final_sql = ""
        
        # Module outputs
        self.module_outputs = {}
        
    def set_module_output(self, module_name: str, output: Any):
        """Store output from a specific module."""
        self.module_outputs[module_name] = output
        
    def get_module_output(self, module_name: str) -> Any:
        """Retrieve output from a specific module."""
        return self.module_outputs.get(module_name)


class BaseAgent(ABC):
    """Base class for Text2SQL agents."""
    
    def __init__(self, config: Dict[str, Any], model: BaseModel):
        self.config = config
        self.model = model
        self.name = config.get("name", "BaseAgent")
        self.modules = []
        self.module_configs = config.get("modules", {})
        
        # Initialize modules based on configuration
        self._initialize_modules()
        
    @abstractmethod
    def _initialize_modules(self):
        """Initialize and configure modules for this agent."""
        pass
    
    def process(self, context: AgentContext) -> str:
        """Main processing pipeline."""
        logger.info(f"Processing question with {self.name}")
        
        # Initialize context with original schema
        context.original_schema = context.dataset.get_schema(context.example)
        context.processed_schema = context.original_schema
        
        # Execute module pipeline
        for module in self.modules:
            if module.is_enabled():
                logger.debug(f"Executing module: {module.name}")
                try:
                    module.process(context)
                except Exception as e:
                    logger.error(f"Module {module.name} failed: {e}")
                    if module.is_critical():
                        raise
        
        # Generate final SQL using the model
        if context.sql_candidates:
            # If we have candidates, use selection logic
            context.final_sql = self._select_best_candidate(context)
        else:
            # Generate directly using the model
            context.final_sql = self._generate_sql(context)
        
        return context.final_sql
    
    def _generate_sql(self, context: AgentContext) -> str:
        """Generate SQL using the base model."""
        prompt = self.model.format_prompt(context.question, context.processed_schema)
        predictions = self.model.generate([prompt])
        return predictions[0] if predictions else ""
    
    def _select_best_candidate(self, context: AgentContext) -> str:
        """Select the best SQL candidate from available options."""
        # Default implementation: return first candidate
        # This can be overridden by specific agents or modules
        return context.sql_candidates[0] if context.sql_candidates else ""
    
    def get_module_by_name(self, name: str) -> Optional['BaseModule']:
        """Get a specific module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def enable_module(self, name: str):
        """Enable a specific module."""
        module = self.get_module_by_name(name)
        if module:
            module.enable()
    
    def disable_module(self, name: str):
        """Disable a specific module."""
        module = self.get_module_by_name(name)
        if module:
            module.disable() 