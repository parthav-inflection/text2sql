from typing import Dict, Any, Optional
import logging

from src.agents.base import BaseAgent, StandardModulePipeline
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class StandardAgent(BaseAgent):
    """Standard Text2SQL agent with configurable modules."""
    
    def __init__(
        self, 
        name: str,
        model: BaseModel, 
        pipeline: StandardModulePipeline,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize standard agent with dependency injection."""
        super().__init__(name, model, pipeline, config)
        
    def get_module_by_name(self, name: str):
        """Get a specific module by name from the pipeline."""
        return self.pipeline.get_module_by_name(name)
    
    def enable_module(self, name: str):
        """Enable a specific module in the pipeline."""
        self.pipeline.enable_module(name)
    
    def disable_module(self, name: str):
        """Disable a specific module in the pipeline."""
        self.pipeline.disable_module(name) 