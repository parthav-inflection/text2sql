from typing import Dict, Any
import logging

from src.agents.base import BaseAgent
from src.models.base import BaseModel
from src.modules.schema.mschema import MSchemaModule

logger = logging.getLogger(__name__)


class StandardAgent(BaseAgent):
    """Standard Text2SQL agent with configurable modules."""
    
    def __init__(self, config: Dict[str, Any], model: BaseModel):
        super().__init__(config, model)
        
    def _initialize_modules(self):
        """Initialize modules based on configuration."""
        self.modules = []
        
        # Initialize modules in priority order
        for module_name, module_config in self.module_configs.items():
            if not module_config.get("enabled", True):
                continue
                
            try:
                module = self._create_module(module_name, module_config)
                if module:
                    self.modules.append(module)
                    logger.info(f"Initialized module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to initialize module {module_name}: {e}")
                if module_config.get("critical", False):
                    raise
        
        # Sort modules by priority
        self.modules.sort()
        logger.info(f"Initialized {len(self.modules)} modules")
    
    def _create_module(self, module_name: str, module_config: Dict[str, Any]):
        """Factory method to create modules by name."""
        # Add module name to config
        module_config = dict(module_config)
        module_config["name"] = module_name
        
        if module_name == "mschema":
            return MSchemaModule(module_config)
        # Future modules will be added here:
        # elif module_name == "schema_linking":
        #     return SchemaLinkingModule(module_config)
        # elif module_name == "candidate_generation":
        #     return CandidateGenerationModule(module_config)
        # elif module_name == "self_refinement":
        #     return SelfRefinementModule(module_config)
        else:
            logger.warning(f"Unknown module type: {module_name}")
            return None 