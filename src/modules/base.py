from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.agents.base import AgentContext

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """Base class for all agent modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.enabled = config.get("enabled", True)
        self.critical = config.get("critical", False)  # If True, failure stops pipeline
        self.priority = config.get("priority", 50)  # Lower numbers execute first
        
        # Module-specific configuration
        self.module_config = config.get("module_config", {})
        
    @abstractmethod
    def process(self, context: 'AgentContext'):
        """Process the context and modify it in-place."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if module is enabled."""
        return self.enabled
    
    def is_critical(self) -> bool:
        """Check if module is critical (failure stops pipeline)."""
        return self.critical
    
    def enable(self):
        """Enable this module."""
        self.enabled = True
        
    def disable(self):
        """Disable this module."""
        self.enabled = False
    
    def validate_config(self):
        """Validate module configuration. Override in subclasses."""
        pass
    
    def __lt__(self, other):
        """For sorting modules by priority."""
        return self.priority < other.priority


class SchemaProcessingModule(BaseModule):
    """Base class for modules that process database schemas."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    @abstractmethod
    def process_schema(self, schema: str, context: 'AgentContext') -> str:
        """Process the schema and return the modified version."""
        pass
    
    def process(self, context: 'AgentContext'):
        """Standard processing: modify the processed_schema."""
        context.processed_schema = self.process_schema(context.processed_schema, context)


class CandidateGenerationModule(BaseModule):
    """Base class for modules that generate SQL candidates."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    @abstractmethod
    def generate_candidates(self, context: 'AgentContext') -> List[str]:
        """Generate SQL candidates for the given context."""
        pass
    
    def process(self, context: 'AgentContext'):
        """Standard processing: add candidates to context."""
        candidates = self.generate_candidates(context)
        context.sql_candidates.extend(candidates)


class SelectionModule(BaseModule):
    """Base class for modules that select the best SQL from candidates."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    @abstractmethod
    def select_best(self, candidates: List[str], context: 'AgentContext') -> str:
        """Select the best SQL candidate."""
        pass
    
    def process(self, context: 'AgentContext'):
        """Standard processing: select best candidate."""
        if context.sql_candidates:
            best_sql = self.select_best(context.sql_candidates, context)
            # Replace candidates with just the best one
            context.sql_candidates = [best_sql]


class RefinementModule(BaseModule):
    """Base class for modules that refine/correct SQL queries."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    @abstractmethod
    def refine_sql(self, sql: str, context: 'AgentContext') -> str:
        """Refine/correct the SQL query."""
        pass 