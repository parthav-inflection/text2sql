from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseDataset(ABC):
    """Abstract base class for all text2sql datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.data_dir = config.get("data_dir", "./data")
        self.subset_size = config.get("subset_size", None)
        
    @abstractmethod
    def download_and_setup(self):
        """Download and setup the dataset."""
        pass
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and return the dataset examples."""
        pass
    
    @abstractmethod
    def get_schema(self, example: Dict[str, Any]) -> str:
        """Get the database schema for an example."""
        pass
    
    @abstractmethod
    def execute_sql(self, sql: str, example: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute SQL query and return (success, result)."""
        pass 