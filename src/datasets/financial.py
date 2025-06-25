import os
import json
import logging
from typing import List, Dict, Any, Tuple
from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class FinancialDataset(BaseDataset):
    """Implementation for the FINANCIAL text2sql dataset with manually organized data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.examples = None
        
    def download_and_setup(self):
        """Setup for manually organized dataset - just ensure directories exist."""
        self._ensure_directories()
        
        # Check if processed data exists
        processed_file = os.path.join(self.processed_dir, "financial_data.json")
        if not os.path.exists(processed_file):
            raise FileNotFoundError(
                f"Processed dataset file not found: {processed_file}\n"
                f"Please manually place your dataset file at this location."
            )
        
        logger.info("Dataset already organized manually - ready to use")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the manually organized FINANCIAL dataset."""
        if self.examples is not None:
            return self.examples
            
        # Ensure setup is complete
        self.download_and_setup()
        
        # Load processed data
        processed_file = os.path.join(self.processed_dir, "financial_data.json")
        
        logger.info("Loading FINANCIAL dataset from manually organized files...")
        with open(processed_file, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)
        
        # Apply subset if specified
        if self.subset_size and self.subset_size < len(all_examples):
            all_examples = all_examples[:self.subset_size]
            logger.info(f"Using subset of {self.subset_size} examples")
        
        # Add schema information to each example using base class method
        for example in all_examples:
            if "schema" not in example:
                example["schema"] = self.get_schema(example)
        
        self.examples = all_examples
        logger.info(f"Loaded {len(self.examples)} FINANCIAL examples")
        return self.examples
    
    # get_schema() and execute_sql() are inherited from BaseDataset
    # They automatically use the standardized databases/{db_id}/{db_id}.sqlite structure
    
    def cleanup(self):
        """Clean up any resources."""
        # Base class handles cleanup automatically
        pass
