import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Tuple
import requests
from tqdm import tqdm
from .base import BaseDataset

logger = logging.getLogger(__name__)


class BirdDataset(BaseDataset):
    """Bird benchmark dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_url = "https://bird-bench.github.io/data/bird.zip"
        self.examples = None
        self.db_connections = {}
    
    def download_and_setup(self):
        """Download and setup the Bird dataset."""
        logger.info("Setting up Bird dataset...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # For now, we'll use a simplified approach - create sample data
        # In production, you'd download and extract the actual Bird dataset
        self._create_sample_data()
        
        logger.info("Bird dataset setup complete")
    
    def _create_sample_data(self):
        """Create sample Bird data for testing."""
        # This creates a minimal sample for testing
        # Replace with actual Bird dataset loading in production
        sample_data = [
            {
                "question": "What is the average age of all employees?",
                "db_id": "company",
                "query": "SELECT AVG(age) FROM employees",
                "schema": "CREATE TABLE employees (id INTEGER, name TEXT, age INTEGER, department TEXT);",
                "difficulty": "easy"
            },
            {
                "question": "How many departments have more than 5 employees?",
                "db_id": "company", 
                "query": "SELECT COUNT(DISTINCT department) FROM employees GROUP BY department HAVING COUNT(*) > 5",
                "schema": "CREATE TABLE employees (id INTEGER, name TEXT, age INTEGER, department TEXT);",
                "difficulty": "medium"
            }
        ]
        
        # Create sample database
        db_path = os.path.join(self.data_dir, "company.db")
        conn = sqlite3.connect(db_path)
        
        # Create and populate sample table
        conn.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER, name TEXT, age INTEGER, department TEXT)")
        sample_employees = [
            (1, "Alice", 30, "Engineering"),
            (2, "Bob", 25, "Marketing"), 
            (3, "Charlie", 35, "Engineering"),
            (4, "Diana", 28, "Sales"),
            (5, "Eve", 32, "Engineering"),
            (6, "Frank", 29, "Marketing")
        ]
        conn.executemany("INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?)", sample_employees)
        conn.commit()
        conn.close()
        
        # Save sample data
        data_file = os.path.join(self.data_dir, "bird_dev.json")
        with open(data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Created sample Bird data with {len(sample_data)} examples")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and return the Bird dataset examples."""
        if self.examples is not None:
            return self.examples
        
        data_file = os.path.join(self.data_dir, "bird_dev.json")
        
        if not os.path.exists(data_file):
            self.download_and_setup()
        
        with open(data_file, 'r') as f:
            all_examples = json.load(f)
        
        # Apply subset if specified
        if self.subset_size:
            all_examples = all_examples[:self.subset_size]
        
        self.examples = all_examples
        logger.info(f"Loaded {len(self.examples)} Bird examples")
        return self.examples
    
    def get_schema(self, example: Dict[str, Any]) -> str:
        """Get the database schema for an example."""
        return example.get("schema", "")
    
    def execute_sql(self, sql: str, example: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute SQL query and return (success, result)."""
        db_id = example["db_id"]
        db_path = os.path.join(self.data_dir, f"{db_id}.db")
        
        if not os.path.exists(db_path):
            return False, f"Database {db_id} not found"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)
    
    def cleanup(self):
        """Clean up database connections."""
        for conn in self.db_connections.values():
            conn.close()
        self.db_connections.clear() 