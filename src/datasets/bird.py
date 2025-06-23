import os
import json
import sqlite3
import logging
import zipfile
import tempfile
from typing import List, Dict, Any, Tuple
import requests
from tqdm import tqdm
from datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class BirdDataset(BaseDataset):
    """BIRD Mini-Dev benchmark dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Official BIRD Mini-Dev dataset URLs
        self.data_urls = {
            "minidev_zip": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip",
            "google_drive": "https://drive.google.com/file/d/1UJyA6I6pTmmhYpwdn8iT9QKrcJqSQAcX/view?usp=sharing"
        }
        self.examples = None
        self.db_connections = {}
        self.extracted_data_path = os.path.join(self.data_dir, "mini_dev_data")
    
    def download_and_setup(self):
        """Download and setup the BIRD Mini-Dev dataset."""
        logger.info("Setting up BIRD Mini-Dev dataset...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Download and extract the dataset
        self._download_and_extract_dataset()
        
        logger.info("BIRD Mini-Dev dataset setup complete")
    
    def _download_and_extract_dataset(self):
        """Download and extract the complete Mini-Dev dataset."""
        # Check if dataset already exists
        json_path = os.path.join(self.extracted_data_path, "mini_dev_sqlite.json")
        if os.path.exists(json_path):
            logger.info("BIRD Mini-Dev dataset already exists, skipping download")
            return
        
        logger.info("Downloading BIRD Mini-Dev dataset...")
        
        # Download the dataset zip
        response = requests.get(self.data_urls["minidev_zip"], stream=True)
        response.raise_for_status()
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Save to temporary file with progress bar
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            downloaded = 0
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
            tmp_path = tmp_file.name
        
        # Extract the zip file
        logger.info("Extracting BIRD Mini-Dev dataset...")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            # Extract all files to data directory
            zip_ref.extractall(self.data_dir)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Check if extraction was successful
        if not os.path.exists(json_path):
            # Sometimes the extracted folder might have a different structure
            # Let's search for the JSON file
            for root, dirs, files in os.walk(self.data_dir):
                if "mini_dev_sqlite.json" in files:
                    found_path = os.path.join(root, "mini_dev_sqlite.json")
                    logger.info(f"Found dataset JSON at: {found_path}")
                    # If it's not in the expected location, copy it there
                    if found_path != json_path:
                        import shutil
                        os.makedirs(self.extracted_data_path, exist_ok=True)
                        shutil.copy2(found_path, json_path)
                        # Also copy the entire directory structure
                        parent_dir = os.path.dirname(found_path)
                        if os.path.basename(parent_dir) != "mini_dev_data":
                            # Copy all contents to our expected location
                            for item in os.listdir(parent_dir):
                                src = os.path.join(parent_dir, item)
                                dst = os.path.join(self.extracted_data_path, item)
                                if os.path.isdir(src):
                                    if os.path.exists(dst):
                                        shutil.rmtree(dst)
                                    shutil.copytree(src, dst)
                                else:
                                    shutil.copy2(src, dst)
                    break
        
        if os.path.exists(json_path):
            logger.info("Dataset extraction successful")
        else:
            raise FileNotFoundError("Could not find mini_dev_sqlite.json after extraction")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and return the BIRD Mini-Dev dataset examples."""
        if self.examples is not None:
            return self.examples
        
        json_path = os.path.join(self.extracted_data_path, "mini_dev_sqlite.json")
        
        if not os.path.exists(json_path):
            self.download_and_setup()
        
        logger.info("Loading BIRD Mini-Dev dataset...")
        with open(json_path, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)
        
        # Apply subset if specified
        if self.subset_size and self.subset_size < len(all_examples):
            all_examples = all_examples[:self.subset_size]
            logger.info(f"Using subset of {self.subset_size} examples")
        
        # Process examples to add database schema information
        processed_examples = []
        for example in all_examples:
            # Add schema information by reading from database
            schema = self._get_database_schema(example["db_id"])
            example["schema"] = schema
            processed_examples.append(example)
        
        self.examples = processed_examples
        logger.info(f"Loaded {len(self.examples)} BIRD Mini-Dev examples")
        return self.examples
    
    def _get_database_schema(self, db_id: str) -> str:
        """Get the database schema for a given database ID."""
        db_path = self._get_database_path(db_id)
        
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return ""
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_parts = []
            for (table_name,) in tables:
                # Get CREATE TABLE statement
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                create_statement = cursor.fetchone()
                if create_statement and create_statement[0]:
                    schema_parts.append(create_statement[0])
            
            conn.close()
            return "\n\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"Error reading schema for {db_id}: {e}")
            return ""
    
    def _get_database_path(self, db_id: str) -> str:
        """Get the full path to a database file."""
        # Try different possible locations for the database files
        possible_paths = [
            os.path.join(self.extracted_data_path, "dev_databases", db_id, f"{db_id}.sqlite"),
            os.path.join(self.data_dir, "dev_databases", db_id, f"{db_id}.sqlite"),
            os.path.join(self.data_dir, "mini_dev_data", "dev_databases", db_id, f"{db_id}.sqlite"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If none found, return the most likely path for debugging
        return possible_paths[0]
    
    def get_schema(self, example: Dict[str, Any]) -> str:
        """Get the database schema for an example."""
        return example.get("schema", "")
    
    def execute_sql(self, sql: str, example: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute SQL query and return (success, result)."""
        db_id = example["db_id"]
        db_path = self._get_database_path(db_id)
        
        if not os.path.exists(db_path):
            return False, f"Database {db_id} not found at {db_path}"
        
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