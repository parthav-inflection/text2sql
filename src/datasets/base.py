import os
import json
import sqlite3
import logging
import zipfile
import tarfile
import tempfile
import shutil
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for all text2sql datasets with standardized structure."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.data_dir = config.get("data_dir", f"./data/{self.name.lower()}")
        self.subset_size = config.get("subset_size", None)
        
        # Standardized directory structure
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed") 
        self.databases_dir = os.path.join(self.data_dir, "databases")
        
        # Dataset URLs from config
        self.urls = config.get("urls", {})
        
    def _ensure_directories(self):
        """Create standardized directory structure."""
        for directory in [self.data_dir, self.raw_dir, self.processed_dir, self.databases_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def _download_file(self, url: str, target_path: str, description: str = "Downloading") -> str:
        """Download a file with progress bar."""
        logger.info(f"Downloading {description} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'wb') as f:
            downloaded = 0
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {target_path}")
        return target_path
        
    def _extract_archive(self, archive_path: str, target_dir: str) -> str:
        """Extract zip/tar archive to target directory."""
        logger.info(f"Extracting {archive_path} to {target_dir}")
        
        os.makedirs(target_dir, exist_ok=True)
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(target_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
            
        logger.info("Extraction complete")
        return target_dir
        
    def _find_file(self, directory: str, filename: str, recursive: bool = True) -> Optional[str]:
        """Find a file in directory structure."""
        if recursive:
            for root, dirs, files in os.walk(directory):
                if filename in files:
                    return os.path.join(root, filename)
        else:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                return filepath
        return None
        
    def _organize_databases(self, source_dir: str, db_pattern: str = "*.sqlite") -> None:
        """Organize database files into standardized structure."""
        import glob
        
        # Find all database files
        db_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.sqlite') or file.endswith('.db'):
                    db_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(db_files)} database files to organize")
        
        for db_file in db_files:
            # Extract db_id from filename (remove extension)
            db_name = os.path.basename(db_file)
            db_id = os.path.splitext(db_name)[0]
            
            # Create target directory: databases/{db_id}/
            target_dir = os.path.join(self.databases_dir, db_id)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy database file
            target_path = os.path.join(target_dir, db_name)
            if not os.path.exists(target_path):
                shutil.copy2(db_file, target_path)
                logger.info(f"Organized database: {db_id} -> {target_path}")
                
    def _get_database_path(self, db_id: str) -> str:
        """Standard database path resolution."""
        # Standard structure: databases/{db_id}/{db_id}.sqlite
        possible_extensions = ['.sqlite', '.db']
        
        for ext in possible_extensions:
            db_path = os.path.join(self.databases_dir, db_id, f"{db_id}{ext}")
            if os.path.exists(db_path):
                return db_path
                
        # Fallback: return most likely path for debugging
        return os.path.join(self.databases_dir, db_id, f"{db_id}.sqlite")
        
    def _extract_schema_from_db(self, db_path: str) -> str:
        """Standard schema extraction from SQLite database."""
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return ""
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get CREATE TABLE statements
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;")
            tables = cursor.fetchall()
            schema_parts = [sql for (sql,) in tables if sql]
            
            conn.close()
            return "\n\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"Error extracting schema from {db_path}: {e}")
            return ""
            
    def _execute_sql_on_db(self, sql: str, db_path: str) -> Tuple[bool, Any]:
        """Standard SQL execution on database."""
        if not os.path.exists(db_path):
            return False, f"Database not found: {db_path}"
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)
    
    # Abstract methods that datasets must implement
    @abstractmethod
    def download_and_setup(self):
        """Download and setup the dataset using the helper methods above."""
        pass
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and return the dataset examples."""
        pass
    
    def get_schema(self, example: Dict[str, Any]) -> str:
        """Get the database schema for an example. Default implementation."""
        # Try to get from example first (if cached)
        if "schema" in example:
            return example["schema"]
            
        # Extract from database
        db_id = example.get("db_id")
        if db_id:
            db_path = self._get_database_path(db_id)
            return self._extract_schema_from_db(db_path)
        
        return ""
    
    def execute_sql(self, sql: str, example: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute SQL query. Default implementation."""
        db_id = example.get("db_id")
        if not db_id:
            return False, "No db_id in example"
            
        db_path = self._get_database_path(db_id)
        return self._execute_sql_on_db(sql, db_path) 