import os
import json
import logging
from typing import List, Dict, Any, Tuple
from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class BirdDataset(BaseDataset):
    """BIRD Mini-Dev benchmark dataset implementation using standardized structure."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.examples = None
        
        # BIRD Mini-Dev dataset URLs
        if not self.urls:
            self.urls = {
                "main": "https://bird-bench.github.io/",
                "data": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip",
                "google_drive": "https://drive.google.com/file/d/1UJyA6I6pTmmhYpwdn8iT9QKrcJqSQAcX/view?usp=sharing"
            }
    
    def download_and_setup(self):
        """Download and setup the BIRD Mini-Dev dataset using standardized utilities."""
        self._ensure_directories()
        
        # Check if already processed
        processed_file = os.path.join(self.processed_dir, "bird_data.json")
        if os.path.exists(processed_file):
            logger.info("BIRD Mini-Dev dataset already processed")
            return
        
        # Check if we have raw data already
        raw_extracted = os.path.join(self.raw_dir, "extracted", "mini_dev_data")
        if not os.path.exists(raw_extracted):
            # Download the dataset
            archive_path = os.path.join(self.raw_dir, "minidev.zip")
            if not os.path.exists(archive_path):
                logger.info("Downloading BIRD Mini-Dev dataset...")
                self._download_file(
                    self.urls["data"], 
                    archive_path, 
                    "Downloading BIRD Mini-Dev"
                )
            
            # Extract the dataset
            extract_dir = os.path.join(self.raw_dir, "extracted")
            self._extract_archive(archive_path, extract_dir)
            
            # The archive might extract to different structures, let's find the data
            mini_dev_json = self._find_file(extract_dir, "mini_dev_sqlite.json")
            if not mini_dev_json:
                raise FileNotFoundError("Could not find mini_dev_sqlite.json in extracted data")
            
            # Copy to our expected location if needed
            expected_location = os.path.join(raw_extracted, "mini_dev_sqlite.json")
            if mini_dev_json != expected_location:
                import shutil
                source_dir = os.path.dirname(mini_dev_json)
                os.makedirs(raw_extracted, exist_ok=True)
                
                # Copy all files from source directory
                for item in os.listdir(source_dir):
                    src = os.path.join(source_dir, item)
                    dst = os.path.join(raw_extracted, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
        
        # Process the main dataset file
        raw_json_path = os.path.join(raw_extracted, "mini_dev_sqlite.json")
        if os.path.exists(raw_json_path):
            self._process_dataset_file(raw_json_path, processed_file)
        
        # Organize databases using the standard utility
        db_source_dir = os.path.join(raw_extracted, "dev_databases")
        if os.path.exists(db_source_dir):
            self._organize_databases(db_source_dir)
        
        logger.info("BIRD Mini-Dev dataset setup complete")
    
    def _process_dataset_file(self, source_file: str, target_file: str):
        """Process the BIRD dataset file into standardized format."""
        with open(source_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # BIRD format is already close to standard, just ensure consistency
        processed_data = []
        for item in raw_data:
            processed_item = {
                "question": item.get("question", ""),
                "db_id": item.get("db_id", ""),
                "SQL": item.get("SQL", ""),
                "difficulty": item.get("difficulty", ""),
                # Keep any other fields as-is
                **{k: v for k, v in item.items() if k not in ["question", "db_id", "SQL", "difficulty"]}
            }
            processed_data.append(processed_item)
        
        # Save processed data
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed {len(processed_data)} BIRD examples")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the BIRD Mini-Dev dataset using standardized pattern."""
        if self.examples is not None:
            return self.examples
        
        # Ensure dataset is downloaded and processed
        self.download_and_setup()
        
        # Load processed data
        processed_file = os.path.join(self.processed_dir, "bird_data.json")
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed BIRD dataset file not found: {processed_file}")
        
        logger.info("Loading BIRD Mini-Dev dataset...")
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
        logger.info(f"Loaded {len(self.examples)} BIRD Mini-Dev examples")
        return self.examples
    
    # get_schema() and execute_sql() are inherited from BaseDataset
    # They automatically use the standardized databases/{db_id}/{db_id}.sqlite structure
    
    def cleanup(self):
        """Clean up any resources."""
        # Base class handles cleanup automatically
        pass 