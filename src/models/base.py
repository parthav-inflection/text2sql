from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModel(ABC):
    """Abstract base class for all text2sql models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.model_path = config["model_path"]
        
    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate SQL queries for given prompts."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass
    
    def format_prompt(self, question: str, schema: str) -> str:
        """Format the input prompt for text2sql generation."""
        return f"""You are a SQL expert. Given the database schema and question, generate a SQL query to answer the question.

                Database Schema:
                {schema}

                Question: {question}

                Instructions:
                - Return only the SQL query, no explanations or additional text
                - Use proper SQL syntax for SQLite
                - Do not include markdown formatting or code blocks
                - End the query with a semicolon

                SQL Query:""" 