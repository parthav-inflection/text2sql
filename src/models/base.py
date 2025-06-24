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
        # Convert full CREATE TABLE statements to simplified schema
        simplified_schema = self._simplify_schema(schema)
        
        return f"""Write a SQL query to answer the question. Output ONLY the SQL query, nothing else.

Database Schema:
{simplified_schema}

Question: {question}

Formatting examples:
Question: How many customers are there?
SELECT COUNT(*) FROM customers;

Question: What is the average age of students?
SELECT AVG(age) FROM students;

Question: {question}
"""
    
    def _simplify_schema(self, schema: str) -> str:
        """Simplify schema from CREATE TABLE statements to table(columns) format."""
        if not schema:
            return ""
        
        import re
        simplified_tables = []
        
        # Extract table information from CREATE TABLE statements
        create_statements = schema.split('\n\n')
        
        for statement in create_statements:
            if 'CREATE TABLE' in statement.upper():
                # Extract table name
                table_match = re.search(r'CREATE TABLE\s+(\w+)', statement, re.IGNORECASE)
                if not table_match:
                    continue
                
                table_name = table_match.group(1)
                
                # Extract column definitions (simplified)
                columns = []
                lines = statement.split('\n')
                for line in lines[1:]:  # Skip CREATE TABLE line
                    line = line.strip()
                    if line and not line.startswith(')') and not line.startswith('CREATE'):
                        # Extract column name and basic type
                        col_match = re.match(r'(\w+)\s+(\w+)', line)
                        if col_match:
                            col_name = col_match.group(1)
                            col_type = col_match.group(2).upper()
                            # Simplify type names
                            if col_type in ['INTEGER', 'INT']:
                                col_type = 'INT'
                            elif col_type in ['VARCHAR', 'TEXT', 'CHAR']:
                                col_type = 'TEXT'
                            elif col_type in ['REAL', 'FLOAT', 'DOUBLE']:
                                col_type = 'REAL'
                            columns.append(f"{col_name} {col_type}")
                
                if columns:
                    simplified_tables.append(f"{table_name}({', '.join(columns)})")
        
        return '\n'.join(simplified_tables) if simplified_tables else schema 