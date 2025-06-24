from typing import Dict, Any, List, TYPE_CHECKING
import logging
import re

from src.modules.base import SchemaProcessingModule

if TYPE_CHECKING:
    from src.agents.base import AgentContext

logger = logging.getLogger(__name__)


class MSchemaModule(SchemaProcessingModule):
    """
    M-Schema Representation Module
    
    Based on XiYan-SQL: A Multi-Generator Ensemble Framework
    Converts database schema to a compact, semi-structured format with:
    - Special tokens like (DB_ID) and # Table
    - Column tuples with name, type, description, PK status, sample values
    - Explicit foreign key relationships
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.include_samples = config.get("module_config", {}).get("include_samples", True)
        self.include_descriptions = config.get("module_config", {}).get("include_descriptions", True)
        self.max_sample_values = config.get("module_config", {}).get("max_sample_values", 3)
        
    def process_schema(self, schema: str, context: 'AgentContext') -> str:
        """Convert schema to M-Schema format."""
        if not schema:
            return schema
            
        try:
            # Get database ID from context if available
            db_id = context.example.get("db_id", "DATABASE")
            
            # Parse the schema and convert to M-Schema format
            mschema = self._convert_to_mschema(schema, db_id, context)
            
            # Store the formatted schema in module outputs
            context.set_module_output(self.name, {
                "original_schema": schema,
                "mschema": mschema,
                "format": "m-schema"
            })
            
            logger.debug(f"Converted schema to M-Schema format for DB: {db_id}")
            return mschema
            
        except Exception as e:
            logger.error(f"Failed to convert schema to M-Schema format: {e}")
            # Return original schema if conversion fails
            return schema
    
    def _convert_to_mschema(self, schema: str, db_id: str, context: 'AgentContext') -> str:
        """Convert CREATE TABLE statements to M-Schema format."""
        lines = []
        
        # Add database identifier
        lines.append(f"(DB_ID) {db_id}")
        lines.append("")
        
        # Parse CREATE TABLE statements
        tables = self._parse_create_tables(schema)
        
        for table_info in tables:
            table_name = table_info["name"]
            columns = table_info["columns"]
            foreign_keys = table_info.get("foreign_keys", [])
            
            # Add table header with column count
            lines.append(f"# Table {table_name} ({len(columns)} columns)")
            
            # Add columns in tuple format
            for col in columns:
                col_tuple = self._format_column_tuple(col, context)
                lines.append(f"  {col_tuple}")
            
            # Add foreign keys if present
            if foreign_keys:
                lines.append("  # Foreign Keys")
                for fk in foreign_keys:
                    lines.append(f"    {fk}")
            
            lines.append("")  # Empty line between tables
        
        return "\n".join(lines)
    
    def _parse_create_tables(self, schema: str) -> List[Dict[str, Any]]:
        """Parse CREATE TABLE statements into structured format."""
        tables = []
        
        # More robust parsing: find CREATE TABLE patterns
        # Handle cases where tables might not be separated by double newlines
        create_pattern = r'CREATE TABLE[^;]*?;'
        create_statements = re.findall(create_pattern, schema, re.IGNORECASE | re.DOTALL)
        
        # Fallback: split by double newlines if no semicolons found
        if not create_statements:
            create_statements = re.split(r'\n\s*\n', schema.strip())
            create_statements = [stmt for stmt in create_statements if 'CREATE TABLE' in stmt.upper()]
        
        for statement in create_statements:
            table_info = self._parse_single_table(statement)
            if table_info:
                tables.append(table_info)
        
        return tables
    
    def _parse_single_table(self, statement: str) -> Dict[str, Any]:
        """Parse a single CREATE TABLE statement."""
        # Extract table name
        table_match = re.search(r'CREATE TABLE\s+(\w+)', statement, re.IGNORECASE)
        if not table_match:
            return None
            
        table_name = table_match.group(1)
        
        # Parse columns
        columns = []
        foreign_keys = []
        
        lines = statement.split('\n')
        for line in lines[1:]:  # Skip CREATE TABLE line
            line = line.strip()
            if not line or line.startswith(')') or line.startswith('CREATE'):
                continue
                
            # Check for FOREIGN KEY constraint
            if 'FOREIGN KEY' in line.upper():
                fk_info = self._parse_foreign_key(line)
                if fk_info:
                    foreign_keys.append(fk_info)
                continue
            
            # Parse column definition
            col_info = self._parse_column_definition(line)
            if col_info:
                columns.append(col_info)
        
        return {
            "name": table_name,
            "columns": columns,
            "foreign_keys": foreign_keys
        }
    
    def _parse_column_definition(self, line: str) -> Dict[str, Any]:
        """Parse a column definition line."""
        # Remove trailing comma and extra whitespace
        line = line.rstrip(',').strip()
        
        # Skip constraint-only lines (but not column definitions with constraints)
        line_upper = line.upper().strip()
        if (line_upper.startswith('PRIMARY KEY') or 
            line_upper.startswith('FOREIGN KEY') or 
            line_upper.startswith('UNIQUE (') or
            line_upper.startswith('INDEX') or
            line_upper.startswith('CONSTRAINT')):
            return None
        
        # Enhanced column pattern: handle more type variations
        col_match = re.match(r'(\w+)\s+(\w+(?:\(\d+(?:,\s*\d+)?\))?(?:\s+UNSIGNED)?)', line, re.IGNORECASE)
        if not col_match:
            return None
            
        col_name = col_match.group(1)
        col_type = col_match.group(2)
        
        # Check for PRIMARY KEY (both inline and constraint)
        is_primary = 'PRIMARY KEY' in line.upper()
        
        # Check for NOT NULL
        not_null = 'NOT NULL' in line.upper()
        
        # Check for UNIQUE
        is_unique = 'UNIQUE' in line.upper()
        
        # Check for DEFAULT value
        default_match = re.search(r'DEFAULT\s+([^\s,]+)', line, re.IGNORECASE)
        default_value = default_match.group(1) if default_match else None
        
        # Normalize type
        normalized_type = self._normalize_type(col_type)
        
        return {
            "name": col_name,
            "type": normalized_type,
            "original_type": col_type,
            "primary_key": is_primary,
            "not_null": not_null,
            "unique": is_unique,
            "default_value": default_value,
            "line": line
        }
    
    def _parse_foreign_key(self, line: str) -> str:
        """Parse a foreign key constraint."""
        line = line.strip().rstrip(',')
        
        # Extract FK relationship: FOREIGN KEY (col) REFERENCES table(col)
        fk_match = re.search(r'FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)', line, re.IGNORECASE)
        if fk_match:
            local_col = fk_match.group(1).strip()
            ref_table = fk_match.group(2).strip()
            ref_col = fk_match.group(3).strip()
            return f"FK({local_col}) -> {ref_table}({ref_col})"
        
        # Fallback: return cleaned line
        return line
    
    def _normalize_type(self, type_str: str) -> str:
        """Normalize SQL types to standard forms."""
        type_str = type_str.upper()
        
        if 'INT' in type_str:
            return 'INTEGER'
        elif any(t in type_str for t in ['VARCHAR', 'CHAR', 'TEXT']):
            return 'TEXT'
        elif any(t in type_str for t in ['REAL', 'FLOAT', 'DOUBLE']):
            return 'REAL'
        elif 'DATE' in type_str:
            return 'DATE'
        else:
            return type_str
    
    def _format_column_tuple(self, col: Dict[str, Any], context: 'AgentContext') -> str:
        """Format column information as M-Schema tuple."""
        col_name = col["name"]
        col_type = col["type"]
        
        # M-Schema tuple format: (column_name, data_type, "description", primary_key_flag, [sample_values])
        components = []
        
        # 1. Column name (always present)
        components.append(col_name)
        
        # 2. Data type (always present)  
        components.append(col_type)
        
        # 3. Description (if enabled)
        if self.include_descriptions:
            description = self._get_column_description(col, context)
            components.append(f'"{description}"')
        else:
            components.append('""')  # Empty description placeholder
        
        # 4. Primary key marker (always include, empty string if not PK)
        pk_marker = "PK" if col.get("primary_key", False) else ""
        components.append(pk_marker)
        
        # 5. Sample values (if enabled and available)
        if self.include_samples:
            samples = self._get_sample_values(col, context)
            if samples:
                # Format samples as quoted strings
                quoted_samples = [f'"{sample}"' for sample in samples]
                sample_str = f"[{', '.join(quoted_samples)}]"
                components.append(sample_str)
        
        return f"({', '.join(components)})"
    
    def _get_column_description(self, col: Dict[str, Any], context: 'AgentContext') -> str:
        """Generate or retrieve column description."""
        col_name = col["name"].lower()
        col_type = col["type"]
        
        # Enhanced heuristic descriptions
        if col_name.endswith('_id') or col_name == 'id':
            return "unique identifier"
        elif 'name' in col_name:
            return "name field"
        elif 'email' in col_name:
            return "email address"
        elif 'phone' in col_name:
            return "phone number"
        elif 'address' in col_name:
            return "address information"
        elif 'date' in col_name or 'time' in col_name or col_type == 'DATE':
            return "date/time value"
        elif 'price' in col_name or 'cost' in col_name or 'amount' in col_name:
            return "monetary value"
        elif 'count' in col_name or 'num' in col_name or 'quantity' in col_name:
            return "numeric count"
        elif 'status' in col_name or 'state' in col_name:
            return "status indicator"
        elif col_type in ['TEXT', 'VARCHAR']:
            return "text field"
        elif col_type in ['INTEGER', 'REAL']:
            return "numeric value"
        else:
            return f"{col_name} field"
    
    def _get_sample_values(self, col: Dict[str, Any], context: 'AgentContext') -> List[str]:
        """Get sample values for the column."""
        # Placeholder for sample value extraction
        # In a full implementation, this would query the database
        # For now, return empty list
        return [] 