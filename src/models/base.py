from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging
import json
import re

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all text2sql models with tool calling support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.model_path = config["model_path"]
        
    @abstractmethod
    def generate(self, prompts: List[str], temperature_override: float = None) -> List[str]:
        """Generate responses for given prompts with optional temperature override."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass
    
    def generate_response_with_tools(self, question: str, schema: str, conversation_history: List[Dict[str, Any]] = None, temperature_override: float = None) -> str:
        """
        Generate response with tool calling capability.
        This is the main interface for interactive SQL execution.
        """
        conversation_history = conversation_history or []
        
        # Create tool calling prompt
        prompt = self._create_tool_calling_prompt(question, schema, conversation_history)
        
        # Generate response
        responses = self.generate([prompt], temperature_override=temperature_override)
        return responses[0] if responses else ""
    
    def generate_summary(self, question: str, sql_result: str, conversation_history: List[Dict[str, Any]] = None, temperature_override: float = None) -> str:
        """
        Generate final human-readable summary of SQL results.
        """
        conversation_history = conversation_history or []
        
        # Create summarization prompt
        prompt = self._create_summarization_prompt(question, sql_result, conversation_history)
        
        # Generate summary
        responses = self.generate([prompt], temperature_override=temperature_override)
        return responses[0] if responses else ""
    
    def _create_tool_calling_prompt(self, question: str, schema: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Create prompt for tool calling interaction."""
        conversation_history = conversation_history or []
        
        # Check if this is a retry (has conversation history)
        if conversation_history:
            return self._create_retry_prompt(question, schema, conversation_history)
        else:
            return self._create_initial_tool_prompt(question, schema)
    
    def _create_initial_tool_prompt(self, question: str, schema: str) -> str:
        """Create initial tool calling prompt."""
        simplified_schema = self._simplify_schema(schema)
        
        prompt = f"""You are a SQL expert assistant. Your task is to answer the user's question by writing and executing SQL queries.

You have access to a tool called 'run_sql_query' that can execute SQL queries against the database. Use this tool to get the data needed to answer the question, then provide a clear, concise summary of the results.

Database Schema:
{simplified_schema}

User Question: {question}

To use the SQL execution tool, respond with a JSON object like this:
{{
  "tool_calls": [{{
    "type": "function",
    "function": {{
      "name": "run_sql_query",
      "arguments": {{
        "sql_query": "YOUR_SQL_QUERY_HERE"
      }}
    }}
  }}]
}}

Write a SQL query to answer the question."""

        return self._format_prompt_for_model(prompt)
    
    def _create_retry_prompt(self, question: str, schema: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Create retry prompt with conversation history."""
        simplified_schema = self._simplify_schema(schema)
        
        prompt = f"""You are a SQL expert assistant. Your previous SQL query failed. Please analyze the error and try again.

Database Schema:
{simplified_schema}

User Question: {question}

Previous attempts:"""

        # Add conversation history
        for i, entry in enumerate(conversation_history):
            if entry['type'] == 'tool_call':
                prompt += f"""

Attempt {i+1}:
SQL: {entry['sql']}
Result: {entry['result']['result']}"""
                if not entry['result']['success']:
                    prompt += f"""
Error: {entry['result'].get('error', 'Unknown error')}"""

        prompt += """

Please write a corrected SQL query. Use the same JSON format:
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "run_sql_query",
      "arguments": {
        "sql_query": "YOUR_CORRECTED_SQL_QUERY_HERE"
      }
    }
  }]
}"""

        return self._format_prompt_for_model(prompt)
    
    def _create_summarization_prompt(self, question: str, sql_result: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Create prompt for summarizing SQL results."""
        prompt = f"""Based on the SQL query results below, provide a clear and concise answer to the user's question.

User Question: {question}

SQL Query Results:
{sql_result}

Please provide a human-readable summary that:
1. Directly answers the user's question
2. Includes all relevant data from the results
3. Is concise and easy to understand
4. Does not mention SQL or technical details

Answer:"""

        return self._format_prompt_for_model(prompt)
    
    def _simplify_schema(self, schema: str) -> str:
        """Simplify schema for better model understanding."""
        # Remove extra whitespace and formatting
        lines = []
        for line in schema.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_prompt_for_model(self, prompt: str) -> str:
        """Format prompt based on model capabilities (override in subclasses)."""
        return prompt 