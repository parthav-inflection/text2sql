from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol
import logging
import json

from src.models.base import BaseModel
from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class SQLExecutionTool:
    """Tool for executing SQL queries and returning formatted results."""
    
    def __init__(self, dataset: BaseDataset, example: Dict[str, Any]):
        self.dataset = dataset
        self.example = example
    
    def execute(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query and return formatted results.
        
        Returns:
            Dict with 'success', 'result', and optional 'error' keys
        """
        try:
            success, result = self.dataset.execute_sql(sql_query, self.example)
            
            if success:
                # Format results for model consumption
                if result is None:
                    formatted_result = "Query executed successfully. No data returned."
                elif isinstance(result, (list, tuple)):
                    if len(result) == 0:
                        formatted_result = "Query executed successfully. No rows found."
                    else:
                        # Format as readable text
                        formatted_result = self._format_query_results(result)
                else:
                    formatted_result = str(result)
                
                return {
                    "success": True,
                    "result": formatted_result,
                    "raw_result": result
                }
            else:
                return {
                    "success": False,
                    "error": str(result),
                    "result": f"SQL execution failed: {result}"
                }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution exception: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "result": f"SQL execution error: {error_msg}"
            }
    
    def _format_query_results(self, results: List[Any]) -> str:
        """Format query results for model consumption."""
        if not results:
            return "No data returned."
        
        # Handle single row
        if len(results) == 1:
            row = results[0]
            if isinstance(row, (tuple, list)):
                if len(row) == 1:
                    return f"Result: {row[0]}"
                else:
                    return f"Result: {', '.join(map(str, row))}"
            else:
                return f"Result: {row}"
        
        # Handle multiple rows
        formatted_rows = []
        for i, row in enumerate(results[:10]):  # Limit to first 10 rows
            if isinstance(row, (tuple, list)):
                formatted_rows.append(f"Row {i+1}: {', '.join(map(str, row))}")
            else:
                formatted_rows.append(f"Row {i+1}: {row}")
        
        result_text = "\n".join(formatted_rows)
        
        if len(results) > 10:
            result_text += f"\n... and {len(results) - 10} more rows"
            
        return result_text


class ModulePipeline(Protocol):
    """Protocol for module pipeline interface."""
    def execute(self, context: 'AgentContext') -> None:
        """Execute the module pipeline."""
        ...


class AgentContext:
    """Context object that flows through the agent pipeline."""
    
    def __init__(self, question: str, example: Dict[str, Any], dataset: BaseDataset):
        self.question = question
        self.example = example
        self.dataset = dataset
        
        # Pipeline state
        self.original_schema = ""
        self.processed_schema = ""
        self.relevant_tables = []
        self.relevant_columns = []
        self.metadata = {}
        
        # Tool calling state
        self.sql_tool = SQLExecutionTool(dataset, example)
        self.tool_calls = []
        self.tool_results = []
        
        # Final output
        self.final_answer = ""
        self.conversation_history = []
        
        # Module outputs
        self.module_outputs = {}
        
    def set_module_output(self, module_name: str, output: Any):
        """Store output from a specific module."""
        self.module_outputs[module_name] = output
        
    def get_module_output(self, module_name: str) -> Any:
        """Retrieve output from a specific module."""
        return self.module_outputs.get(module_name)


class BaseAgent(ABC):
    """Base class for Text2SQL agents using tool calling and summarization."""
    
    def __init__(
        self, 
        name: str,
        model: BaseModel, 
        pipeline: ModulePipeline,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent with injected dependencies.
        
        Args:
            name: Agent name
            model: The base model for tool calling and summarization
            pipeline: The module pipeline for processing
            config: Optional configuration overrides
        """
        self.name = name
        self.model = model
        self.pipeline = pipeline
        self.config = config or {}
        self.max_iterations = config.get('max_iterations', 5) if config else 5
        self.retry_temperature_boost = config.get('retry_temperature_boost', 0.1) if config else 0.1
        
        logger.info(f"Initialized {self.name} with model {model.name}")
        
    def process(self, context: AgentContext) -> str:
        """Main processing pipeline using tool calling approach."""
        logger.info(f"Processing question with {self.name}: {context.question}")
        
        # Initialize context with schema
        context.original_schema = context.dataset.get_schema(context.example)
        context.processed_schema = context.original_schema
        
        # Execute module pipeline for preprocessing
        try:
            self.pipeline.execute(context)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Continue with original schema
            
        # Run interactive tool calling session
        context.final_answer = self._run_tool_calling_session(context)
        
        return context.final_answer
    
    def _run_tool_calling_session(self, context: AgentContext) -> str:
        """Run interactive tool calling session with the model."""
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Tool calling iteration {iteration}/{self.max_iterations}")
            
            # Generate model response with tool calling (pass retry attempt for temperature adjustment)
            response = self._generate_model_response(context, retry_attempt=iteration-1)
            
            # Check if model made a tool call
            tool_call = self._extract_tool_call(response)
            
            if tool_call:
                # Execute the tool call
                tool_result = context.sql_tool.execute(tool_call['sql_query'])
                
                # Store tool interaction
                context.tool_calls.append(tool_call)
                context.tool_results.append(tool_result)
                context.conversation_history.append({
                    'type': 'tool_call',
                    'sql': tool_call['sql_query'],
                    'result': tool_result
                })
                
                logger.info(f"Executed SQL: {tool_call['sql_query']}")
                logger.info(f"Result: {tool_result['result']}")
                
                # If successful, get final summary
                if tool_result['success']:
                    final_answer = self._generate_final_summary(context, tool_result)
                    return final_answer
                else:
                    # SQL failed, continue for retry
                    logger.warning(f"SQL failed: {tool_result['error']}")
                    continue
            else:
                # No tool call found, treat as final answer
                logger.info("No tool call found, treating as final answer")
                return response.strip()
        
        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return "I was unable to generate a proper SQL query to answer your question after multiple attempts."
    
    def _generate_model_response(self, context: AgentContext, retry_attempt: int = 0) -> str:
        """Generate model response with tool calling capability."""
        # Calculate temperature for this attempt
        temperature_override = None
        if retry_attempt > 0:
            # Increase temperature for retries to encourage exploration
            base_temp = getattr(self.model, 'sampling_params', None)
            if base_temp and hasattr(base_temp, 'temperature'):
                base_temperature = base_temp.temperature
            else:
                base_temperature = 0.0  # Default fallback
            
            temperature_override = base_temperature + (retry_attempt * self.retry_temperature_boost)
            logger.info(f"Retry attempt {retry_attempt}: using temperature {temperature_override:.2f}")
        
        # Use the model's tool calling interface
        response = self.model.generate_response_with_tools(
            question=context.question,
            schema=context.processed_schema,
            conversation_history=context.conversation_history,
            temperature_override=temperature_override
        )
        return response
    
    def _extract_tool_call(self, response: str) -> Optional[Dict[str, str]]:
        """Extract tool call from model response."""
        # Look for tool call patterns
        patterns = [
            r'"run_sql_query"[^}]*"sql_query":\s*"([^"]+)"',
            r'"sql_query":\s*"([^"]+)"',
            r'run_sql_query\(["\']([^"\']+)["\']\)',
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                sql_query = match.group(1).strip()
                return {'sql_query': sql_query}
        
        return None
    
    def _generate_final_summary(self, context: AgentContext, tool_result: Dict[str, Any]) -> str:
        """Generate final human-readable summary."""
        summary_response = self.model.generate_summary(
            question=context.question,
            sql_result=tool_result['result'],
            conversation_history=context.conversation_history
        )
        return summary_response.strip()
    
    def generate_answer(self, question: str, schema: Dict[str, Any], dataset: BaseDataset, example: Dict[str, Any]) -> str:
        """
        High-level method to generate human-readable answer for a question.
        This is the main interface for the evaluation framework.
        """
        # Create context and process
        context = AgentContext(question, example, dataset)
        result_answer = self.process(context)
        return result_answer
    
    def cleanup(self):
        """Clean up agent resources."""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()


class StandardModulePipeline:
    """Standard implementation of module pipeline."""
    
    def __init__(self, modules: List['BaseModule']):
        """Initialize with list of modules sorted by priority."""
        self.modules = sorted(modules, key=lambda m: m.priority)
        logger.info(f"Initialized pipeline with {len(self.modules)} modules")
        
    def execute(self, context: AgentContext) -> None:
        """Execute modules in priority order."""
        # Execute module pipeline
        for module in self.modules:
            if module.is_enabled():
                logger.debug(f"Executing module: {module.name}")
                try:
                    module.process(context)
                except Exception as e:
                    logger.error(f"Module {module.name} failed: {e}")
                    if module.is_critical():
                        raise
                        
    def get_module_by_name(self, name: str) -> Optional['BaseModule']:
        """Get a specific module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def enable_module(self, name: str):
        """Enable a specific module."""
        module = self.get_module_by_name(name)
        if module:
            module.enable()
    
    def disable_module(self, name: str):
        """Disable a specific module."""
        module = self.get_module_by_name(name)
        if module:
            module.disable() 