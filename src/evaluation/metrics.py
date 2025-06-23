import logging
from typing import List, Dict, Any, Tuple
import sqlparse
import re

logger = logging.getLogger(__name__)


def execution_accuracy(predictions: List[str], examples: List[Dict[str, Any]], dataset) -> float:
    """Calculate execution accuracy by comparing query results."""
    correct = 0
    total = len(predictions)
    
    for pred_sql, example in zip(predictions, examples):
        try:
            # Clean up the predicted SQL
            pred_sql = extract_sql_from_response(pred_sql)
            
            # Execute predicted SQL
            pred_success, pred_result = dataset.execute_sql(pred_sql, example)
            
            # Execute ground truth SQL
            gt_sql = example.get("SQL", example.get("query", ""))
            gt_success, gt_result = dataset.execute_sql(gt_sql, example)
            
            # Compare results
            if pred_success and gt_success:
                if normalize_result(pred_result) == normalize_result(gt_result):
                    correct += 1
            elif not pred_success and not gt_success:
                # Both failed - could be considered correct in some cases
                pass
                
        except Exception as e:
            logger.warning(f"Error evaluating example: {e}")
            continue
    
    return correct / total if total > 0 else 0.0


def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from model response."""
    if not response:
        return ""
    
    response = response.strip()
    
    # Handle common response formats
    
    # 1. Look for SQL between ```sql and ``` 
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Look for SQL between ``` and ```
    code_block_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 3. Look for SQL after "SQL Query:" or similar patterns
    sql_patterns = [
        r'SQL\s*Query\s*:\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)',
        r'Query\s*:\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)',
        r'Answer\s*:\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)',
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 4. Look for SQL keywords and extract from there
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    for keyword in sql_keywords:
        pattern = rf'\b{keyword}\b'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            # Find the start of the SQL statement
            start_pos = match.start()
            sql_part = response[start_pos:]
            
            # Clean up the SQL part
            lines = sql_part.split('\n')
            sql_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    sql_lines.append(line)
                    # Stop if we hit explanatory text
                    if line.endswith(';'):
                        break
                    # Stop if next line looks like explanation
                    if any(stop_word in line.lower() for stop_word in ['explanation', 'this query', 'note:', 'the above']):
                        break
            
            if sql_lines:
                return ' '.join(sql_lines)
    
    # 5. If no patterns match, clean the response and hope for the best
    # Remove common prefixes/suffixes
    response = re.sub(r'^(Here is|Here\'s|The|A|An)\s+', '', response, flags=re.IGNORECASE)
    response = re.sub(r'(SQL\s+)?(query|statement)\s*:?\s*', '', response, flags=re.IGNORECASE)
    
    # Take first line/sentence that looks like SQL
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line and any(keyword.lower() in line.lower() for keyword in sql_keywords):
            return line
    
    return response.strip()


def normalize_result(result: Any) -> str:
    """Normalize query results for comparison."""
    if result is None:
        return ""
    
    if isinstance(result, (list, tuple)):
        # Sort tuples/rows for consistent comparison
        try:
            normalized = sorted([tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in result])
            return str(normalized)
        except (TypeError, ValueError):
            # If sorting fails, convert to string as-is
            return str(result)
    
    return str(result)


def calculate_metrics(predictions: List[str], examples: List[Dict[str, Any]], dataset, metric_names: List[str]) -> Dict[str, float]:
    """Calculate specified metrics for predictions."""
    metrics = {}
    
    for metric_name in metric_names:
        if metric_name == "execution_accuracy":
            metrics[metric_name] = execution_accuracy(predictions, examples, dataset)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            metrics[metric_name] = 0.0
    
    return metrics 