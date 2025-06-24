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
    
    # 1. Look for SQL between ```sql and ``` (prefer last occurrence for verbose responses)
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        # Take the last SQL block (often the final answer in verbose responses)
        sql = matches[-1].strip()
        if sql:
            return _clean_sql(sql)
    
    # 2. Look for SQL between ``` and ``` (prefer last occurrence)
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Take the last code block
        sql = matches[-1].strip()
        if sql and _contains_sql_keywords(sql):
            return _clean_sql(sql)
    
    # 3. Look for SQL after specific patterns (take first occurrence)
    sql_patterns = [
        r'(?:Here (?:is|\'s) the SQL query:?\s*\n*)(.*?)(?:\n\n|\n(?=\w+:)|\nThis query|$)',
        r'(?:SQL (?:Query|query):?\s*\n*)(.*?)(?:\n\n|\n(?=\w+:)|\nThis query|$)',
        r'(?:Final SQL query:?\s*\n*)(.*?)(?:\n\n|\n(?=\w+:)|\nThis query|$)',
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if sql and _contains_sql_keywords(sql):
                return _clean_sql(sql)
    
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
                    # Stop if we hit explanatory text
                    if any(stop_word in line.lower() for stop_word in [
                        'this query', 'explanation', 'note:', 'the above', 
                        'step-by-step', 'reasoning:', 'here is', 'this sql'
                    ]):
                        break
                    sql_lines.append(line)
                    if line.endswith(';'):
                        break
                elif sql_lines:  # Stop if we have SQL and hit empty line
                    break
            
            if sql_lines:
                return _clean_sql(' '.join(sql_lines))
    
    # 5. If no patterns match, clean the response and hope for the best
    return _clean_sql(response)


def _contains_sql_keywords(text: str) -> bool:
    """Check if text contains SQL keywords."""
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP"]
    text_upper = text.upper()
    return any(keyword in text_upper for keyword in sql_keywords)


def _clean_sql(sql: str) -> str:
    """Clean SQL query by removing common prefixes and suffixes."""
    if not sql:
        return ""
    
    # Remove common prefixes
    sql = re.sub(r'^(Here is|Here\'s|The|A|An)\s+', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'(SQL\s+)?(query|statement)\s*:?\s*', '', sql, flags=re.IGNORECASE)
    
    # Remove explanatory suffixes
    sql = re.sub(r'\s*(This query|The query|This SQL).*$', '', sql, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())
    
    # Ensure ends with semicolon
    if sql and not sql.endswith(';'):
        sql += ';'
    
    return sql


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