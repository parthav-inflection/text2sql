import logging
from typing import List, Dict, Any, Tuple
import sqlparse

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
            gt_sql = example.get("query", "")
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
    # Handle common response formats
    response = response.strip()
    
    # Look for SQL between ```sql and ``` 
    if "```sql" in response.lower():
        start = response.lower().find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    # Look for SQL between ``` and ```
    if response.startswith("```") and response.count("```") >= 2:
        lines = response.split("\n")
        sql_lines = []
        in_code_block = False
        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                sql_lines.append(line)
        if sql_lines:
            return "\n".join(sql_lines).strip()
    
    # If no code blocks, try to find SQL keywords
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
    for keyword in sql_keywords:
        if keyword in response.upper():
            # Find the line with SQL keyword and take everything after
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if keyword in line.upper():
                    return "\n".join(lines[i:]).strip()
    
    # Return as-is if no patterns match
    return response


def normalize_result(result: Any) -> Any:
    """Normalize query results for comparison."""
    if isinstance(result, list):
        # Sort lists of tuples for consistent comparison
        if result and isinstance(result[0], tuple):
            return sorted(result)
        return sorted(result) if all(isinstance(x, (int, float, str)) for x in result) else result
    return result


def calculate_metrics(predictions: List[str], examples: List[Dict[str, Any]], dataset, metrics: List[str]) -> Dict[str, float]:
    """Calculate specified metrics."""
    results = {}
    
    if "execution_accuracy" in metrics:
        results["execution_accuracy"] = execution_accuracy(predictions, examples, dataset)
        logger.info(f"Execution accuracy: {results['execution_accuracy']:.3f}")
    
    return results 