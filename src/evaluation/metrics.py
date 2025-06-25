import logging
from typing import List, Dict, Any, Tuple
import sqlparse
import re
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

logger = logging.getLogger(__name__)

# Define the custom GEval metric for SQL correctness
correctness_metric = GEval(
    name="Correctness",
    threshold=0.8,
    criteria="""Evaluate the factual correctness of the 'actual output' against the 'context', which contains the ground truth data from the database. The 'input' is the original question.

1.  **Data Equivalence**:
    *   Does the 'actual output' contain the same data as the 'context'?
    *   Are all numerical values, dates, and specific data points from the 'context' present and accurate in the 'actual output'?
    *   Are there any missing or extra data points in the 'actual output' compared to the 'context'?

2.  **Factual Consistency**:
    *   Does the 'actual output' introduce any information that contradicts the 'context'?
    *   Are relationships between data points preserved correctly?

Scoring Guidelines:
- Score 1.0: The 'actual output' is perfectly equivalent to the 'context'.
- Score 0.9: All critical information from the 'context' is present, minor formatting differences are acceptable.
- Score 0.8: Most essential information from the 'context' is present and correct.
- Score < 0.8: The 'actual output' is missing critical information or contains factual errors when compared to the 'context'.

Note: The evaluation should focus purely on data equivalence and factual correctness.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
)


def deepeval_correctness(execution_results: List[Dict[str, Any]]) -> float:
    """Calculate SQL correctness using a custom GEval metric on execution results."""
    successful_cases = 0
    total_cases = len(execution_results)
    if total_cases == 0:
        return 0.0

    for result in execution_results:
        try:
            example = result["example"]
            pred_result = result.get("pred_result")
            gt_result = result.get("gt_result")

            # Convert gt_result to list of strings for context
            if gt_result is None:
                context_list = []
            elif isinstance(gt_result, (list, tuple)):
                context_list = [str(row) for row in gt_result]
            else:
                context_list = [str(gt_result)]

            test_case = LLMTestCase(
                input=example["question"],
                actual_output=normalize_result(pred_result),
                context=context_list
            )
            
            correctness_metric.measure(test_case)
            logger.info(f"  - DeepEval Correctness score: {correctness_metric.score}, successful: {correctness_metric.is_successful()}")
            if correctness_metric.reason is not None:
                logger.info(f"  - DeepEval reason: {correctness_metric.reason}")

            if correctness_metric.is_successful():
                successful_cases += 1
        except Exception as e:
            logger.warning(f"Error evaluating example with DeepEval: {e}")
            continue
    
    return successful_cases / total_cases if total_cases > 0 else 0.0


def execution_accuracy(execution_results: List[Dict[str, Any]]) -> float:
    """Calculate execution accuracy by comparing pre-computed query results."""
    correct = 0
    total = len(execution_results)
    if total == 0:
        return 0.0
    
    for result in execution_results:
        if result.get("pred_success") and result.get("gt_success"):
            if normalize_result(result["pred_result"]) == normalize_result(result["gt_result"]):
                correct += 1
        elif not result.get("pred_success") and not result.get("gt_success"):
            # Both failed - could be considered correct in some cases
            pass
                
    return correct / total


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
    """
    Executes SQL queries and calculates specified metrics.
    Executes each query only once and reuses the results for all metrics.
    """
    
    execution_results = []
    for pred_sql_raw, example in zip(predictions, examples):
        result_payload = {"example": example}
        
        # Execute predicted SQL
        try:
            pred_sql = extract_sql_from_response(pred_sql_raw)
            pred_success, pred_result = dataset.execute_sql(pred_sql, example)
            result_payload["pred_success"] = pred_success
            result_payload["pred_result"] = pred_result
        except Exception as e:
            logger.warning(f"Error executing predicted SQL for example: {e}")
            result_payload["pred_success"] = False
            result_payload["pred_result"] = None

        # Execute ground truth SQL
        try:
            gt_sql = example.get("SQL", example.get("query", ""))
            gt_success, gt_result = dataset.execute_sql(gt_sql, example)
            result_payload["gt_success"] = gt_success
            result_payload["gt_result"] = gt_result
        except Exception as e:
            logger.warning(f"Error executing ground truth SQL for example: {e}")
            result_payload["gt_success"] = False
            result_payload["gt_result"] = None

        execution_results.append(result_payload)
        
    metrics = {}
    for metric_name in metric_names:
        if metric_name == "execution_accuracy":
            metrics[metric_name] = execution_accuracy(execution_results)
        elif metric_name == "deepeval_correctness":
            metrics[metric_name] = deepeval_correctness(execution_results)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            metrics[metric_name] = 0.0
            
    return metrics 