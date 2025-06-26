import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import deepeval for human-readable answer evaluation
try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    
    # Define the custom GEval metric for answer correctness
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
    
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not available. Install with: pip install deepeval")
    DEEPEVAL_AVAILABLE = False
    correctness_metric = None


def deepeval_correctness(execution_results: List[Dict[str, Any]]) -> float:
    """Calculate answer correctness using DeepEval metric on human-readable predictions."""
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available for evaluation")
        return 0.0
        
    successful_cases = 0
    total_cases = len(execution_results)
    if total_cases == 0:
        return 0.0

    for result in execution_results:
        try:
            example = result["example"]
            pred_answer = result.get("pred_answer", "")
            gt_result = result.get("gt_result")

            # Convert gt_result to list of strings for context
            if gt_result is None:
                context_list = ["No data returned."]
            elif isinstance(gt_result, (list, tuple)):
                if len(gt_result) == 0:
                    context_list = ["No data found."]
                else:
                    context_list = [str(row) for row in gt_result]
            else:
                context_list = [str(gt_result)]

            test_case = LLMTestCase(
                input=example["question"],
                actual_output=pred_answer,
                context=context_list
            )
            
            correctness_metric.measure(test_case)

            result["deepeval_score"] = correctness_metric.score
            result["deepeval_reason"] = correctness_metric.reason

            logger.info(f"  - DeepEval Correctness score: {correctness_metric.score}, successful: {correctness_metric.is_successful()}")
            if correctness_metric.reason is not None:
                logger.info(f"  - DeepEval reason: {correctness_metric.reason}")

            if correctness_metric.is_successful():
                successful_cases += 1
        except Exception as e:
            logger.warning(f"Error evaluating example with DeepEval: {e}")
            continue
    
    return successful_cases / total_cases if total_cases > 0 else 0.0


def calculate_metrics(predictions: List[str], examples: List[Dict[str, Any]], dataset, metric_names: List[str]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Calculate specified metrics for human-readable predictions.
    Executes ground truth SQL queries once to get reference data for comparison.
    Returns a tuple of (metrics, execution_results).
    """
    
    execution_results = []
    for pred_answer, example in zip(predictions, examples):
        result_payload = {
            "example": example,
            "pred_answer": pred_answer.strip()
        }
        
        # Execute ground truth SQL to get reference data
        try:
            gt_sql = example.get("SQL", example.get("query", ""))
            if gt_sql:
                gt_success, gt_result = dataset.execute_sql(gt_sql, example)
                result_payload["gt_success"] = gt_success
                result_payload["gt_result"] = gt_result
                
                if gt_success:
                    logger.debug(f"Ground truth result: {gt_result}")
                else:
                    logger.warning(f"Ground truth SQL failed: {gt_result}")
            else:
                logger.warning("No ground truth SQL found in example")
                result_payload["gt_success"] = False
                result_payload["gt_result"] = None
                
        except Exception as e:
            logger.warning(f"Error executing ground truth SQL for example: {e}")
            result_payload["gt_success"] = False
            result_payload["gt_result"] = None

        execution_results.append(result_payload)
        
    # Calculate metrics
    metrics = {}
    for metric_name in metric_names:
        if metric_name == "deepeval_correctness":
            metrics[metric_name] = deepeval_correctness(execution_results)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            metrics[metric_name] = 0.0
            
    return metrics, execution_results 