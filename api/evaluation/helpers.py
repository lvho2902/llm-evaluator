from model.ollama_model import OllamaModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate
from metrics import (
    GEval, 
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

def create_metrics(custom_model):
    """
    Create a list of metrics for evaluating the model's output.
    """
    return [
        # GEval(
        #     name="Correctness",
        #     criteria="Determine whether the actual output is factually correct based on the expected output.",
        #     evaluation_steps=[
        #         "Check whether the facts in 'actual output' contradict any facts in 'expected output'.",
        #         "Heavily penalize omission of detail.",
        #         "Vague language or contradicting opinions are acceptable."
        #     ],
        #     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #     model=custom_model, async_mode=False
        # ),
        AnswerRelevancyMetric(model=custom_model),
        FaithfulnessMetric(model=custom_model),
        ContextualPrecisionMetric(model=custom_model),
        ContextualRecallMetric(model=custom_model),
        # ContextualRelevancyMetric(model=custom_model, async_mode=False)
    ]

def create_test_cases(predictions, retrieved_docs):
    """
    Create an LLM test case from the prediction and retrieved documents.
    """
    test_cases = []

    for i in range(0, len(predictions)):
        test_case = LLMTestCase(
            input=predictions[i].get('question'),
            expected_output=predictions[i].get('answer'),
            actual_output=predictions[i].get('result'),
            retrieval_context=retrieved_docs[i])
        test_cases.append(test_case)
    return test_cases

def create_test_case(prediction, retrieved_docs):
    """
    Create an LLM test case from the prediction and retrieved documents.
    """
    return LLMTestCase(
        input=prediction.get('question'),
        expected_output=prediction.get('answer'),
        actual_output=prediction.get('result'),
        retrieval_context=retrieved_docs
    )

def format_test_results(test_results):
    """
    Format the test results into a dictionary.
    """
    if not test_results:
        return {'deepeval': {}}
    
    test_result = test_results[0]
    return {
        'deepeval': {
            metric.name: {
                "score": metric.score,
                "reason": metric.reason
            }
            for metric in test_result.metrics_data
        }
    }

def run_deep_eval(model, prediction, retrieved_docs):
    """
    Run deep evaluation on the model's output using multiple metrics.
    """
    try:
        custom_model = OllamaModel(model=model)
        metrics = create_metrics(custom_model)
        test_case = create_test_case(prediction, retrieved_docs)
        test_results = evaluate([test_case], metrics, run_async=True, use_cache=True)
        format_results = format_test_results(test_results)
        return format_results

    except KeyError as e:
        return {"error": f"Missing required prediction field: {e}"}
    except Exception as e:
        return {"error": str(e)}