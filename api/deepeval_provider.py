import os
from custom_llm import CustomOllama
from langchain_community.chat_models import ChatOllama
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

OLLAMA_SERVER_URL="https://forward-martha-gentle-bronze.trycloudflare.com"

def make_model(model="llama3.1"):
    custom_model = ChatOllama(model=model, base_url=os.getenv('OLLAMA_SERVER_URL', OLLAMA_SERVER_URL), temperature=0)
    return CustomOllama(model=custom_model)

def run_deep_eval(model, prediction):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model = model
    )

    test_case = LLMTestCase(
        input=prediction['question'],
        actual_output=prediction['result'],
        expected_output=prediction['answer']
    )

    answer_relevancy_metric = AnswerRelevancyMetric(model=model, threshold=0.7)
    test_results = evaluate([test_case], [answer_relevancy_metric, correctness_metric])

    test_result = test_results[0]

    print(test_result)

    # Create the formatted dictionary
    formatted_test_result = {
        metric.name: {
            "score": metric.score,
            "reason": metric.reason
        }
        for metric in test_result.metrics_data
    }

    return formatted_test_result


