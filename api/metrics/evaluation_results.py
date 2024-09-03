from typing import List, Optional
from pydantic import BaseModel, Field

class MetricResult:
    score: int
    reason: str

class ConsistencyResult:
    questions: str
    answers: str
    result: str
    score: float

class DeepEvalEvluationResult:
    correctness: MetricResult
    answer_relevancy: MetricResult
    nswer_relevancy: MetricResult
    faithfulness: MetricResult
    contextual_precision: MetricResult
    contextual_recall: MetricResult
    contextual_precision: MetricResult
    contextual_relevancy: MetricResult
    consistency: ConsistencyResult

