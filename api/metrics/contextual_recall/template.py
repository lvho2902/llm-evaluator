class ContextualRecallTemplate:
    @staticmethod
    def generate_reason(
        expected_output, supportive_reasons, unsupportive_reasons, score
    ):
        return f"""
Given the original expected output, a list of supportive reasons, and a list of unsupportive reasons (which is deduced directly from the 'expected output'), and a contextual recall score (closer to 1 the better), summarize a CONCISE reason for the score.
A supportive reason is the reason why a certain sentence in the original expected output can be attributed to the node in the retrieval context.
An unsupportive reason is the reason why a certain sentence in the original expected output cannot be attributed to anything in the retrieval context.
In your reason, you should related supportive/unsupportive reasons to the sentence number in expected output, and info regarding the node number in retrieval context to support your final reason. The first mention of "node(s)" should specify "node(s) in retrieval context)".

### Example JSON Output:
{{
    "reason": "The score is <contextual_recall_score> because <your_reason>."
}}
===== END OF EXAMPLE ======

**IMPORTANT:**
DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

### Instructions:

Here is the Contextual Recall Score, Expected Output, Supportive Reasons, and Unsupportive Reasons:

**Contextual Recall Score:**
{score}

**Expected Output:**
{expected_output}

**Supportive Reasons:**
{supportive_reasons}

**Unsupportive Reasons:**
{unsupportive_reasons}

**RESPONSE:**
- Return only and exactly in the JSON format as shown in the Example JSON.
- Do not include any additional text, explanations, or information outside of the JSON format.
- Your response must strictly follow the example JSON structure provided above.
"""

    @staticmethod
    def generate_verdicts(expected_output, retrieval_context):
        return f"""
For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the nodes of retrieval contexts. Please generate a list of JSON with two keys: `verdict` and `reason`.
The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the sentence can be attributed to any parts of the retrieval context, else answer 'no'.
The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said sentence. You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 

### Example JSON Output:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
        ...
    ]  
}}
===== END OF EXAMPLE ======

**IMPORTANT:**
Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of sentences in of `expected output`.

### Instructions:

Here is the Expected Output and Retrieval Context:

**Expected Output:**
{expected_output}

**Retrieval Context:**
{retrieval_context}

**RESPONSE:**
- Return only and exactly in the JSON format as shown in the Example JSON.
- Do not include any additional text, explanations, or information outside of the JSON format.
- Your response must strictly follow the example JSON structure provided above.
"""
