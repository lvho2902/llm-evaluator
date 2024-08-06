# LL-7SZFs6jbtQFVWQaVAA3VstfKx8Ecq3O0do5Y5oUS2fLHJ3UcCSGNu03HeqmEQ41F
import os

os.environ['REQUESTS_CA_BUNDLE'] = 'D:/GITHUB/llm-evaluator/api.llama-api.crt'


from llamaapi import LlamaAPI
from langchain.chains import create_tagging_chain
from langchain_experimental.llms import ChatLlamaAPI

llama = LlamaAPI("LL-7SZFs6jbtQFVWQaVAA3VstfKx8Ecq3O0do5Y5oUS2fLHJ3UcCSGNu03HeqmEQ41F")
model = ChatLlamaAPI(client=llama)

schema = {
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "the sentiment encountered in the passage",
        },
        "aggressiveness": {
            "type": "integer",
            "description": "a 0-10 score of how aggressive the passage is",
        },
        "language": {"type": "string", "description": "the language of the passage"},
    }
}

chain = create_tagging_chain(schema, model)
chain.run("give me your money")