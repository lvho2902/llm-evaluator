from langchain_community.llms import EdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
import os

EDENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYTViOWY5Y2EtMDA2My00MWJhLTkwZjQtZjgwYzNjYmM1YThhIiwidHlwZSI6ImFwaV90b2tlbiJ9.4klbUzRdEaxeqz5oCDDGXnDqcEc-vJ2fSeQE648pb2I"

# llm = EdenAI(edenai_api_key=EDENAI_API_KEY,
#             feature="text",
#             provider="openai",
#             temperature=0.2,
#             max_tokens=250)


# prompt = """
# User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
# Assistant:
# """

# print(llm(prompt))

# embeddings = EdenAiEmbeddings(edenai_api_key=EDENAI_API_KEY, provider="openai")

# docs = ["It's raining right now", "cats are cute"]
# document_result = embeddings.embed_documents(docs)