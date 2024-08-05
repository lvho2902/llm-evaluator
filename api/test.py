from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

llm = ChatMistralAI(
    mistral_api_key="M3yhMBJSuk55x1LmwlihJ8EtEvFlWIlF",
    model="mistral-small",
    temperature=0,
    max_retries=2,
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)