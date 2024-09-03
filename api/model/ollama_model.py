import os
import logging
from typing import Optional, Tuple, List
from deepeval.models import DeepEvalBaseLLM
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOllama

class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the OllamaModel with the specified model name.
        """
        self.model_name = model or "llama3.1"
        self.args = args
        self.kwargs = kwargs
        super().__init__(self.model_name)

    def load_model(self) -> ChatOllama:
        """
        Load the ChatOllama model with configuration from environment variables.
        """
        try:
            return ChatOllama(
                model=self.model_name,
                base_url=os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434"),
                temperature=0,
                num_ctx=16384,
                *self.args,
                **self.kwargs,
            )
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generate a response for a given prompt using the model.
        """
        try:
            ollama_model = self.load_model()
            res = ollama_model.invoke(prompt)
            logging.info(f"Response generated for prompt '{prompt}':\n {res.content}")
            return res.content
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response for a given prompt using the model.
        """
        try:
            ollama_model = self.load_model()
            res = await ollama_model.ainvoke(prompt)
            # logging.info(f"Asynchronous response generated for prompt '{prompt}':\n {res.content}")
            logging.info(f"Asynchronous response generated':\n {res.content}")
            return res.content
        except Exception as e:
            logging.error(f"Error generating asynchronous response: {str(e)}")
            raise

    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> List[str]:
        """
        Generate multiple sample responses for a given prompt.
        """
        try:
            ollama_model = self.load_model()
            original_n, original_temperature = ollama_model.n, ollama_model.temperature
            ollama_model.n, ollama_model.temperature = n, temperature

            generations = ollama_model._generate([HumanMessage(prompt)]).generations
            completions = [r.text for r in generations]

            # Restore original parameters
            ollama_model.n, ollama_model.temperature = original_n, original_temperature

            logging.info(f"Generated {n} samples for prompt '{prompt}':\n {completions}")
            return completions
        except Exception as e:
            logging.error(f"Error generating samples: {str(e)}")
            raise

    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        """
        return self.model_name