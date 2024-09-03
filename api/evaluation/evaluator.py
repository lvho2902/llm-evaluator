import io
import os
import re
import json
import time
import random
import logging
import itertools
import sentry_sdk
import pandas as pd
from dotenv import load_dotenv
import pypdf
import faiss
from typing import Dict, List, Union
from json import JSONDecodeError

from langchain.chains import RetrievalQA, QAGenerationChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import SVMRetriever, TFIDFRetriever
from langchain_community.vectorstores import FAISS
from langchain.evaluation import QAEvalChain
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from text_utils import (
    GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST,
    GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI,
    QA_CHAIN_PROMPT, CONSISTENCY_QA_CHAIN_PROMPT, GRADE_ANSWERS_CONSISTENCY_PROMPT
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import EdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from model.ollama_model import OllamaModel
from database.data_provider import DataProvider
import evaluation.helpers as helpers

def generate_eval(text: str, chunk: int, evaluator, prompt: str, logger: logging.Logger) -> List[Dict]:
    """
    Generates an evaluation QA pair from the given text chunk.
    """
    logger.error("Generating QA pair...")
    num_of_chars = len(text)
    eval_set = []
    for _ in range(10):
        try:
            starting_index = random.randint(0, num_of_chars - chunk)
            sub_sequence = text[starting_index:starting_index + chunk]
            chain = QAGenerationChain.from_llm(evaluator, prompt=prompt)
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            break
        except JSONDecodeError:
            logger.error("Error generating QA pair, retrying...")
    return list(itertools.chain.from_iterable(eval_set))

def split_texts(text: str, chunk_size: int, chunk_overlap: int, split_method: str, logger: logging.Logger) -> List[str]:
    """
    Splits the provided text based on the specified method.
    """
    logger.info("Splitting document...")
    splitter_class = {
        "RecursiveTextSplitter": RecursiveCharacterTextSplitter,
        "CharacterTextSplitter": CharacterTextSplitter
    }.get(split_method, RecursiveCharacterTextSplitter)

    text_splitter = splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def make_llm(model: str, logger: logging.Logger):
    """
    Instantiates and returns an LLM based on the specified model name.
    """
    logger.info(f"Creating LLM instance for model: {model}")
    ollama_base_url = os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")
    logger.debug(f"Ollama base URL set to: {ollama_base_url}")
    llm_mapping = {
        "mistral": ChatOllama(model="mistral", temperature=0, top_k=0, base_url=ollama_base_url),
        "llama3": ChatOllama(model="llama3", temperature=0, base_url=ollama_base_url),
        "llama3.1": ChatOllama(model="llama3.1", temperature=0, base_url=ollama_base_url),
        "llama2:13b": ChatOllama(model="llama2:13b", temperature=0, base_url=ollama_base_url)
    }

    llm_instance = llm_mapping.get(model)
    if llm_instance:
        logger.info(f"Successfully created LLM instance for model: {model}")
        return llm_instance
    else:
        logger.warning(f"Model '{model}' not recognized. Falling back to default OpenAI model.")
        return ChatOpenAI(model_name=model, temperature=0)

def make_evaluator(evaluator: str, logger: logging.Logger) -> Union[ChatOllama, ChatOpenAI]:
    """
    Instantiates and returns an evaluator LLM based on the specified evaluator name.
    """
    logger.info(f"Creating evaluator instance for model: {evaluator}")
    ollama_base_url = os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")
    # Define a mapping of evaluator names to model instances
    llm_mapping = {
        "mistral": ChatOllama(model="mistral", temperature=0, base_url=ollama_base_url),
        "llama3": ChatOllama(model="llama3", temperature=0, base_url=ollama_base_url),
        "llama3.1": ChatOllama(model="llama3.1", temperature=0, base_url=ollama_base_url),
        "llama2:13b": ChatOllama(model="llama2:13b", temperature=0, base_url=ollama_base_url)
    }
    evaluator_instance = llm_mapping.get(evaluator)
    if evaluator_instance:
        logger.info(f"Successfully created evaluator instance for model: {evaluator}")
        return evaluator_instance
    else:
        logger.warning(f"Evaluator model '{evaluator}' not recognized. Falling back to default OpenAI model.")
        return ChatOpenAI(model_name=evaluator, temperature=0)
    

def make_retriever(splits: List[str], retriever_type: str, embeddings: str, num_neighbors: int, logger: logging.Logger) -> Union[FAISS, SVMRetriever, TFIDFRetriever]:
    """
    Creates and returns a retriever based on the specified type and embeddings.
    """
    logger.info("Creating retriever...")
    embedding_mapping = {
        "Ollama": OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434"))
    }
    embd = embedding_mapping.get(embeddings, OpenAIEmbeddings())
    logger.info(f"Using embeddings: {embeddings}")
    retriever_mapping = {
        "similarity-search": lambda: FAISS.from_texts(splits, embd).as_retriever(k=num_neighbors),
        "SVM": lambda: SVMRetriever.from_texts(splits, embd),
        "TF-IDF": lambda: TFIDFRetriever.from_texts(splits)
    }
    retriever_func = retriever_mapping.get(retriever_type)
    if retriever_func:
        retriever = retriever_func()
        logger.info(f"Created retriever of type: {retriever_type}")
        return retriever
    else:
        logger.error(f"Unknown retriever type: {retriever_type}. Falling back to default TF-IDF retriever.")
        return TFIDFRetriever.from_texts(splits)

def make_chain(llm, retriever, prompt: str, input_key: str, logger: logging.Logger):
    """
    Creates a QA chain with the provided LLM, retriever, and prompt.
    """
    logger.info("Creating QA chain...")
    chain_type_kwargs = {"prompt": prompt}
    return RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs, input_key=input_key
    )

def grade_model_answer(predicted_dataset, predictions, grader_llm, grade_answer_prompt: str, logger: logging.Logger):
    """
    Grades the model's answers using the specified grading prompt.
    """
    logger.info("Grading model answer...")
    prompt = {
        "Fast": GRADE_ANSWER_PROMPT_FAST,
        "Descriptive w/ bias check": GRADE_ANSWER_PROMPT_BIAS_CHECK,
        "OpenAI grading prompt": GRADE_ANSWER_PROMPT_OPENAI
    }.get(grade_answer_prompt, GRADE_ANSWER_PROMPT)
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt)
    return eval_chain.evaluate(predicted_dataset, predictions, question_key="question", prediction_key="result")

def grade_model_retrieval(gt_dataset, predictions, grader_llm, grade_docs_prompt: str, logger: logging.Logger):
    """
    Grades the relevance of retrieved documents using the specified grading prompt.
    """
    logger.info("Grading document relevance...")
    prompt = GRADE_DOCS_PROMPT_FAST if grade_docs_prompt == "Fast" else GRADE_DOCS_PROMPT
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt)
    return eval_chain.evaluate(gt_dataset, predictions, question_key="question", prediction_key="result")

def make_retrieved_docs(retriever, qa_pair):
    """
    Retrieves documents relevant to the QA pair's question.
    """
    return [doc.page_content for doc in retriever.get_relevant_documents(qa_pair["question"])]

def run_consistency_eval(chain, evaluator, text, logger: logging.Logger):    
    """
    Runs a consistency evaluation on generated QA pairs.
    """
    logger.info("Running consistency evaluation...")
    eval_pair = generate_eval(text, len(text), evaluator, CONSISTENCY_QA_CHAIN_PROMPT, logger)
    if not eval_pair:
        logger.error("Failed to generate evaluation pair for consistency check.")
        return {}
    predictions = [chain({'question': q}) for q in eval_pair[0]['question']]
    eval_chain = QAEvalChain.from_llm(llm=evaluator, prompt=GRADE_ANSWERS_CONSISTENCY_PROMPT)
    results = "\n\n".join([f"Answer {i+1}: {pred['result']}" for i, pred in enumerate(predictions)])
    inputs = [{"query": "", "answer": "", "result": results}]
    graded_outputs = eval_chain.apply(inputs)

    graded_results = {
        'questions': "\n\n".join([f"QUESTION {i+1}: {pred['question']}" for i, pred in enumerate(predictions)]),
        'answers': "\n\n".join([f"ANSWER {i+1}: {pred['result']}" for i, pred in enumerate(predictions)]),
        'results': graded_outputs[0]['results'],
        'score': 1 if "Inconsistent" not in graded_outputs[0]['results'] else 0
    }
    return graded_results

load_dotenv()
if os.environ.get("ENVIRONMENT") != "development":
    sentry_sdk.init(
    dsn="https://065aa152c4de4e14af9f9e7335c8eae4@o4505106202820608.ingest.sentry.io/4505106207735808",
    traces_sample_rate=1.0,
    )

def read_file(file, logger):
    """Reads the content of a file and handles different types."""
    logger.info(f"Reading file: {file.filename}")
    contents = file.file.read()

    if file.content_type == 'application/pdf':
        return extract_pdf_text(contents, file.filename, logger)
    elif file.content_type == 'text/plain':
        logger.info(f"File {file.filename} is a TXT")
        return contents.decode()
    else:
        logger.warning(f"Unsupported file type for file: {file.filename}")
        return None

def extract_pdf_text(contents, filename, logger):
    """Extracts text from a PDF file."""
    logger.info(f"File {filename} is a PDF")
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF file {filename}: {str(e)}")
        return None

def assemble_results(deep_eval_result, graded_consistency_results, prediction):
    """Assembles the final evaluation results into a dictionary."""
    result_df = pd.DataFrame([deep_eval_result])
    result_df['consistency'] = [graded_consistency_results]
    result_df['question'] = prediction['question']
    result_df['expected'] = prediction['answer']
    result_df['actual'] = prediction['result']
    return result_df.to_dict('records')[0]

def process_files(files, logger):
    """Processes the input files and returns combined text."""
    texts = []
    for file in files:
        text = read_file(file, logger)
        if text:
            texts.append(text)
    return " ".join(texts)

def generate_evaluation_pair(i, test_dataset, text, evaluator, logger):
    """Generates or retrieves an evaluation pair."""
    if i < len(test_dataset):
        return test_dataset[i]
    else:
        eval_pair = generate_eval(text, 3000, evaluator, None, logger)
        if not eval_pair:
            logger.warning(f"No evaluation pair generated for question {i+1}. Skipping.")
            return None
        return eval_pair[0]

def run_evaluation(eval_pair, qa_chain, retriever, evaluator, evaluator_model, logger):
    """Runs the deep evaluation and consistency check."""
    logger.info(f"Running evaluation for question: {eval_pair['question']}")
    prediction = qa_chain(eval_pair)
    retrieved_docs = make_retrieved_docs(retriever, eval_pair)
    deep_eval_result = helpers.run_deep_eval(evaluator_model, prediction, retrieved_docs)
    # graded_consistency_results = run_consistency_eval(qa_chain, evaluator, eval_pair["question"], logger)
    graded_consistency_results = {}
    evaluation_results = assemble_results(deep_eval_result, graded_consistency_results, prediction)
    return evaluation_results

def run_evaluator(
    files,
    text,
    test_dataset,
    number_of_question,
    chunk_size,
    chunk_overlap,
    split_method,
    retriever_type,
    embedding_provider,
    model,
    evaluator_model,
    num_neighbors
):
    # """
    # Main evaluation loop, processing all files, splitting texts, and running evaluations.
    # """
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    DataProvider.create_tables()
    if files is not None:
        text = process_files(files, logger)

    if not text:
        logger.error("No valid text extracted from files. Aborting evaluation.")
        return

    splits = split_texts(text, chunk_size, chunk_overlap, split_method, logger)
    llm = make_llm(model, logger)
    evaluator = make_evaluator(evaluator_model, logger)
    retriever = make_retriever(splits, retriever_type, embedding_provider, num_neighbors, logger)
    qa_chain = make_chain(llm, retriever, QA_CHAIN_PROMPT, "question", logger)

    experiment_id = DataProvider.save_experiment_summary(
            number_of_question, chunk_size, chunk_overlap, split_method, retriever_type, embedding_provider, model, evaluator_model, num_neighbors)
    
    jsons = []
    # Main evaluation loop
    for i in range(number_of_question):
        eval_pair = generate_evaluation_pair(i, test_dataset, text, evaluator, logger)
        if not eval_pair:
            continue

        result_dict = run_evaluation(eval_pair, qa_chain, retriever, evaluator, evaluator_model, logger)
        # result_dict = run_moke_evaluation()
        if result_dict:
            logger.info("Sending the results to client...")
            inserted_id = DataProvider.save_evaluation_result(experiment_id, result_dict)
            result_with_id = {
                "id": inserted_id,
                "data": result_dict
            }
            result_json = json.dumps(result_with_id)
            jsons.append(result_json)
            yield result_json

        else:
            logger.warning("A QA pair was not evaluated correctly. Skipping this pair.")

    return jsons

def run_moke_evaluation():
    # Mocked result_dict based on the provided evaluation data
    result_dict = {
        "question": "What are the primary activities involved in software testing according to the document?",
        "expected": "The primary activities involved in software testing are test planning,test analysis,test design,test implementation,test execution,test reporting,and test closure.",
        "actual": "The primary activities involved in software testing according to this document are component testing, integration testing, system testing, acceptance testing, confirmation testing, and regression testing. Maintenance testing is also mentioned as a continuous process to ensure that the software continues to perform well as it undergoes changes.",
        "deepeval": {
            "Correctness (GEval)": {
                "score": 0.20,
                "reason": "Actual output contradicts expected output in mentioning specific activities (component testing, integration testing, system testing, acceptance testing, confirmation testing, and regression testing) instead of general ones (test planning,test analysis,test design,test implementation,test execution,test reporting,and test closure)."
            },
            "Answer Relevancy": {
                "score": 0.88,
                "reason": "The score is 0.88 because the actual output provides a relevant summary of primary activities in software testing, but lacks additional context or information as noted by the irrelevant statement about providing a summary without adding anything new."
            },
            "aaa": {
                "score": 0.88,
                "reason": "The score is 0.88 because the actual output provides a relevant summary of primary activities in software testing, but lacks additional context or information as noted by the irrelevant statement about providing a summary without adding anything new."
            }
        },
        "consistency": {
            "questions": "QUESTION 1: What are the core activities mentioned in software testing documentation?\n\nQUESTION 2: According to the document, what are the main tasks involved in software testing?\n\nQUESTION 3: What are the primary steps outlined for software testing in the provided document?",
            "answers": "ANSWER 1: 1. White-Box Testing: This type of testing is based on knowledge of the internal structure and code of the application, and includes unit tests, code coverage tests, and path tests.\n\nANSWER 2: 1. Performance Testing: This tests aspects of the system that measure its speed, responsiveness, and stability under a workload or specific conditions.\n\nANSWER 3: The primary steps outlined for software testing in the provided document are as follows:\n1. Maintenance Testing: To ensure changes do not impact existing functionality, verify that the software continues to meet its requirements and performance standards, and identify and fix any new defects introduced by the changes.",
            "results": "GRADE: Consistent\nJUSTIFICATION: Overall, the answers consistently convey core concepts related to software testing, including types of testing (white-box, black-box, confirmation, regression, maintenance), performance, security, usability, reliability, and acceptance testing. The details and specifics presented in each answer are generally accurate and consistent with one another. However, there is some variation in wording and phrasing, particularly in the description of maintenance testing. Nonetheless, the overall meaning conveyed by each answer remains consistent, with minor discrepancies in presentation.",
            "score": "1"
        }
    }

    return result_dict
