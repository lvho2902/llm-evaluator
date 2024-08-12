"""
This is an API to support the LLM QA chain auto-evaluator. 
"""

import io
import os
from dotenv import load_dotenv
import sentry_sdk
import json
import time
import pypdf
import random
import logging
import itertools
import faiss
import pandas as pd
from typing import Dict, List
from json import JSONDecodeError
from langchain.llms import MosaicML
from langchain.llms import Replicate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.retrievers import SVMRetriever
from langchain.evaluation.qa import QAEvalChain
from langchain.retrievers import TFIDFRetriever
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import MosaicMLInstructorEmbeddings
from fastapi import FastAPI, File, UploadFile, Form
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.llms import EdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings

from statistics import mean
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt')
nltk.download('wordnet')

def generate_eval(text, chunk, grader_llm, logger):
    """
    Generate question answer pair from input text 
    @param text: text to generate eval set from
    @param chunk: chunk size to draw question from text
    @param logger: logger
    @return: dict with keys "question" and "answer"
    """

    logger.info("`Generating eval QA pair ...`")
    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk)
    sub_sequence = text[starting_index:starting_index+chunk]
    chain = QAGenerationChain.from_llm(grader_llm)
    eval_set = []
    # Catch any QA generation errors and re-try until QA pair is generated
    awaiting_answer = True
    while awaiting_answer:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            awaiting_answer = False
        except JSONDecodeError:
            logger.error("Error on question")
            starting_index = random.randint(0, num_of_chars-chunk)
            sub_sequence = text[starting_index:starting_index+chunk]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


def split_texts(text, chunk_size, overlap, split_method, logger):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size: charecters per split
    @param overlap: charecter overlap between splits
    @param split_method: method used to split text
    @param logger: logger
    @return: list of str splits
    """

    logger.info("`Splitting doc ...`")
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    return splits


def make_llm(model):
    """
    Make LLM
    @param model: LLM to use
    @return: LLM
    """

    if model in ("gpt-3.5-turbo", "gpt-4"):
        llm = ChatOpenAI(model_name=model, temperature=0)
    elif model == "eden-gpt-3.5-turbo-instruct":
        llm = EdenAI(edenai_api_key=os.getenv('EDENAI_API_KEY'), feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
        
    else: raise ValueError(f"Unknown model: {model}")
    return llm

def make_grader(garder):
    
    # Note: GPT-4 grader is advised by OAI
    if(garder == "ollama-mistral-7b"):
        garder_llm = ChatOllama(model="mistral", base_url=os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434"))
    elif(garder == "ollama-llama-3.1-8b"):
        lgarder_llmm = ChatOllama(model="llama3.1", base_url=os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434"))
    elif(garder == "eden-gpt-3.5-turbo-instruct"):
        garder_llm = EdenAI(edenai_api_key=os.getenv('EDENAI_API_KEY'), feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
    else:
        llm=ChatOpenAI(model_name="gpt-4", temperature=0)
    return garder_llm
    

def make_retriever(splits, retriever_type, embeddings, num_neighbors, logger):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @param logger: logger
    @return: retriever
    """

    logger.info("`Making retriever ...`")
    
    # Set embeddings
    if embeddings == "OpenAI":
        embd = OpenAIEmbeddings()
    elif embeddings == "EdenOpenAI":
        embd = EdenAiEmbeddings(edenai_api_key=os.getenv('EDENAI_API_KEY'), provider="openai")

    # Select retriever
    if retriever_type == "similarity-search":
        vectorstore = FAISS.from_texts(splits, embd)
        retriever = vectorstore.as_retriever(k=num_neighbors)
    elif retriever_type == "SVM":
        retriever = SVMRetriever.from_texts(splits, embd)
    elif retriever_type == "TF-IDF":
        retriever = TFIDFRetriever.from_texts(splits)
    return retriever

def make_chain(llm, retriever):

    """
    Make retrieval chain
    @param llm: llm
    @param retriever: retriever
    @return: QA chain
    """

    # Select prompt 
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

    # Select model 
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            chain_type_kwargs=chain_type_kwargs,
                                            input_key="question")
    return qa_chain


def grade_model_answer(predicted_dataset, predictions, grader_llm, grade_answer_prompt, logger):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    eval_chain = QAEvalChain.from_llm(llm = grader_llm, prompt=prompt)

    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(gt_dataset, predictions, grader_llm, grade_docs_prompt, logger):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    logger.info("`Grading relevance of retrieved docs ...`")
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    eval_chain = QAEvalChain.from_llm(llm = grader_llm,prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs

def calculate_bleu(reference, candidate):
    """
    Calculates the BLEU score between a reference and a candidate sentence.
    
    :param reference: The ground truth sentence (string).
    :param candidate: The generated sentence by the model (string).
    :return: BLEU score (float).
    """
    reference_tokens = []
    reference_tokens.append(reference.split())  # Tokenize reference
    candidate_tokens = candidate.split()  # Tokenize candidate
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    return bleu_score

def calculate_rouge(reference, candidate):
    """
    Calculates the ROUGE-1, ROUGE-2, and ROUGE-L scores between a reference and a candidate sentence.
    
    :param reference: The ground truth sentence (string).
    :param candidate: The generated sentence by the model (string).
    :return: Array containing ROUGE scores (rouge1, rouge2).
    """
    # Initialize CountVectorizer for unigram (ROUGE-1) and bigram (ROUGE-2) calculations
    vectorizer = CountVectorizer(ngram_range=(1, 2))

    # Fit and transform the texts
    vectors = vectorizer.fit_transform([reference, candidate])

    # Compute cosine similarity (as a proxy for ROUGE score)
    cosine_sim = cosine_similarity(vectors)

    # The diagonal contains the similarity of each text with itself, and the off-diagonal contains the similarity between the texts
    rouge_scores = cosine_sim[0, 1]
    return rouge_scores

def calculate_meteor_score(reference_text, candidate_text):
    """
    Calculate the METEOR score between a reference text and a candidate text.

    Parameters:
    reference_text (str): The reference text or ground truth sentence. This can be a single sentence or multiple sentences 
                          concatenated together, which will be tokenized into a list of words.
    candidate_text (str): The candidate text or the generated sentence that is being evaluated. This will be tokenized 
                           into a list of words for comparison against the reference text.

    Returns:
    float: The METEOR score between the reference text and the candidate text. The score is a float value representing 
           the similarity between the two texts, with higher values indicating better similarity.
    """
    # Tokenize the texts
    reference_tokens = nltk.word_tokenize(reference_text)
    candidate_tokens = nltk.word_tokenize(candidate_text)

    # METEOR score requires the reference tokens to be in a list of lists
    # Each list within the list represents a reference sentence
    reference_tokens_list = [reference_tokens]

    # Calculate METEOR score
    score = meteor_score(reference_tokens_list, candidate_tokens)

    return score

def evaluate_statistical_scores(predictions, logger):
    """
    Evaluates BLEU and ROUGE scores from a list of dictionaries.
    
    :param predictions: List of dictionaries containing 'question', 'answer' (reference), and 'result' (generated sentence).
    :param logger: Logger for logging information.
    :return: Tuple containing:
             - Average BLEU score (float).
             - Overall average of the ROUGE-1, ROUGE-2, and ROUGE-L F-measure scores (float).
    """
    logger.info("Evaluating statistical scores from predictions...")

    bleu_scores = []
    rouge_scores = []
    meteor_scores = []

    for item in predictions:
        reference = item['answer']
        candidate = item['result']
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
        
        # Calculate ROUGE scores
        rouge_score = calculate_rouge(reference, candidate)
        rouge_scores.append(rouge_score)

        # Calculate METEOR  scores
        meteor_score = calculate_meteor_score(reference, candidate)
        meteor_scores.append(meteor_score)


    # Calculate average BLEU score
    avg_bleu_score = mean(bleu_scores) if bleu_scores else 0
    avg_rouge_scores = mean(rouge_scores) if rouge_scores else 0
    avg_meteor_scores = mean(meteor_scores) if meteor_scores else 0

    return avg_bleu_score, avg_rouge_scores, avg_meteor_scores

def run_eval(chain, retriever, eval_qa_pair, grader_llm, grade_prompt, retriever_type, num_neighbors, text, logger):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @param text: full document text
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    logger.info("`Running eval ...`")
    predictions = []
    retrieved_docs = []
    gt_dataset = []
    latency = []

    # Get answer and log latency
    start_time = time.time()
    predictions.append(chain(eval_qa_pair))
    gt_dataset.append(eval_qa_pair)
    end_time = time.time()
    elapsed_time = end_time - start_time
    latency.append(elapsed_time)

    # Extract text from retrieved docs
    retrieved_doc_text = ""
    docs = retriever.get_relevant_documents(eval_qa_pair["question"])
    for i, doc in enumerate(docs):
        retrieved_doc_text += "Doc %s: " % str(i+1) + \
            doc.page_content + " "

    # Log
    retrieved = {"question": eval_qa_pair["question"],
                 "answer": eval_qa_pair["answer"], "result": retrieved_doc_text}
    retrieved_docs.append(retrieved)

    # Grade
    graded_answers = grade_model_answer(
        gt_dataset, predictions, grader_llm, grade_prompt, logger)
    graded_retrieval = grade_model_retrieval(
        gt_dataset, retrieved_docs, grader_llm, grade_prompt, logger)
    
    avg_bleu_score, avg_rouge_score, avg_meteor_scores = evaluate_statistical_scores(predictions, logger)

    return graded_answers, graded_retrieval, avg_bleu_score, avg_rouge_score, avg_meteor_scores, latency, predictions

load_dotenv()

if os.environ.get("ENVIRONMENT") != "development":
    sentry_sdk.init(
    dsn="https://065aa152c4de4e14af9f9e7335c8eae4@o4505106202820608.ingest.sentry.io/4505106207735808",
    traces_sample_rate=1.0,
    )

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://evaluator-ui.vercel.app/"
    "https://evaluator-ui.vercel.app"
    "evaluator-ui.vercel.app/"
    "evaluator-ui.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Auto Evaluator!"}


def run_evaluator(
    files,
    num_eval_questions,
    chunk_chars,
    overlap,
    split_method,
    retriever_type,
    embeddings,
    model_version,
    grader,
    grade_prompt,
    num_neighbors,
    test_dataset
):

    # Set up logging
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Read content of files
    texts = []
    fnames = []
    for file in files:
        logger.info("Reading file: {}".format(file.filename))
        contents = file.file.read()
        # PDF file
        if file.content_type == 'application/pdf':
            logger.info("File {} is a PDF".format(file.filename))
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)
            fnames.append(file.filename)
        # Text file
        elif file.content_type == 'text/plain':
            logger.info("File {} is a TXT".format(file.filename))
            texts.append(contents.decode())
            fnames.append(file.filename)
        else:
            logger.warning(
                "Unsupported file type for file: {}".format(file.filename))
    text = " ".join(texts)

    logger.info("Splitting texts")
    splits = split_texts(text, chunk_chars, overlap, split_method, logger)

    logger.info("Make LLM")
    llm = make_llm(model_version)

    grader_llm = make_grader(grader)

    logger.info("Make retriever")
    retriever = make_retriever(
        splits, retriever_type, embeddings, num_neighbors, llm, logger)

    logger.info("Make chain")
    qa_chain = make_chain(llm, retriever)

    for i in range(num_eval_questions):

        # Generate one question
        if i < len(test_dataset):
            eval_pair = test_dataset[i]
        else:
            eval_pair = generate_eval(text, 3000, grader_llm, logger)
            if len(eval_pair) == 0:
                # Error in eval generation
                continue
            else:
                # This returns a list, so we unpack to dict
                eval_pair = eval_pair[0]

        # Run eval
        graded_answers, graded_retrieval, avg_bleu_score, avg_rouge_score, avg_meteor_scores, latency, predictions = run_eval(
            qa_chain, retriever, eval_pair, grader_llm, grade_prompt, retriever_type, num_neighbors, text, logger)
        
        # Assemble output
        d = pd.DataFrame(predictions)

        if(grader == "openai"):
            d['answerScore'] = [g['text'] for g in graded_answers]
            d['retrievalScore'] = [g['text'] for g in graded_retrieval]
        else:
            d['answerScore'] = [g['results'] for g in graded_answers]
            d['retrievalScore'] = [g['results'] for g in graded_retrieval]

        d['latency'] = latency

        # Summary statistics
        d['answerScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                             'justification': text} for text in d['answerScore']]
        d['retrievalScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                                'justification': text} for text in d['retrievalScore']]

        d['avgBleuScore'] = f"{avg_bleu_score:.3f}"
        d['avgRougeScore'] = f"{avg_rouge_score:3f}"
        d['avgMeteorScores'] = f"{avg_meteor_scores:3f}"

        # Convert dataframe to dict
        d_dict = d.to_dict('records')
        if len(d_dict) == 1:
            yield json.dumps({"data":  d_dict[0]})
        else:
            logger.warn(
                "A QA pair was not evaluated correctly. Skipping this pair.")


@app.post("/evaluator-stream")
async def create_response(
    files: List[UploadFile] = File(...),
    num_eval_questions: int = Form(5),
    chunk_chars: int = Form(1000),
    overlap: int = Form(100),
    split_method: str = Form("RecursiveTextSplitter"),
    retriever_type: str = Form("similarity-search"),
    embeddings: str = Form("OpenAI"),
    model_version: str = Form("gpt-3.5-turbo"),
    grader: str = Form("openai"),
    grade_prompt: str = Form("Fast"),
    num_neighbors: int = Form(3),
    test_dataset: str = Form("[]"),
):
    test_dataset = json.loads(test_dataset)
    return EventSourceResponse(run_evaluator(files, num_eval_questions, chunk_chars,
                                             overlap, split_method, retriever_type, embeddings, model_version, grader, grade_prompt, num_neighbors, test_dataset), headers={"Content-Type": "text/event-stream", "Connection": "keep-alive", "Cache-Control": "no-cache"})
