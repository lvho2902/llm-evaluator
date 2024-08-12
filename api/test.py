import os
import io
import random
import itertools
import pypdf
from json import JSONDecodeError
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import EdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.chains import QAGenerationChain
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA

EDENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmFjNmJiZWUtZTdiMy00NTRkLTk4MmUtY2Y0ODQ2NGMzMzFkIiwidHlwZSI6ImFwaV90b2tlbiJ9.8YU_EHliBgJ2EqNBGNddA7fbTqkTSKF4lFHDz772xpQ"

def make_llm():
    llm = EdenAI(edenai_api_key=EDENAI_API_KEY, feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
    return llm

def make_grader():
    garder_llm = EdenAI(edenai_api_key=EDENAI_API_KEY, feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
    return garder_llm

def make_retriever(splits, num_neighbors):
    embd = EdenAiEmbeddings(edenai_api_key=os.getenv('EDENAI_API_KEY'), provider="openai")
    vectorstore = FAISS.from_texts(splits, embd)
    retriever = vectorstore.as_retriever(k=num_neighbors)
    return retriever

def make_chain(llm, retriever):
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            chain_type_kwargs=chain_type_kwargs,
                                            input_key="question")
    return qa_chain

def generate_eval(text, chunk, grader_llm, prompt):
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk)
    sub_sequence = text[starting_index:starting_index+chunk]
    chain = QAGenerationChain.from_llm(grader_llm, prompt)
    eval_set = []
    # Catch any QA generation errors and re-try until QA pair is generated
    awaiting_answer = True
    while awaiting_answer:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            awaiting_answer = False
        except JSONDecodeError:
            starting_index = random.randint(0, num_of_chars-chunk)
            sub_sequence = text[starting_index:starting_index+chunk]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair

texts = []
fnames = []

pdf_path = "C:\\Users\\levan\\Downloads\\ale-omniswitch-milestone-plugin-user-guide-v3-0-rev-a-en.pdf"
texts = []
fnames = []

# Open and read the PDF file
with open(pdf_path, 'rb') as file:
    contents = file.read()
    if file.name.lower().endswith('.pdf'):
        pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
        fnames.append(pdf_path)  # Add the filename or path
    else:
        print("Unsupported file type for file: {}".format(pdf_path))

full_text = " ".join(texts)

num_eval_questions = 1
test_dataset = []

from text_utils import QA_GENERATION_CHAIN_PROMPT

for i in range(num_eval_questions):
    if i < len(test_dataset):
        eval_pair = test_dataset[i]
    else:
        eval_pair = generate_eval(text, 3000, make_llm(), QA_GENERATION_CHAIN_PROMPT)
        if len(eval_pair) == 0:
            # Error in eval generation
            continue
        else:
            # This returns a list, so we unpack to dict
            eval_pair = eval_pair[0]

print(eval_pair)