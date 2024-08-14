import os
import io
import random
import itertools
import pypdf
from json import JSONDecodeError
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import EdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.chains import QAGenerationChain
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA, SELF_CHECK_QA_CHAIN_PROMPT
EDENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmFjNmJiZWUtZTdiMy00NTRkLTk4MmUtY2Y0ODQ2NGMzMzFkIiwidHlwZSI6ImFwaV90b2tlbiJ9.8YU_EHliBgJ2EqNBGNddA7fbTqkTSKF4lFHDz772xpQ"
OLLAMA_SERVER_URL="https://0778-34-83-80-70.ngrok-free.app"
def make_llm():
    # llm = EdenAI(edenai_api_key=EDENAI_API_KEY, feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
    llm = ChatOllama(model="llama3.1", base_url=OLLAMA_SERVER_URL)
    return llm

def make_grader():
    # garder_llm = EdenAI(edenai_api_key=EDENAI_API_KEY, feature="text", provider="openai", model="gpt-3.5-turbo-instruct")
    garder_llm = ChatOllama(model="llama3.1", base_url=OLLAMA_SERVER_URL)
    return garder_llm

from langchain_community.embeddings import OllamaEmbeddings

def make_retriever(splits):
    embd = OllamaEmbeddings(model="llama3.1")
    embd.base_url=OLLAMA_SERVER_URL
    vectorstore = FAISS.from_texts(splits, embd)
    retriever = vectorstore.as_retriever(k=3)
    return retriever

def make_chain(llm, retriever):
    chain_type_kwargs = {"prompt": SELF_CHECK_QA_CHAIN_PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            chain_type_kwargs=chain_type_kwargs,
                                            input_key="question")
    return qa_chain

def split_texts(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    splits = text_splitter.split_text(text)
    return splits

def generate_eval(text, chunk, grader_llm, prompt):
    print("`Generating eval QA pair ...`")
    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk)
    sub_sequence = text[starting_index:starting_index+chunk]
    chain = QAGenerationChain.from_llm(grader_llm, prompt)
    eval_set = []
    awaiting_answer = True
    while awaiting_answer:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            awaiting_answer = False
        except JSONDecodeError:
            print("Error on question")
            starting_index = random.randint(0, num_of_chars-chunk)
            sub_sequence = text[starting_index:starting_index+chunk]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


# texts = []
# fnames = []

# pdf_path = "C:\\Users\\levan\\Downloads\\ale-omniswitch-milestone-plugin-user-guide-v3-0-rev-a-en.pdf"
# texts = []
# fnames = []

# # Open and read the PDF file
# with open(pdf_path, 'rb') as file:
#     contents = file.read()
#     if file.name.lower().endswith('.pdf'):
#         pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#         texts.append(text)
#         fnames.append(pdf_path)  # Add the filename or path
#     else:
#         print("Unsupported file type for file: {}".format(pdf_path))

# full_text = " ".join(texts)

# num_eval_questions = 1
# test_dataset = []

# from text_utils import QA_GENERATION_CHAIN_PROMPT

# for i in range(num_eval_questions):
#     if i < len(test_dataset):
#         eval_pair = test_dataset[i]
#     else:
#         eval_pair = generate_eval(text, 3000, make_llm(), QA_GENERATION_CHAIN_PROMPT)
#         if len(eval_pair) == 0:
#             # Error in eval generation
#             continue
#         else:
#             # This returns a list, so we unpack to dict
#             eval_pair = eval_pair[0]

# # Given dictionary
#     eval_pair = {
#         'question': [
#             'Only PoE OmniSwitches can be added to the ALOM plugin.', 
#             'The ALOM plugin is compatible with only PoE OmniSwitches.',
#             'PoE OmniSwitches are required for adding them to the ALOM plugin.'
#         ],
#         'answer': True
#     }

# # Convert to the desired format
#     parsed_format = [{ 'question': question, 'answer': eval_pair['answer'] } for question in eval_pair['question']]  

#     gt_dataset = parsed_format
#     predictions = []


#     splits = split_texts(text)
#     llm = make_llm()
#     retriever = make_retriever(splits)

#     qa_chain = make_chain(llm, retriever)

#     for set in gt_dataset:
#         predictions.append(qa_chain(set))

#     print(predictions)

#     # Check if all 'answer' values are True
#     all_true = all(is_true(entry['answer']) for entry in predictions)

#     print(all_true)  # This will print True if all answers are True, otherwise False


# List of predictions
predictions = [
    {'question': "Using the universal driver for some vendor's cameras in Milestone xProtect VMS can result in an incorrect MAC address being associated with the camera.", 'answer': 'True', 'result': ' True'},
    {'question': "The use of a universal driver for certain vendors' cameras in Milestone xProtect VMS may lead to incorrect MAC address association.", 'answer': 'True', 'result': ' True'},
    {'question': 'Associating a camera with an incorrect MAC address using a universal driver in Milestone xProtect VMS is possible.', 'answer': 'True', 'result': ' True'}
]



