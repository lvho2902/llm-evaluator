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
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, CONSISTENCY_QA_CHAIN_PROMPT, GRADE_ANSWERS_CONSISTENCY_PROMPT

OLLAMA_SERVER_URL="https://rides-airports-gotten-essex.trycloudflare.com"
def make_llm():
    llm = ChatOllama(model="llama3.1", base_url=OLLAMA_SERVER_URL)
    return llm

def make_grader():
    garder_llm = ChatOllama(model="llama3.1", base_url=OLLAMA_SERVER_URL)
    return garder_llm

from langchain_community.embeddings import OllamaEmbeddings

def make_retriever(splits):
    embd = OllamaEmbeddings(model="nomic-embed-text")
    embd.base_url=OLLAMA_SERVER_URL
    vectorstore = FAISS.from_texts(splits, embd)
    retriever = vectorstore.as_retriever(k=3)
    return retriever

def make_chain(llm, retriever):
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
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

predictions = [
    {
        # "result": "The main objective of the ALOM plugin is to provide a user-friendly interface within the Milestone VMS system, \
        # allowing operators to manage and monitor various components such as cameras, switches, and ports without requiring separate user interfaces. \
        # This includes functions like checking the status, power consumption, temperature, and CPU usage of OmniSwitches; \
        # managing PoE power allocations for each port; and controlling cameras by resetting them, pinging them, or checking their URLs. \
        # The ALOM plugin also offers information about each camera's vendor, IP address, MAC address, status, traffic, power consumption, \
        # maximum power available, PoE status, Locked status (LPS), priority level, and more.",
        "result" : "I don't know."
    },
    {
        "result": "The primary function of the ALOM (Advanced LAN Management) plugin is to provide detailed information about each port on an OmniSwitch, \
        ncluding the status of PoE (Power over Ethernet), the amount of PoE power usage and maximum available, the lock/unlock status of the port security (LPS), \
        and more. Additionally, it provides information about devices connected to specific ports, such as cameras, their IP addresses, MAC addresses, priority levels, \
        and statuses. The plugin also allows for various actions like resetting the camera, pinging the camera, removing the camera from the port, \
        setting power priority, performing a TDR cable test (on certain models), and executing switch-level actions such as writing memory or configuring SNMP traps."
    },
    {
        "result": "The core purpose of the ALOM (Advanced Live Operation Monitor) plugin in the Milestone VMS system is to provide remote management \
        capabilities for network devices such as cameras, switches, and other security equipment. This includes monitoring key parameters like power \
        consumption, temperature, CPU usage, and port status, as well as controlling features like PoE (Power over Ethernet) and Locked Port Status (LPS). \
        The ALOM plugin is accessible through the Smart Client interface, allowing operators to manage their surveillance system efficiently without \
        needing to connect to a separate user interface."
    }
]

from langchain.evaluation.qa import QAEvalChain


eval_chain = QAEvalChain.from_llm(llm = make_llm(), prompt=GRADE_ANSWERS_CONSISTENCY_PROMPT)

results = ""
for i in range(len(predictions)):
    results += "Answer " + str(i + 1) + ". " + predictions[i]['result'] + "\n\n"
inputs = [{"query": "", "answer": "", "result": results}]
graded_outputs = eval_chain.apply(inputs)


print(graded_outputs[0]['results'])

