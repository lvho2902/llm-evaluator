from api.auths import user_sign_in
from api.files import upload_file
from api.rag import process_doc_to_vector_db
from api.chats import create_new_chat
from api.ollama import generate_chat_completion

import json

signin_response = user_sign_in("bot@tma.com.vn", "5az5yd9b")
token = signin_response["token"]
upload_file_reponse = upload_file(token, "docs\istqb\ISTQB_CTFL_Syllabus-v4.0.txt")
file_id = upload_file_reponse["id"]
process_doc_reponse = process_doc_to_vector_db(token, file_id)
collection_name = process_doc_reponse["collection_name"]

models = ["llama3:latest"]
messages = []
files = [{"id": file_id, "type": "file", "collection_name": collection_name}]

create_new_chat_reponse = create_new_chat(
    token=token,
    chat={
        "id": "",
        "title": "New Chat",
        "models": models,
        "message": messages
    }
)
chat_id = create_new_chat_reponse["id"]

messages.append({
    "role":"user",
    "content": "What are the primary activities involved in software testing according to the document?"
})

response = generate_chat_completion(
    token=token,
    body={
        "stream": True,
        "model": models[0],
        "messages": messages,
        "files" : files
    }
)

print(response["citations"][0]["document"])