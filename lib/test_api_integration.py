import pytest
from unittest.mock import patch, MagicMock

from api.auths import user_sign_in
from api.files import upload_file
from api.rag import process_doc_to_vector_db
from api.chats import create_new_chat
from api.ollama import generate_chat_completion

# Assuming the functions are in a module named `api_module`
# from api_module import user_sign_in, upload_file, process_doc_to_vector_db, create_new_chat, generate_chat_completion

@pytest.fixture
def mock_api_responses(mocker):
    mock_sign_in = mocker.patch('api.auths.user_sign_in', return_value={"token": "dummy_token"})
    mock_upload_file = mocker.patch('api.files.upload_file', return_value={"id": "file_id_123"})
    mock_process_doc = mocker.patch('api.rag.process_doc_to_vector_db', return_value={"collection_name": "collection_abc"})
    mock_create_chat = mocker.patch('api.chats.create_new_chat', return_value={"id": "chat_id_456"})
    mock_generate_completion = mocker.patch('api.ollama.generate_chat_completion', return_value={"citations": [{"document": "Some document content"}]})
    return mock_sign_in, mock_upload_file, mock_process_doc, mock_create_chat, mock_generate_completion

def test_user_sign_in(mock_api_responses):
    mock_sign_in, _, _, _, _ = mock_api_responses
    response = user_sign_in("bot@tma.com.vn", "5az5yd9b")
    assert response == {"token": "dummy_token"}
    mock_sign_in.assert_called_once_with("bot@tma.com.vn", "5az5yd9b")

def test_upload_file(mock_api_responses):
    _, mock_upload_file, _, _, _ = mock_api_responses
    response = upload_file("dummy_token", "docs\\istqb\\ISTQB_CTFL_Syllabus-v4.0.txt")
    assert response == {"id": "file_id_123"}
    mock_upload_file.assert_called_once_with("dummy_token", "docs\\istqb\\ISTQB_CTFL_Syllabus-v4.0.txt")

def test_process_doc_to_vector_db(mock_api_responses):
    _, _, mock_process_doc, _, _ = mock_api_responses
    response = process_doc_to_vector_db("dummy_token", "file_id_123")
    assert response == {"collection_name": "collection_abc"}
    mock_process_doc.assert_called_once_with("dummy_token", "file_id_123")

def test_create_new_chat(mock_api_responses):
    _, _, _, mock_create_chat, _ = mock_api_responses
    response = create_new_chat(
        token="dummy_token",
        chat={
            "id": "",
            "title": "New Chat",
            "models": ["llama3:latest"],
            "message": []
        }
    )
    assert response == {"id": "chat_id_456"}
    mock_create_chat.assert_called_once_with(
        token="dummy_token",
        chat={
            "id": "",
            "title": "New Chat",
            "models": ["llama3:latest"],
            "message": []
        }
    )

def test_generate_chat_completion(mock_api_responses):
    _, _, _, _, mock_generate_completion = mock_api_responses
    response = generate_chat_completion(
        token="dummy_token",
        body={
            "stream": True,
            "model": "llama3:latest",
            "messages": [{"role": "user", "content": "What are the primary activities involved in software testing according to the document?"}],
            "files": [{"id": "file_id_123", "type": "file", "collection_name": "collection_abc"}]
        }
    )
    assert response["citations"][0]["document"] == "Some document content"
    mock_generate_completion.assert_called_once_with(
        token="dummy_token",
        body={
            "stream": True,
            "model": "llama3:latest",
            "messages": [{"role": "user", "content": "What are the primary activities involved in software testing according to the document?"}],
            "files": [{"id": "file_id_123", "type": "file", "collection_name": "collection_abc"}]
        }
    )

