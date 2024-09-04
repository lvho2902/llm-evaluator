import requests
import json
from typing import Optional, Dict, Any

RAG_API_BASE_URL = 'http://localhost:8000/rag/api/v1'

def get_rag_config(token: str) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/config"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def update_rag_config(token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/config/update"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def get_rag_template(token: str) -> str:
    url = f"{RAG_API_BASE_URL}/template"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get('template', '')
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def get_query_settings(token: str) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/query/settings"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def update_query_settings(token: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/query/settings/update"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.post(url, headers=headers, json=settings)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def process_doc_to_vector_db(token: str, file_id: str) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/process/doc"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        'file_id': file_id
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def upload_doc_to_vector_db(token: str, collection_name: str, file_path: str) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/doc"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    files = {
        'file': open(file_path, 'rb')
    }
    data = {
        'collection_name': collection_name
    }
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
    finally:
        files['file'].close()

def query_doc(token: str, collection_name: str, query: str, k: Optional[int] = None) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/query/doc"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        'collection_name': collection_name,
        'query': query,
        'k': k
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def query_collection(token: str, collection_names: str, query: str, k: Optional[int] = None) -> Dict[str, Any]:
    url = f"{RAG_API_BASE_URL}/query/collection"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        'collection_names': collection_names,
        'query': query,
        'k': k
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
