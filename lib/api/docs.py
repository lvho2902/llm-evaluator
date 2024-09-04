import requests
import json
from typing import Optional, Dict, Any

WEBUI_API_BASE_URL = 'http://localhost:8000/api/v1'

def create_new_doc(
    token: str,
    collection_name: str,
    filename: str,
    name: str,
    title: str,
    content: Optional[Dict[str, Any]] = None
):
    url = f"{WEBUI_API_BASE_URL}/documents/create"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        'collection_name': collection_name,
        'filename': filename,
        'name': name,
        'title': title,
        **({'content': json.dumps(content)} if content else {})
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err  # Rethrow the exception with the error detail
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def get_docs(token: str = ''):
    url = f"{WEBUI_API_BASE_URL}/documents/"
    headers = {
        'Accept': 'application/json',
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

def get_doc_by_name(token: str, name: str):
    url = f"{WEBUI_API_BASE_URL}/documents/docs"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    params = {
        'name': name
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def update_doc_by_name(token: str, name: str, form: Dict[str, str]):
    url = f"{WEBUI_API_BASE_URL}/documents/doc/update"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    params = {
        'name': name
    }
    try:
        response = requests.post(url, headers=headers, params=params, json=form)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
