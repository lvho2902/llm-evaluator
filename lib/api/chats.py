import requests

WEBUI_API_BASE_URL = 'http://localhost:8000/api/v1'

def create_new_chat(token: str, chat: dict):
    url = f"{WEBUI_API_BASE_URL}/chats/new"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        'chat': chat
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err  # Rethrow the exception with the error detail
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def get_chat_by_id(token: str, id: str):
    url = f"{WEBUI_API_BASE_URL}/chats/{id}"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
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
