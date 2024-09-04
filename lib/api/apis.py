import requests
from typing import List, Dict, Optional, Union

WEBUI_BASE_URL = 'http://localhost:8000'

def get_models(token: str = '') -> List[Dict]:
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if token:
            headers['Authorization'] = f'Bearer {token}'

        response = requests.get(f'{WEBUI_BASE_URL}/api/models', headers=headers)
        response.raise_for_status()

        data = response.json()
        models = data.get('data', [])

        # Sort models
        models = sorted(
            (model for model in models if model),
            key=lambda m: (
                m.get('info', {}).get('meta', {}).get('position', float('inf')),
                m['name'].lower()
            )
        )

        return models
    except requests.RequestException as e:
        print(f"Error: {e}")
        raise

def chat_completed(token: str, body: Dict) -> Dict:
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        response = requests.post(f'{WEBUI_BASE_URL}/api/chat/completed', json=body, headers=headers)
        response.raise_for_status()

        return response.json()
    except requests.RequestException as e:
        error = e.response.json().get('detail', str(e))
        print(f"Error: {error}")
        raise

def generate_search_query(token: str, model: str, messages: List[Dict], prompt: str) -> str:
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        body = {
            'model': model,
            'messages': messages,
            'prompt': prompt
        }

        response = requests.post(f'{WEBUI_BASE_URL}/api/task/query/completions', json=body, headers=headers)
        response.raise_for_status()

        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', prompt)
        return content.replace('"', '').replace("'", '')
    except requests.RequestException as e:
        error = e.response.json().get('detail', str(e))
        print(f"Error: {error}")
        raise

def generate_moa_completion(token: str, model: str, prompt: str, responses: List[str]) -> Union[requests.Response, requests.Session]:
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        body = {
            'model': model,
            'prompt': prompt,
            'responses': responses,
            'stream': True
        }

        session = requests.Session()
        response = session.post(f'{WEBUI_BASE_URL}/api/task/moa/completions', json=body, headers=headers, stream=True)
        response.raise_for_status()

        return response, session
    except requests.RequestException as e:
        print(f"Error: {e}")
        raise
