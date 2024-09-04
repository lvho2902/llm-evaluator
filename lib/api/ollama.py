import requests, json

OLLAMA_API_BASE_URL = 'http://localhost:8000/ollama'

def generate_text_completion(token='', model='', text=''):
    """
    Generate text completion using the specified model and prompt text.

    :param token: Bearer token for authorization
    :param model: The model to use for text completion
    :param text: The prompt text to generate completion for
    :return: Response object or raises an error
    """
    url = f'{OLLAMA_API_BASE_URL}/api/generate'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    payload = {
        'model': model,
        'prompt': text,
        'stream': True
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response
    except requests.RequestException as err:
        raise SystemExit(err)

def generate_chat_completion(token='', body=None):
    """
    Generate chat completion with the provided body.

    :param token: Bearer token for authorization
    :param body: The body of the chat request
    :return: Response object and an AbortController-like object (here, just a placeholder)
    """
    url = f'{OLLAMA_API_BASE_URL}/api/chat'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    try:
        s = requests.Session()
        with s.post(url, headers=headers, stream=True, json=body) as resp:
            return process_response(resp.iter_lines())

    except requests.RequestException as err:
        raise SystemExit(err)
    

def process_response(response):
    response_message = {
        'done': False,
        'content': '',
        'citations': None
        # 'statusHistory': [],
        # 'error': None,
        # 'context': None,
        # 'info': {}
    }

    for line in response:
        try:
            line = line.decode('utf-8')
            if line:
                data = json.loads(line)
                if 'citations' in data:
                    response_message['citations'] = data['citations']
                    continue
                if 'detail' in data:
                    raise Exception(data['detail'])
                if not data.get('done', True):
                    response_message['content'] += data['message']['content']
                else:
                    response_message['done'] = True
                    # response_message['context'] = data.get('context', None)
                    # response_message['info'] = {
                        # 'total_duration': data.get('total_duration', 0),
                        # 'load_duration': data.get('load_duration', 0),
                        # 'sample_count': data.get('sample_count', 0),
                        # 'sample_duration': data.get('sample_duration', 0),
                        # 'prompt_eval_count': data.get('prompt_eval_count', 0),
                        # 'prompt_eval_duration': data.get('prompt_eval_duration', 0),
                        # 'eval_count': data.get('eval_count', 0),
                        # 'eval_duration': data.get('eval_duration', 0),
                    # }
                    break
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return response_message