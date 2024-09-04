import requests

WEBUI_API_BASE_URL = 'http://localhost:8000/api/v1'

def upload_file(token: str, file_path: str):
    url = f"{WEBUI_API_BASE_URL}/files/"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    files = {
        'file': open(file_path, 'rb')  # Open the file in binary read mode
    }
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err  # Rethrow the exception with the error detail
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
    finally:
        files['file'].close()  # Ensure the file is closed

def get_files(token: str = ''):
    url = f"{WEBUI_API_BASE_URL}/files/"
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

def get_file_by_id(token: str, file_id: str):
    url = f"{WEBUI_API_BASE_URL}/files/{file_id}"
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

def get_file_content_by_id(file_id: str):
    url = f"{WEBUI_API_BASE_URL}/files/{file_id}/content"
    headers = {
        'Accept': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content  # Returns the file content as bytes
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err
