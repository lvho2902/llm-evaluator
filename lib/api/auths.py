import requests
import os

WEBUI_API_BASE_URL = 'http://localhost:8000/api/v1'

def get_session_user(token: str):
    url = f"{WEBUI_API_BASE_URL}/auths/"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers, cookies={'token': token})
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err  # Rethrow the exception with the error detail
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err

def user_sign_in(email: str, password: str):
    url = f"{WEBUI_API_BASE_URL}/auths/signin"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'email': email,
        'password': password
    }
    try:
        response = requests.post(url, json=data, headers=headers, cookies={'token': ''})
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        error_detail = response.json().get('detail', 'Unknown error')
        print(http_err, error_detail)
        raise http_err
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise err