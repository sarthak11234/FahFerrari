import requests
import json
import os

def verify_chat():
    print("Testing Chatbot API...")
    app_url = os.environ.get('APP_URL', 'http://localhost:5000').rstrip('/')
    url = f"{app_url}/api/chat" if not app_url.endswith('/api') else f"{app_url}/chat"
    payload = {"query": "tell me about 2026 season predictions"}
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    verify_chat()
