import requests
import json

def verify_chat():
    print("Testing Chatbot API...")
    url = 'http://localhost:5000/api/chat'
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
