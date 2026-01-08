import requests
import json

base_url = 'http://localhost:5000/api'

def check_endpoint(endpoint):
    print(f"\nChecking {endpoint}...")
    try:
        response = requests.get(f"{base_url}{endpoint}")
        print(f"Status: {response.status_code}")
        try:
            data = response.json()
            print("Data sample:", str(data)[:200])
        except:
            print("Response text:", response.text[:200])
            
    except Exception as e:
        print(f"Error: {e}")

check_endpoint('/health')
check_endpoint('/historical')
check_endpoint('/predictions/2026')
