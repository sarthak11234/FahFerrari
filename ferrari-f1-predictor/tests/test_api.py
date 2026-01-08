"""
API integration tests
"""

import pytest
from api.app import app
import json

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_historical_data(client):
    """Test historical data endpoint"""
    response = client.get('/api/historical/2025')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_points' in data

def test_predictions(client):
    """Test predictions endpoint"""
    response = client.get('/api/predictions/2026')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predicted_points' in data

def test_chat(client):
    """Test chatbot endpoint"""
    response = client.post('/api/chat',
                          json={'query': 'How did Ferrari do in 2025?'},
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data