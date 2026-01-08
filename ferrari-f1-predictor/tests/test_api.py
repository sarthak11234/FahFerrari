"""
API integration tests
"""

import pytest
from api.app import app
import json

import pandas as pd
from unittest.mock import MagicMock
import api.app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    
    # Mock data to handle missing models/data on startup
    api.app.historical_data = pd.DataFrame([{
        'year': 2025,
        'total_points': 398,
        'wins': 0,
        'podiums': 10
    }])
    
    api.app.predictions_2026 = {
        'predicted_points': 520,
        'confidence_interval': (480, 560),
        'predicted_position': 3
    }
    
    mock_bot = MagicMock()
    mock_bot.generate_response.return_value = "Ferrari performed well in 2025 considering the challenges."
    api.app.chatbot = mock_bot

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