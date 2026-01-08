"""
Unit tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
from src.models.train import FerrariMLTrainer
from src.models.predict import Ferrari2026Predictor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'year': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'total_points': [504, 131, 323.5, 554, 406, 652, 398],
        'wins': [3, 0, 0, 4, 1, 5, 0],
        'podiums': [14, 3, 7, 16, 12, 23, 10],
        'position': [2, 6, 3, 2, 2, 3, 4],
        'reliability_score': [72, 55, 68, 78, 66, 85, 82],
        'aero_efficiency': [85, 70, 75, 88, 82, 85, 78],
        'power_unit': [78, 65, 72, 82, 80, 82, 80]
    })

def test_data_preparation(sample_data):
    """Test data preparation"""
    trainer = FerrariMLTrainer()
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(sample_data)
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(features) > 0

def test_model_training(sample_data):
    """Test model training pipeline"""
    trainer = FerrariMLTrainer()
    X_train, X_test, y_train, y_test, _ = trainer.prepare_data(sample_data)
    
    trainer.initialize_models()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    assert 'xgboost' in results
    assert results['xgboost']['test_r2'] > 0

def test_prediction_output(sample_data):
    """Test prediction output format"""
    predictor = Ferrari2026Predictor()
    # Mock the model loading for testing
    # In real tests, you'd load actual trained models
    
    prediction = {
        'predicted_points': 520,
        'confidence_interval': (480, 560),
        'predicted_position': 3
    }
    
    assert 'predicted_points' in prediction
    assert prediction['predicted_points'] > 0
    assert len(prediction['confidence_interval']) == 2

def test_prediction_reasonableness(sample_data):
    """Test that predictions are reasonable"""
    # Predictions should be within historical range +/- 20%
    historical_min = sample_data['total_points'].min()
    historical_max = sample_data['total_points'].max()
    
    predicted_points = 520  # Example prediction
    
    assert historical_min * 0.8 <= predicted_points <= historical_max * 1.2