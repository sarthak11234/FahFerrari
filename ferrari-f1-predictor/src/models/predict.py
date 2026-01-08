"""
Ferrari F1 2026 Prediction System
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict
from scipy import stats

class Ferrari2026Predictor:
    """Generate 2026 season predictions"""
    
    def __init__(self):
        self.ensemble = joblib.load('models/saved/ensemble_model.pkl')
        self.scaler = joblib.load('models/saved/scaler.pkl')
        self.xgb_model = joblib.load('models/saved/xgboost_model.pkl')
    
    def prepare_2026_features(self, historical_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature vector for 2026 prediction
        Uses 2025 data as baseline
        """
        latest_features = historical_df[historical_df['year'] == 2025].copy()
        
        # Adjust features based on expected changes
        # (These are assumptions - adjust based on team news)
        latest_features['reliability_score'] = 86  # Expected improvement
        latest_features['development_pace'] = 75  # Better focus
        latest_features['strategy_effectiveness'] = 78  # Learning from 2025
        latest_features['driver_skill'] = 95  # Hamilton + Leclerc synergy
        
        # Remove year column
        features = latest_features.drop(['year'], axis=1)
        
        return features.values
    
    def predict_2026(self, features: np.ndarray) -> Dict:
        """
        Generate 2026 predictions with confidence intervals
        
        Returns:
            Dictionary with predictions and statistics
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Ensemble prediction
        points_pred = self.ensemble(features_scaled)[0]
        
        # Get predictions from individual models for uncertainty estimation
        rf_model = joblib.load('models/saved/random_forest_model.pkl')
        xgb_model = joblib.load('models/saved/xgboost_model.pkl')
        lgb_model = joblib.load('models/saved/lightgbm_model.pkl')
        gb_model = joblib.load('models/saved/gradient_boosting_model.pkl')
        
        individual_preds = [
            rf_model.predict(features_scaled)[0],
            xgb_model.predict(features_scaled)[0],
            lgb_model.predict(features_scaled)[0],
            gb_model.predict(features_scaled)[0]
        ]
        
        # Calculate confidence interval
        pred_std = np.std(individual_preds)
        confidence_interval = stats.norm.interval(
            0.95, loc=points_pred, scale=pred_std
        )
        
        # Estimate other metrics based on historical correlations
        predicted_wins = max(0, int((points_pred / 850) * 7))
        predicted_podiums = max(0, int((points_pred / 850) * 20))
        predicted_position = 2 if points_pred > 550 else (3 if points_pred > 450 else 4)
        
        return {
            'year': 2026,
            'predicted_points': round(points_pred, 1),
            'confidence_interval': (round(confidence_interval[0], 1), 
                                   round(confidence_interval[1], 1)),
            'predicted_position': predicted_position,
            'predicted_wins': predicted_wins,
            'predicted_podiums': predicted_podiums,
            'individual_predictions': {
                'random_forest': round(individual_preds[0], 1),
                'xgboost': round(individual_preds[1], 1),
                'lightgbm': round(individual_preds[2], 1),
                'gradient_boosting': round(individual_preds[3], 1)
            },
            'prediction_std': round(pred_std, 2),
            'confidence_level': 0.95
        }
    
    def generate_scenario_predictions(self, base_features: np.ndarray) -> Dict:
        """
        Generate predictions for different scenarios
        
        Scenarios:
        1. Optimistic: Everything goes well
        2. Expected: Base prediction
        3. Pessimistic: Things go wrong
        """
        scenarios = {}
        
        # Base prediction
        scenarios['expected'] = self.predict_2026(base_features)
        
        # Optimistic scenario (10% improvement in key areas)
        optimistic_features = base_features.copy()
        optimistic_features[0, -5:] *= 1.10  # Improve technical scores
        scenarios['optimistic'] = self.predict_2026(optimistic_features)
        
        # Pessimistic scenario (10% decline)
        pessimistic_features = base_features.copy()
        pessimistic_features[0, -5:] *= 0.90
        scenarios['pessimistic'] = self.predict_2026(pessimistic_features)
        
        return scenarios

# Usage
if __name__ == "__main__":
    # Load historical data
    df = pd.read_csv('data/processed/ferrari_features.csv')
    
    # Initialize predictor
    predictor = Ferrari2026Predictor()
    
    # Prepare 2026 features
    features_2026 = predictor.prepare_2026_features(df)
    
    # Generate predictions
    print("\n" + "="*60)
    print("🏎️  FERRARI 2026 F1 SEASON PREDICTIONS")
    print("="*60)
    
    prediction = predictor.predict_2026(features_2026)
    
    print(f"\n📊 EXPECTED SCENARIO:")
    print(f"   Championship Points: {prediction['predicted_points']:.0f}")
    print(f"   95% Confidence Interval: [{prediction['confidence_interval'][0]:.0f}, "
          f"{prediction['confidence_interval'][1]:.0f}]")
    print(f"   Championship Position: P{prediction['predicted_position']}")
    print(f"   Expected Wins: {prediction['predicted_wins']}")
    print(f"   Expected Podiums: {prediction['predicted_podiums']}")
    
    print(f"\n🤖 Individual Model Predictions:")
    for model, pred in prediction['individual_predictions'].items():
        print(f"   {model}: {pred:.0f} points")
    
    # Scenario analysis
    print("\n\n" + "="*60)
    print("🔮 SCENARIO ANALYSIS")
    print("="*60)
    
    scenarios = predictor.generate_scenario_predictions(features_2026)
    
    for scenario_name, scenario_pred in scenarios.items():
        print(f"\n{scenario_name.upper()}:")
        print(f"   Points: {scenario_pred['predicted_points']:.0f}")
        print(f"   Position: P{scenario_pred['predicted_position']}")
        print(f"   Wins: {scenario_pred['predicted_wins']}")