"""
Ferrari F1 2026 Prediction System
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
from scipy import stats

class FerrariEnsemble:
    """Ensemble model wrapper for pickling"""
    def __init__(self, models: Dict, weights: Dict):
        self.models = models
        self.weights = weights
        
    def __call__(self, X):
        predictions = []
        for name, weight in self.weights.items():
            model = self.models[name]['model']
            pred = model.predict(X) * weight
            predictions.append(pred)
        return np.sum(predictions, axis=0)

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
        
        # Remove non-feature columns (must match training data)
        # Training drops: 'year', 'total_points', 'position'
        # latest_features already has 'year' dropped below? No, we should drop all at once.
        
        # We need to ensure we drop the exact same columns as training
        cols_to_drop = ['year', 'total_points', 'position']
        
        # Check if they exist before dropping to be safe
        existing_cols = [c for c in cols_to_drop if c in latest_features.columns]
        features = latest_features.drop(existing_cols, axis=1)
        
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
        
        # New: Detailed predictions
        race_by_race = self.simulate_race_by_race(points_pred, predicted_wins)
        driver_champ = self.predict_driver_championship(points_pred, predicted_wins)
        constructor_champ = self.predict_constructors_championship(points_pred)
        
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
            'confidence_level': 0.95,
            # NEW DATA
            'race_predictions': race_by_race,
            'driver_standings': driver_champ,
            'constructors_standings': constructor_champ
        }

    def simulate_race_by_race(self, total_points: float, total_wins: int) -> List[Dict]:
        """Simulate results for 24 races with variance"""
        races = [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Japanese GP", "Chinese GP",
            "Miami GP", "Emilia Romagna GP", "Monaco GP", "Canadian GP", "Spanish GP",
            "Austrian GP", "British GP", "Hungarian GP", "Belgian GP", "Dutch GP",
            "Italian GP", "Azerbaijan GP", "Singapore GP", "US GP", "Mexico City GP",
            "Sao Paulo GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP"
        ]
        
        avg_points = total_points / 24
        results = []
        wins_distributed = 0
        
        for i, race in enumerate(races):
            # Add some tailored variance (e.g., Ferrari strong in Monaco/Monza)
            multiplier = 1.0
            if "Monaco" in race or "Italian" in race or "Singapore" in race:
                multiplier = 1.5  # Ferrari tracks
            elif "British" in race or "Belgian" in race:
                multiplier = 0.8  # Historically tougher
                
            projected = avg_points * multiplier * np.random.normal(1, 0.2)
            
            # Assign wins
            is_win = False
            if wins_distributed < total_wins:
                # Higher chance to win at favorable tracks
                if multiplier > 1.2 and np.random.random() > 0.4:
                    projected = 25
                    is_win = True
                    wins_distributed += 1
            
            # Cap at 26 (win + fast lap)
            projected = min(26, max(0, projected))
            
            # Calculate position estimate based on standard F1 points
            points = int(projected)
            if points >= 25: pos = "P1"
            elif points >= 18: pos = "P2"
            elif points >= 15: pos = "P3"
            elif points >= 12: pos = "P4"
            elif points >= 10: pos = "P5"
            elif points >= 8: pos = "P6"
            elif points >= 6: pos = "P7"
            elif points >= 4: pos = "P8"
            elif points >= 2: pos = "P9"
            elif points >= 1: pos = "P10"
            else: pos = ">P10"
            
            # Performance score (percentage of max points)
            perf_score = int((projected / 26) * 100)
            
            results.append({
                'round': i + 1,
                'circuit': race,
                'predicted_points': points,
                'race_position': pos,
                'performance_score': f"{perf_score}%",
                'win_probability': f"{int(min(99, (projected/26)*90)) if projected > 0 else 0}%"
            })
            
        return results

    def predict_driver_championship(self, total_points: float, wins: int) -> Dict:
        """Split points between drivers based on synergy and history"""
        # Leclerc vs Hamilton split
        # Assumption: Leclerc slight edge due to team tenure, Hamilton consistency
        leclerc_share = 0.52
        hamilton_share = 0.48
        
        leclerc_pts = int(total_points * leclerc_share)
        hamilton_pts = int(total_points * hamilton_share)
        
        return {
            'leclerc': {
                'points': leclerc_pts,
                'wins': int(wins * 0.6),
                'podiums': int(wins * 0.6) + 8,
                'rank_est': 'P3'
            },
            'hamilton': {
                'points': hamilton_pts,
                'wins': int(wins * 0.4),
                'podiums': int(wins * 0.4) + 6,
                'rank_est': 'P4'
            }
        }
        
    def predict_constructors_championship(self, ferrari_points: float) -> List[Dict]:
        """Estimate full detailed standings"""
        # Benchmarks relative to Ferrari prediction
        standings = [
            {'team': 'Ferrari', 'points': int(ferrari_points)},
            {'team': 'Red Bull Racing', 'points': int(ferrari_points * 1.15)}, # Still benchmark
            {'team': 'McLaren', 'points': int(ferrari_points * 1.05)}, # Very close
            {'team': 'Mercedes', 'points': int(ferrari_points * 0.85)},
            {'team': 'Aston Martin', 'points': int(ferrari_points * 0.5)},
            {'team': 'Alpine', 'points': int(ferrari_points * 0.2)},
            {'team': 'Williams', 'points': int(ferrari_points * 0.15)},
            {'team': 'RB', 'points': int(ferrari_points * 0.1)},
            {'team': 'Haas', 'points': int(ferrari_points * 0.08)},
            {'team': 'Sauber/Audi', 'points': int(ferrari_points * 0.05)},
        ]
        
        # Sort
        standings.sort(key=lambda x: x['points'], reverse=True)
        
        # Add positions
        for i, team in enumerate(standings):
            team['position'] = i + 1
            
        return standings
    
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