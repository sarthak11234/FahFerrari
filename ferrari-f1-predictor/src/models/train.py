"""
Ferrari F1 ML Model Training Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FerrariMLTrainer:
    """Train and evaluate ML models for Ferrari F1 predictions"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'total_points',
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare training and test sets
        
        Args:
            df: Feature dataframe
            target_col: Target variable column name
            test_size: Test set proportion
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Select features (exclude target and identifiers)
        feature_cols = [col for col in df.columns 
                       if col not in ['year', target_col, 'position']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def initialize_models(self):
        """Initialize all models with optimal hyperparameters"""
        
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            
            'xgboost': xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
    
    def train_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train all models and return evaluation metrics
        
        Returns:
            Dictionary with model performance metrics
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=3, scoring='r2')
            
            # Metrics
            results[name] = {
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'model': model
            }
            
            print(f"✅ {name}:")
            print(f"   Test R²: {results[name]['test_r2']:.3f}")
            print(f"   Test MAE: {results[name]['test_mae']:.1f} points")
            print(f"   CV R²: {results[name]['cv_r2_mean']:.3f} (±{results[name]['cv_r2_std']:.3f})")
        
        return results
    
    def create_ensemble(self, results: Dict, weights: Dict = None) -> callable:
        """
        Create weighted ensemble of models
        
        Args:
            results: Model results dictionary
            weights: Model weights (default: equal weighting)
            
        Returns:
            Ensemble prediction function
        """
        if weights is None:
            weights = {
                'random_forest': 0.25,
                'xgboost': 0.35,
                'lightgbm': 0.25,
                'gradient_boosting': 0.15
            }
        
        def ensemble_predict(X):
            predictions = []
            for name, weight in weights.items():
                model = results[name]['model']
                pred = model.predict(X) * weight
                predictions.append(pred)
            return np.sum(predictions, axis=0)
        
        return ensemble_predict
    
    def save_models(self, results: Dict, ensemble_fn: callable):
        """Save trained models to disk"""
        
        # Save individual models
        for name, result in results.items():
            joblib.dump(result['model'], f'models/saved/{name}_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/saved/scaler.pkl')
        
        # Save ensemble function
        joblib.dump(ensemble_fn, 'models/saved/ensemble_model.pkl')
        
        print("\n✅ All models saved successfully!")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from XGBoost"""
        
        xgb_model = self.models['xgboost']
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance

# Main training script
if __name__ == "__main__":
    # Load processed features
    df = pd.read_csv('data/processed/ferrari_features.csv')
    
    # Initialize trainer
    trainer = FerrariMLTrainer()
    
    # Prepare data
    print("📊 Preparing data...")
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df)
    
    # Initialize models
    print("\n🎯 Initializing models...")
    trainer.initialize_models()
    
    # Train all models
    print("\n🏋️ Training models...")
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    print("\n🔮 Creating ensemble...")
    ensemble = trainer.create_ensemble(results)
    
    # Test ensemble
    ensemble_pred = ensemble(X_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n🎊 Ensemble Performance:")
    print(f"   R²: {ensemble_r2:.3f}")
    print(f"   MAE: {ensemble_mae:.1f} points")
    
    # Feature importance
    importance = trainer.get_feature_importance(feature_cols)
    print("\n📊 Top 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save everything
    trainer.save_models(results, ensemble)