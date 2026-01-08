from src.models.predict import Ferrari2026Predictor
import pandas as pd
import sys

def verify_system():
    print("🧪 Verifying Ferrari F1 Predictor System...")
    
    # 1. Check data file
    try:
        df = pd.read_csv('data/processed/ferrari_features.csv')
        print(f"✅ Loaded features: {len(df)} rows")
        print(f"   Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
        
    # 2. Check Predictor
    try:
        predictor = Ferrari2026Predictor()
        print("✅ Predictor initialized")
        
        features_2026 = predictor.prepare_2026_features(df)
        print("✅ Prepared 2026 features")
        
        prediction = predictor.predict_2026(features_2026)
        print("✅ generated 2026 prediction")
        
        print("\n📊 2026 FORECAST:")
        print(f"   Points: {prediction['predicted_points']}")
        print(f"   Position: {prediction['predicted_position']}")
        print(f"   Confidence Interval: {prediction['confidence_interval']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_system():
        print("\n✨ System Validated Successfully!")
        sys.exit(0)
    else:
        print("\n💀 System Validation Failed!")
        sys.exit(1)
