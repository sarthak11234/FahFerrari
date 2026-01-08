from src.chatbot.query_engine import FerrariChatbot
import pandas as pd
import traceback

def reproduce_error():
    print("reproducing error...")
    try:
        # Load real data
        df = pd.read_csv('data/processed/ferrari_features.csv')
        print(f"Columns: {df.columns.tolist()}")
        
        # Mock predictions
        preds = {
            'predicted_points': 500,
            'confidence_interval': (450, 550),
            'predicted_position': 3,
            'predicted_wins': 2,
            'predicted_podiums': 10
        }
        
        chatbot = FerrariChatbot(df, preds)
        
        # Trigger the failing query
        print("Query: 'how was 2025 season'")
        response = chatbot.generate_response("how was 2025 season")
        print(f"Response: {response}")
        
    except Exception as e:
        print("\n❌ CAUGHT EXPECTED ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_error()
