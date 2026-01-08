"""
Ferrari F1 Predictor - Flask REST API
"""

from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import joblib
from src.chatbot.query_engine import FerrariChatbot
from src.models.predict import Ferrari2026Predictor
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = CustomJSONProvider(app)

# Configure logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data and models on startup
try:
    historical_data = pd.read_csv('data/processed/ferrari_features.csv')
    predictor = Ferrari2026Predictor()
    
    # Generate predictions
    features_2026 = predictor.prepare_2026_features(historical_data)
    predictions_2026 = predictor.predict_2026(features_2026)
    
    # Initialize chatbot
    chatbot = FerrariChatbot(historical_data, predictions_2026)
    
    logger.info("✅ Models and data loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    historical_data = None
    predictions_2026 = None
    chatbot = None

# ============ API ROUTES ============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Ferrari F1 Predictor API is running',
        'models_loaded': chatbot is not None
    })

@app.route('/api/historical/<int:year>', methods=['GET'])
def get_historical_data(year):
    """Get historical data for a specific year"""
    try:
        if historical_data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        year_data = historical_data[historical_data['year'] == year]
        
        if year_data.empty:
            return jsonify({'error': f'No data for year {year}'}), 404
        
        return jsonify(year_data.to_dict(orient='records')[0])
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical', methods=['GET'])
def get_all_historical():
    """Get all historical data"""
    try:
        if historical_data is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        return jsonify(historical_data.to_dict(orient='records'))
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/2026', methods=['GET'])
def get_2026_predictions():
    """Get 2026 season predictions"""
    try:
        if predictions_2026 is None:
            return jsonify({'error': 'Predictions not available'}), 500
        
        return jsonify(predictions_2026)
    
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/scenarios', methods=['GET'])
def get_scenario_predictions():
    """Get scenario-based predictions"""
    try:
        features_2026 = predictor.prepare_2026_features(historical_data)
        scenarios = predictor.generate_scenario_predictions(features_2026)
        
        return jsonify(scenarios)
    
    except Exception as e:
        logger.error(f"Error generating scenarios: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint"""
    try:
        if chatbot is None:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        response = chatbot.generate_response(query)
        
        return jsonify({
            'query': query,
            'response': response,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    """Get driver statistics"""
    return jsonify({
        'leclerc': {
            'name': 'Charles Leclerc',
            'year': 2025,
            'position': 5,
            'points': 242,
            'wins': 0,
            'podiums': 7
        },
        'hamilton': {
            'name': 'Lewis Hamilton',
            'year': 2025,
            'position': 6,
            'points': 156,
            'wins': 0,
            'podiums': 0
        }
    })

@app.route('/api/compare', methods=['POST'])
def compare_seasons():
    """Compare multiple seasons"""
    try:
        data = request.get_json()
        years = data.get('years', [])
        
        if not years or len(years) < 2:
            return jsonify({'error': 'At least 2 years required'}), 400
        
        comparison = []
        for year in years:
            year_data = historical_data[historical_data['year'] == year]
            if not year_data.empty:
                comparison.append(year_data.to_dict(orient='records')[0])
        
        return jsonify(comparison)
    
    except Exception as e:
        logger.error(f"Error comparing seasons: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/summary', methods=['GET'])
def get_summary_stats():
    """Get overall summary statistics"""
    try:
        summary = {
            'total_seasons': len(historical_data),
            'years_covered': f"{historical_data['year'].min()}-{historical_data['year'].max()}",
            'total_points': int(historical_data['total_points'].sum()),
            'total_wins': int(historical_data['wins'].sum()),
            'total_podiums': int(historical_data['podiums'].sum()),
            'best_season': {
                'year': int(historical_data.loc[historical_data['total_points'].idxmax(), 'year']),
                'points': int(historical_data['total_points'].max())
            },
            'worst_season': {
                'year': int(historical_data.loc[historical_data['total_points'].idxmin(), 'year']),
                'points': int(historical_data['total_points'].min())
            }
        }
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)