"""
Ferrari F1 Predictor - Flask REST API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import joblib
from src.chatbot.query_engine import FerrariChatbot
from src.models.predict import Ferrari2026Predictor
from pathlib import Path
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load 2027 Preseason Forecast artifact (10,000 Monte Carlo runs)
predictions_2027 = None
try:
    pred_2027_path = Path(__file__).resolve().parents[1] / "data" / "predictions" / "preseason_2027_forecast.json"
    if pred_2027_path.exists():
        with pred_2027_path.open("r", encoding="utf-8") as f:
            predictions_2027 = json.load(f)
        logger.info("✅ 2027 Preseason Forecast loaded successfully")
except Exception as e:
    logger.warning(f"Could not load 2027 forecast: {e}")

# Load 2026 Preseason Legacy Forecast & Scenarios artifacts
predictions_2026 = None
scenarios_2026 = None
try:
    pred_2026_path = Path(__file__).resolve().parents[1] / "data" / "predictions" / "legacy_2026_forecast.json"
    if pred_2026_path.exists():
        with pred_2026_path.open("r", encoding="utf-8") as f:
            predictions_2026 = json.load(f)
        logger.info("✅ 2026 Legacy Forecast loaded successfully")
        
    scenarios_path = Path(__file__).resolve().parents[1] / "data" / "predictions" / "scenarios_2026.json"
    if scenarios_path.exists():
        with scenarios_path.open("r", encoding="utf-8") as f:
            scenarios_2026 = json.load(f)
except Exception as e:
    logger.warning(f"Could not load 2026 pre-cached artifacts: {e}")

# Load historical data and models on startup
historical_data = None
try:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "ferrari_features.csv"
    if csv_path.exists():
        historical_data = pd.read_csv(csv_path)
    elif Path('data/processed/ferrari_features.csv').exists():
        historical_data = pd.read_csv('data/processed/ferrari_features.csv')
    logger.info("✅ Historical features loaded successfully")
except Exception as e:
    logger.warning(f"Could not load historical features: {e}")

predictor = None
try:
    predictor = Ferrari2026Predictor()
    if historical_data is not None and predictions_2026 is None:
        features_2026 = predictor.prepare_2026_features(historical_data)
        predictions_2026 = predictor.predict_2026(features_2026)
except Exception as e:
    logger.info(f"Model engine inference bypassed; utilizing pre-cached 2026 baseline: {e}")

# Initialize chatbot
chatbot = None
try:
    chatbot = FerrariChatbot(historical_data, predictions_2026, predictions_2027)
    logger.info("✅ Chatbot initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize chatbot: {e}")

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

@app.route('/api/predictions/2027', methods=['GET'])
def get_2027_predictions():
    """Get 2027 season predictions (10,000 Monte Carlo simulations)"""
    try:
        if predictions_2027 is None:
            return jsonify({'error': '2027 Predictions not available'}), 404
        
        return jsonify(predictions_2027)
    
    except Exception as e:
        logger.error(f"Error fetching 2027 predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/2027/provenance', methods=['GET'])
def get_2027_provenance():
    """Get 2027 forecast freeze metadata and provenance"""
    try:
        if predictions_2027 is None:
            return jsonify({'error': '2027 Predictions not available'}), 404
        
        return jsonify(predictions_2027.get('freeze_metadata', {}))
    
    except Exception as e:
        logger.error(f"Error fetching 2027 provenance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/scenarios', methods=['GET'])
def get_scenario_predictions():
    """Get scenario-based predictions"""
    try:
        if predictor is not None and historical_data is not None:
            features_2026 = predictor.prepare_2026_features(historical_data)
            return jsonify(predictor.generate_scenario_predictions(features_2026))
        if scenarios_2026 is not None:
            return jsonify(scenarios_2026)
        return jsonify({'error': 'Scenarios not available'}), 404
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

# Static Frontend SPA Routing (Single-Container Full-Stack Support)
STATIC_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api'):
        return jsonify({'error': 'Endpoint not found'}), 404
    if STATIC_DIST.exists():
        target = STATIC_DIST / path
        if path != "" and target.exists() and not target.is_dir():
            return send_from_directory(STATIC_DIST, path)
        return send_from_directory(STATIC_DIST, 'index.html')
    return jsonify({
        'name': 'FahFerrari Performance Intelligence API',
        'season': '2027',
        'status': 'OPERATIONAL'
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)