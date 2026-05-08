"""
Flask Web Application for Fake News Detection System
"""

import os
from flask import Flask, render_template, request, jsonify
from predict import FakeNewsPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fakenews-detection-secret-key-2024'

predictor = None


def initialize_predictor():
    """Initialize the predictor with trained model"""
    global predictor
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    try:
        predictor = FakeNewsPredictor(model_path)
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Warning: Model not found. Please run train.py first.")
        return False


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Empty text provided'
            }), 400
        
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please run train.py first.'
            }), 500
        
        result = predictor.predict(text)
        
        return jsonify({
            'success': True,
            'label': result['label'],
            'confidence': result['confidence'],
            'probabilities': result.get('probabilities', {})
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION WEB APPLICATION")
    print("="*60)
    
    initialize_predictor()
    
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
