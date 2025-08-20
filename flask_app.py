from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # For server deployment
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Global variables for model and data
model = None
feature_names = ['allergic_reaction', 'Dose_2', 'important_of_Vaccination', 
                'misinformation_about_vaccines', 'believe_vaccines_safe', 
                'severity_of_side_effects', 'Region']

def load_model():
    """Load or initialize the model"""
    global model
    try:
        # Try to load pre-trained model
        with open('covid_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        # Initialize a dummy model for demonstration
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Create dummy training data
        X_dummy = np.random.randn(100, len(feature_names))
        y_dummy = np.random.choice([0, 1], 100)
        model.fit(X_dummy, y_dummy)

def create_plot():
    """Create a sample performance plot"""
    models = ['Random Forest', 'XGBoost', 'Decision Tree', 'Logistic Regression']
    accuracy = [0.6709, 0.6456, 0.6582, 0.6076]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracy, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 0.8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def home():
    """Home page"""
    plot_url = create_plot()
    return render_template('index.html', plot_url=plot_url)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('allergic_reaction', 0)),
            float(data.get('dose_2', 0)),
            float(data.get('vaccination_importance', 3)),
            float(data.get('misinformation', 0)),
            float(data.get('vaccine_belief', 3)),
            float(data.get('severity_concern', 3)),
            float(data.get('region', 0))
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Calculate risk score
        risk_score = probability[1]  # Probability of side effects
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "High Risk"
            risk_color = "#d32f2f"
            recommendation = "Enhanced monitoring and pre-medication recommended"
        elif risk_score > 0.4:
            risk_level = "Moderate Risk"
            risk_color = "#f57c00"
            recommendation = "Standard monitoring with additional precautions"
        else:
            risk_level = "Low Risk"
            risk_color = "#388e3c"
            recommendation = "Standard monitoring protocol sufficient"
        
        # Feature importance (mock data)
        feature_importance = {
            'allergic_reaction': features[0] * 0.122,
            'dose_2': features[1] * (-0.029),
            'vaccination_importance': features[2] * 0.028,
            'misinformation': features[3] * 0.025,
            'vaccine_belief': features[4] * (-0.018),
            'severity_concern': features[5] * 0.016,
            'region': features[6] * 0.010
        }
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/stats')
def get_stats():
    """API endpoint for model statistics"""
    stats = {
        'model_accuracy': 0.6709,
        'f1_score': 0.7451,
        'auc_roc': 0.7443,
        'total_features': len(feature_names),
        'selected_features': feature_names,
        'dataset_size': 395,
        'validation_method': '5-fold Cross-Validation',
        'algorithm': 'Novel MAFS + Random Forest'
    }
    return jsonify(stats)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
