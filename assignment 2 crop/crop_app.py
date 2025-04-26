from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'crop_recommendation_secret_key'

# Crop Information Dictionary
CROP_INFO = {
    'rice': {
        'season': 'Kharif (Monsoon)',
        'duration': '120-150 days',
        'water_needs': 'High',
        'brief': 'Staple food crop that grows well in wet, humid conditions.'
    },
    'maize': {
        'season': 'Kharif/Rabi',
        'duration': '90-120 days',
        'water_needs': 'Moderate',
        'brief': 'Versatile crop that adapts well to different climates.'
    },
    'chickpea': {
        'season': 'Rabi (Winter)',
        'duration': '95-110 days',
        'water_needs': 'Low to Moderate',
        'brief': 'Drought-tolerant legume rich in protein.'
    },
    'kidneybeans': {
        'season': 'Kharif/Rabi',
        'duration': '85-95 days',
        'water_needs': 'Moderate',
        'brief': 'Nutritious legume that prefers well-drained soil.'
    },
    'pigeonpeas': {
        'season': 'Kharif',
        'duration': '120-180 days',
        'water_needs': 'Low to Moderate',
        'brief': 'Drought-resistant pulse crop with deep root system.'
    },
    'mothbeans': {
        'season': 'Kharif',
        'duration': '75-90 days',
        'water_needs': 'Low',
        'brief': 'Heat and drought tolerant legume.'
    },
    'mungbean': {
        'season': 'Kharif/Spring',
        'duration': '60-75 days',
        'water_needs': 'Moderate',
        'brief': 'Quick-growing pulse crop with good nutritional value.'
    },
    'blackgram': {
        'season': 'Kharif',
        'duration': '90-120 days',
        'water_needs': 'Moderate',
        'brief': 'Important pulse crop rich in protein.'
    },
    'lentil': {
        'season': 'Rabi',
        'duration': '120-150 days',
        'water_needs': 'Low to Moderate',
        'brief': 'Cool-season legume crop with high protein content.'
    },
    'pomegranate': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Fruit crop with high antioxidant content.'
    },
    'banana': {
        'season': 'Perennial',
        'duration': '300-365 days',
        'water_needs': 'High',
        'brief': 'Tropical fruit rich in potassium and vitamins.'
    },
    'mango': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Popular tropical fruit known as the king of fruits.'
    },
    'grapes': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Versatile fruit used for both fresh consumption and wine.'
    },
    'watermelon': {
        'season': 'Summer',
        'duration': '80-100 days',
        'water_needs': 'High',
        'brief': 'Refreshing summer fruit with high water content.'
    },
    'muskmelon': {
        'season': 'Summer',
        'duration': '85-100 days',
        'water_needs': 'Moderate',
        'brief': 'Sweet summer fruit rich in vitamins A and C.'
    },
    'apple': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Popular temperate fruit rich in fiber and antioxidants.'
    },
    'orange': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Citrus fruit high in vitamin C.'
    },
    'papaya': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Tropical fruit rich in papain enzyme.'
    },
    'coconut': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'High',
        'brief': 'Tropical palm with multiple uses.'
    },
    'cotton': {
        'season': 'Kharif',
        'duration': '150-180 days',
        'water_needs': 'Moderate',
        'brief': 'Important fiber crop for textile industry.'
    },
    'jute': {
        'season': 'Kharif',
        'duration': '120-150 days',
        'water_needs': 'High',
        'brief': 'Natural fiber crop used for making bags and ropes.'
    },
    'coffee': {
        'season': 'Perennial',
        'duration': 'Year-round',
        'water_needs': 'Moderate',
        'brief': 'Popular beverage crop grown in shaded conditions.'
    }
}

# Valid ranges for input parameters
VALID_RANGES = {
    'nitrogen': (0, 140),
    'phosphorus': (5, 145),
    'potassium': (5, 205),
    'temperature': (8.83, 43.68),
    'humidity': (14.26, 99.98),
    'ph': (3.50, 9.94),
    'rainfall': (20.21, 298.56)
}

def validate_input(data):
    """Validate input parameters against valid ranges."""
    errors = []
    for param, (min_val, max_val) in VALID_RANGES.items():
        value = data.get(param)
        if value is None:
            errors.append(f"{param} is required")
        elif not isinstance(value, (int, float)):
            errors.append(f"{param} must be a number")
        elif value < min_val or value > max_val:
            errors.append(f"{param} must be between {min_val} and {max_val}")
    return errors

def prepare_model():
    """Prepare and return the machine learning model."""
    try:
        # Try to load existing model
        if os.path.exists('crop_model.pkl'):
            logger.info("Loading existing model...")
            return joblib.load('crop_model.pkl')
        
        logger.info("Training new model...")
        # If model doesn't exist, train a new one
        df = pd.read_csv('Crop_recommendation.csv')
        
        # Prepare features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        
        # Save the model
        joblib.dump(model, 'crop_model.pkl')
        logger.info("Model saved successfully")
        
        return model
    except Exception as e:
        logger.error(f"Error preparing model: {str(e)}")
        return None

def get_crop_images():
    """Build a dictionary of available crop images."""
    crop_images = {}
    image_dir = os.path.join('static', 'images', 'crops')
    
    # Ensure the directory exists
    os.makedirs(image_dir, exist_ok=True)
    
    # Create a default image if it doesn't exist
    default_image_path = os.path.join(image_dir, 'default.jpg')
    if not os.path.exists(default_image_path):
        logger.warning("Default crop image not found. Please add one.")
    
    # Get all jpg images in the crops directory
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    # Build the dictionary mapping crop names to their image URLs
    for image_file in image_files:
        crop_name = os.path.basename(image_file).replace('.jpg', '').lower()
        crop_images[crop_name] = url_for('static', filename=f'images/crops/{crop_name}.jpg')
    
    logger.info(f"Found {len(crop_images)} crop images")
    return crop_images

# Ensure the images directory exists
os.makedirs('static/images/crops', exist_ok=True)

# Load the model
model = prepare_model()

# Get crop images
CROP_IMAGES = get_crop_images()

@app.route('/')
def home():
    """Home page route."""
    return render_template('crop_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle crop prediction."""
    try:
        # Extract form data
        input_data = {
            'nitrogen': float(request.form['nitrogen']),
            'phosphorus': float(request.form['phosphorus']),
            'potassium': float(request.form['potassium']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Validate input
        errors = validate_input(input_data)
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('home'))
        
        if model is None:
            flash("Model not available. Please check the logs for details.", 'danger')
            return redirect(url_for('home'))
        
        # Prepare features for prediction
        features = np.array([[
            input_data['nitrogen'],
            input_data['phosphorus'],
            input_data['potassium'],
            input_data['temperature'],
            input_data['humidity'],
            input_data['ph'],
            input_data['rainfall']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(features)[0]
        # Get top 3 recommendations with probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            (model.classes_[idx], round(probabilities[idx] * 100, 2))
            for idx in top_3_idx
        ]
        
        # Log the prediction
        logger.info(f"Prediction made: {prediction} with confidence {max(probabilities)*100:.2f}%")
        
        return render_template(
            'crop_result.html',
            prediction=prediction,
            recommendations=recommendations,
            input_values={
                'Nitrogen': input_data['nitrogen'],
                'Phosphorus': input_data['phosphorus'],
                'Potassium': input_data['potassium'],
                'Temperature': input_data['temperature'],
                'Humidity': input_data['humidity'],
                'pH': input_data['ph'],
                'Rainfall': input_data['rainfall']
            },
            crop_info=CROP_INFO,
            crop_images=CROP_IMAGES
        )
        
    except ValueError as e:
        flash("Please enter valid numerical values for all fields", 'danger')
        logger.error(f"Value error in prediction: {str(e)}")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'danger')
        logger.error(f"Error in prediction: {str(e)}")
        return redirect(url_for('home'))

@app.route('/about')
def about():
    """About page route."""
    return render_template('crop_about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 