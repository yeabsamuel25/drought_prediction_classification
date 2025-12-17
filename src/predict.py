"""
Prediction Module
Makes predictions on new data using all trained models
"""

import joblib
import pandas as pd
import numpy as np
import os


def load_models_and_scaler():
    """Load all saved models and scaler"""
    print("\n" + "=" * 80)
    print("LOADING MODELS AND SCALER")
    print("=" * 80)
    
    # Check if models exist
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'SVM': 'models/svm.pkl',
        'KNN': 'models/knn.pkl',
        'Naive Bayes': 'models/naive_bayes.pkl'
    }
    
    missing_files = []
    for name, path in model_files.items():
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("\n❌ ERROR: Missing model files!")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run 'python main.py' first to train the models.")
        return None, None
    
    # Check if scaler exists
    if not os.path.exists('models/scaler.pkl'):
        print("\n❌ ERROR: Scaler not found!")
        print("   Expected: models/scaler.pkl")
        print("\nPlease run 'python main.py' first to train the models.")
        return None, None
    
    # Load all models
    models = {}
    for name, path in model_files.items():
        models[name] = joblib.load(path)
        print(f"   ✅ Loaded: {name}")
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    print(f"   ✅ Loaded: Scaler")
    
    print("\n✅ All models and scaler loaded successfully!")
    
    return models, scaler


def get_feature_names():
    """Return the list of features needed for prediction"""
    return [
        'RH2M',              # Relative Humidity at 2 Meters (%)
        'T2M_MAX',           # Maximum Temperature (°C)
        'T2M_MIN',           # Minimum Temperature (°C)
        'WS2M',              # Wind Speed at 2 Meters (m/s)
        'T2M',               # Mean Temperature (°C)
        'ALLSKY_SFC_SW_DWN', # Solar Radiation (kW-hr/m²/day)
        'PRECTOTCORR',       # Precipitation (mm/day)
        'spei',              # Drought Index (SPEI)
        'lat_sin',           # Latitude (sine)
        'lat_cos',           # Latitude (cosine)
        'lon_sin',           # Longitude (sine)
        'lon_cos',           # Longitude (cosine)
        'month_sin',         # Month (sine)
        'month_cos'          # Month (cosine)
    ]


def get_feature_descriptions():
    """Return descriptions for each feature"""
    return {
        'RH2M': 'Relative Humidity (0-100%, e.g., 65)',
        'T2M_MAX': 'Maximum Temperature (°C, e.g., 35)',
        'T2M_MIN': 'Minimum Temperature (°C, e.g., 15)',
        'WS2M': 'Wind Speed (m/s, e.g., 3.5)',
        'T2M': 'Mean Temperature (°C, e.g., 25)',
        'ALLSKY_SFC_SW_DWN': 'Solar Radiation (kW-hr/m²/day, e.g., 15)',
        'PRECTOTCORR': 'Precipitation (mm/day, e.g., 2.5)',
        'spei': 'SPEI Drought Index (-3 to +3, negative=drought, e.g., -1.5)',
        'lat_sin': 'Latitude sine (-1 to 1, e.g., 0.5)',
        'lat_cos': 'Latitude cosine (-1 to 1, e.g., 0.87)',
        'lon_sin': 'Longitude sine (-1 to 1, e.g., -0.23)',
        'lon_cos': 'Longitude cosine (-1 to 1, e.g., 0.97)',
        'month_sin': 'Month sine (-1 to 1, e.g., 0.5)',
        'month_cos': 'Month cosine (-1 to 1, e.g., 0.87)'
    }


def get_feature_ranges():
    """Return typical ranges for features"""
    return {
        'RH2M': (0, 100),
        'T2M_MAX': (-10, 50),
        'T2M_MIN': (-20, 40),
        'WS2M': (0, 20),
        'T2M': (-15, 45),
        'ALLSKY_SFC_SW_DWN': (0, 30),
        'PRECTOTCORR': (0, 50),
        'spei': (-3, 3),
        'lat_sin': (-1, 1),
        'lat_cos': (-1, 1),
        'lon_sin': (-1, 1),
        'lon_cos': (-1, 1),
        'month_sin': (-1, 1),
        'month_cos': (-1, 1)
    }


def predict_single_sample(models, scaler, feature_values):
    """
    Make predictions using all models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scaler : StandardScaler
        Fitted scaler
    feature_values : list or array
        Values for all 14 features
        
    Returns:
    --------
    predictions : dict
        Dictionary with predictions and probabilities from each model
    """
    # Convert to DataFrame
    feature_names = get_feature_names()
    X_new = pd.DataFrame([feature_values], columns=feature_names)
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions with all models
    predictions = {}
    
    for model_name, model in models.items():
        # Get prediction
        pred = model.predict(X_new_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_new_scaled)[0]
            confidence = proba[pred] * 100  # Convert to percentage
        else:
            confidence = None
        
        predictions[model_name] = {
            'prediction': int(pred),
            'label': 'Drought' if pred == 1 else 'No Drought',
            'confidence': confidence
        }
    
    return predictions


def print_predictions(predictions, feature_values):
    """Print predictions in a nice format"""
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    # Print input summary
    print("\n📊 INPUT DATA SUMMARY:")
    feature_names = get_feature_names()
    key_features = ['RH2M', 'T2M', 'PRECTOTCORR', 'spei']
    
    for feat in key_features:
        idx = feature_names.index(feat)
        value = feature_values[idx]
        print(f"   {feat:.<20} {value}")
    
    # Print predictions from each model
    print("\n" + "=" * 80)
    print("PREDICTIONS FROM ALL MODELS:")
    print("=" * 80)
    
    for model_name, result in predictions.items():
        emoji = "🌵" if result['prediction'] == 1 else "💧"
        
        print(f"\n{emoji} {model_name:.<30}", end=" ")
        
        if result['label'] == 'Drought':
            print(f"→ \033[91m{result['label']}\033[0m", end="")  # Red
        else:
            print(f"→ \033[92m{result['label']}\033[0m", end="")  # Green
        
        if result['confidence']:
            print(f" (Confidence: {result['confidence']:.1f}%)")
        else:
            print()
    
    # Consensus
    drought_count = sum(1 for r in predictions.values() if r['prediction'] == 1)
    no_drought_count = len(predictions) - drought_count
    
    print("\n" + "=" * 80)
    print("CONSENSUS:")
    print("=" * 80)
    print(f"   Models predicting DROUGHT:    {drought_count}/4")
    print(f"   Models predicting NO DROUGHT: {no_drought_count}/4")
    
    if drought_count > no_drought_count:
        print("\n   🌵 MAJORITY VOTE: \033[91mDROUGHT\033[0m")
        print("   ⚠️  Recommendation: Implement drought mitigation measures")
    elif no_drought_count > drought_count:
        print("\n   💧 MAJORITY VOTE: \033[92mNO DROUGHT\033[0m")
        print("   ✅ Recommendation: Normal conditions expected")
    else:
        print("\n   ⚖️  TIE: Models are split!")
        print("   ⚠️  Recommendation: Monitor conditions closely")
    
    if drought_count == 4:
        print("\n   🚨 ALL MODELS AGREE: High confidence DROUGHT prediction!")
    elif no_drought_count == 4:
        print("\n   ✅ ALL MODELS AGREE: High confidence NO DROUGHT prediction!")


def predict_from_dict(models, scaler, feature_dict):
    """Make prediction from a dictionary of feature values"""
    feature_names = get_feature_names()
    feature_values = [feature_dict[name] for name in feature_names]
    return predict_single_sample(models, scaler, feature_values)


def get_example_data():
    """Return example data for testing"""
    return {
        'drought_example': {
            'name': 'Severe Drought Conditions',
            'description': 'Hot, dry conditions with low precipitation',
            'data': {
                'RH2M': 35.0,              # Low humidity
                'T2M_MAX': 38.0,           # High temperature
                'T2M_MIN': 22.0,
                'WS2M': 4.5,
                'T2M': 30.0,               # Hot
                'ALLSKY_SFC_SW_DWN': 25.0, # High solar radiation
                'PRECTOTCORR': 0.2,        # Very low precipitation
                'spei': -2.1,              # Strong negative SPEI (drought!)
                'lat_sin': 0.5,
                'lat_cos': 0.87,
                'lon_sin': -0.23,
                'lon_cos': 0.97,
                'month_sin': 0.5,
                'month_cos': 0.87
            }
        },
        'no_drought_example': {
            'name': 'Normal Wet Conditions',
            'description': 'Moderate temperature with good rainfall',
            'data': {
                'RH2M': 75.0,              # Good humidity
                'T2M_MAX': 28.0,           # Moderate temperature
                'T2M_MIN': 18.0,
                'WS2M': 2.5,
                'T2M': 23.0,               # Comfortable
                'ALLSKY_SFC_SW_DWN': 15.0,
                'PRECTOTCORR': 5.0,        # Good rainfall
                'spei': 1.2,               # Positive SPEI (wet!)
                'lat_sin': 0.5,
                'lat_cos': 0.87,
                'lon_sin': -0.23,
                'lon_cos': 0.97,
                'month_sin': 0.5,
                'month_cos': 0.87
            }
        },
        'moderate_example': {
            'name': 'Moderate Drought Risk',
            'description': 'Below average rainfall, warm temperatures',
            'data': {
                'RH2M': 50.0,              # Medium humidity
                'T2M_MAX': 32.0,           # Warm
                'T2M_MIN': 20.0,
                'WS2M': 3.0,
                'T2M': 26.0,
                'ALLSKY_SFC_SW_DWN': 18.0,
                'PRECTOTCORR': 1.5,        # Low rainfall
                'spei': -0.8,              # Slightly negative SPEI
                'lat_sin': 0.5,
                'lat_cos': 0.87,
                'lon_sin': -0.23,
                'lon_cos': 0.97,
                'month_sin': 0.5,
                'month_cos': 0.87
            }
        }
    }


def validate_input(value, feature_name):
    """Validate user input is within reasonable range"""
    ranges = get_feature_ranges()
    
    if feature_name not in ranges:
        return True, value
    
    min_val, max_val = ranges[feature_name]
    
    if value < min_val or value > max_val:
        return False, f"Value {value} outside typical range [{min_val}, {max_val}]"
    
    return True, value