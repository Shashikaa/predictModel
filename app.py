import os
import logging
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
with open('risk_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")
        
        # Validate input data
        if 'features' not in data or not isinstance(data['features'], list):
            return jsonify({"error": "Invalid input format. Expected a list of features."}), 400
        
        features = data['features']
        
        # Check if the number of features is correct
        if len(features) != 22:  # Adjust this number based on your actual feature count
            return jsonify({"error": f"Invalid number of features. Expected 22, got {len(features)}."}), 400
        
        # Define expected feature names (adjust according to your actual feature names)
        feature_names = ['latitude', 'longitude', 'History_Of_Loss', 
                         'Activity_stationary', 'Activity_walking', 
                         'Item_Type_backpack', 'Item_Type_keys', 
                         'Item_Type_wallet', 'Item_Type_phone',
                         # Add other feature names as needed...
                         ]

        # Create a DataFrame with default values for all features
        default_values = {name: 0 for name in feature_names}  # Default all to 0
        input_data = pd.DataFrame([default_values])

        # Update DataFrame with actual input values
        for i, name in enumerate(feature_names):
            if i < len(features):
                input_data.at[0, name] = features[i]

        # Make prediction
        prediction = model.predict(input_data)
        logging.info(f"Prediction: {prediction[0]}")
        
        return jsonify({'Risk_Level': prediction[0]})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
