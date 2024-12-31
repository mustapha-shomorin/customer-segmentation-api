from flask import Flask, request, jsonify
import joblib
import numpy as np


# Load the saved model and preprocessing pipeline
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
kmeans = joblib.load('kmeans_model.pkl')

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Customer Segmentation API!"



# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
        # Get the input data
        data = request.get_json()

        # Input validation
        if 'features' not in data or not isinstance(data['features'], list):
            return jsonify({'error': 'Invalid input. "features" must be a list.'}), 400
        
        # Ensure the input length matches model requirements
        features = np.array(data['features']).reshape(1, -1)
        if features.shape[1] != 4:  # Adjust the number based on your input size
            return jsonify({'error': 'Input must contain exactly 4 features.'}), 400

        # Preprocess the input data
        scaled_features = scaler.transform(features)
        reduced_features = pca.transform(scaled_features)

        # Make predictions
        cluster = kmeans.predict(reduced_features)[0]

        # Return the prediction
        return jsonify({'cluster': int(cluster)})


# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)