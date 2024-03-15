from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('agrigeo_model.pkl')  # Replace with your actual model file path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming JSON input, make sure to replace 'feature1', 'feature2', etc., with your actual feature names
        data = request.json
        features = [
          
        ]

        # Convert features to float (if needed)
        features = [float(feature) for feature in features]

        # Make a prediction
        prediction = model.predict([features])[0]

        # Convert NumPy array to a standard Python list
        prediction = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
