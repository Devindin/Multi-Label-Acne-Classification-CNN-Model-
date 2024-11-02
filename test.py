from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": "*"}})  # Enable CORS for /classify route with any origin

# Load the pre-trained model
model = load_model('acne_classification_model.keras')

# Define the classes
acne_types = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']

def classify_images(image_path):
    try:
        # Adjust image loading based on model's expected input size
        input_shape = model.input_shape[1:]  # Exclude batch dimension
        target_size = (input_shape[0], input_shape[1])  # (height, width)

        # Load and preprocess the image to the required size
        input_image = tf.keras.utils.load_img(image_path, target_size=target_size)
        input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0

        # Expand dimensions for batch size
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Get predictions
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0]).numpy()  # Apply softmax for probabilities

        # Map predictions to classes and probabilities
        outcome = [{"class": acne_types[i], "probability": float(result[i]) * 100} for i in range(len(acne_types))]

        # Sort and get the top 3 results
        top_3_outcome = sorted(outcome, key=lambda x: x["probability"], reverse=True)[:3]

        return top_3_outcome
    except Exception as e:
        return {"error": str(e)}

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Define upload directory
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Perform classification
    result = classify_images(file_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
