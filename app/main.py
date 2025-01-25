import os
import uuid

import cv2
from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS

from app.model_training_loading import load_or_train_model, super_resolve_with_multiplier

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Paths
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stored_models", "trained_super_resolution_model.keras")
INPUT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_assets", "test_img.png")
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_assets", "output_super_resolved_image.jpg")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_assets")
UPLOAD_FOLDER=  os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
# Load or train the model
# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load or train the model
model = load_or_train_model(MODEL_PATH, DATASET_PATH)

@app.route('/upload', methods=['POST'])
def upload_and_process_image():
    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if not file:
            return jsonify({"error": "Invalid image file"}), 400

        # Save the uploaded image
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)

        # Read and process the image
        with open(input_path, 'rb') as f:
            image_binary = f.read()

        multiplier = int(request.form.get('multiplier', 2))  # Default multiplier = 2
        super_resolved_image = super_resolve_with_multiplier(model, image_binary, multiplier=multiplier)

        # Save the processed image
        output_filename = f"{unique_id}_super_resolved.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, super_resolved_image)

        # Generate a public URL for the image
        output_url = url_for('get_image', filename=output_filename, _external=True)

        return jsonify({"success": True, "output_url": output_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)