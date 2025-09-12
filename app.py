import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# *** THIS IS THE FIX: Changed 'norequest' to 'request' ***
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# --- Configuration ---
MODEL_FILENAME = 'deepfake_model.h5'
IMAGE_SIZE = (128, 128) 

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Load your trained AI model ---
print("Attempting to load the deepfake detection model...")
model = None
if os.path.exists(MODEL_FILENAME):
    try:
        model = load_model(MODEL_FILENAME, compile=False) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model file: {e}")
else:
    print(f"⚠️ Warning: Model file '{MODEL_FILENAME}' not found.")
# -------------------------------------------------------------------

# --- Preprocessing Function ---
def preprocess_image(image_stream):
    image_array = np.frombuffer(image_stream.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, IMAGE_SIZE)
    normalized_image = resized_image / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image
# -----------------------------

# === ROUTES TO SERVE THE WEBSITE ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader')
def uploader():
    return render_template('uploader.html')
# ========================================

# === YOUR API ROUTE FOR PREDICTIONS ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not model:
        return jsonify({'error': 'AI model is not loaded on the server.'}), 500

    if file:
        try:
            processed_image = preprocess_image(file.stream)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0] 

            is_fake = confidence > 0.5 
            result_text = 'DEEPFAKE' if is_fake else 'REAL'
            display_confidence = (confidence if is_fake else 1 - confidence) * 100
            
            print(f"Prediction successful. Result: {result_text}, Confidence: {display_confidence:.2f}%")

            return jsonify({
                'result': result_text,
                'confidence': round(display_confidence, 2)
            })
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return jsonify({'error': 'Failed to process the file with the model.'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500
# =================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

