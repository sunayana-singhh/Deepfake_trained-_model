import torch
import torchvision.models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# --- Configuration ---
MODEL_FILENAME = 'deepfake_hybrid_model_full.pth'
IMAGE_SIZE = (224, 224) 

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Load your trained AI model ---
print("Attempting to load the deepfake detection model...")
model = None
if os.path.exists(MODEL_FILENAME):
    try:
        # Allow MobileNetV2 class (used in your checkpoint)
        torch.serialization.add_safe_globals([torchvision.models.mobilenetv2.MobileNetV2])
        model = torch.load(MODEL_FILENAME, map_location='cpu', weights_only=False)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model file: {e}")
else:
    print(f"⚠️ Warning: Model file '{MODEL_FILENAME}' not found.")

# --- Define preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Preprocessing Function ---
def preprocess_image(image_stream):
    try:
        # Read image data
        image_data = image_stream.read()
        if len(image_data) == 0:
            raise ValueError("Empty image file received")
        
        image_array = np.frombuffer(image_data, np.uint8)
        if len(image_array) == 0:
            raise ValueError("Failed to convert image data to numpy array")
        
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image. Unsupported format or corrupted file")
        
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply preprocessing
        input_tensor = preprocess(pil_image).unsqueeze(0)
        
        return input_tensor
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise

def preprocess_frame(frame):
    """Convert OpenCV frame to model input tensor"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply preprocessing
        input_tensor = preprocess(pil_image).unsqueeze(0)
        
        return input_tensor
    except Exception as e:
        print(f"Error in preprocess_frame: {e}")
        raise

def process_video(video_stream):
    """Process video file and return aggregated prediction"""
    try:
        # Save video temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_stream.seek(0)
            temp_file.write(video_stream.read())
            temp_path = temp_file.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        frame_count = 0
        predictions = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video with {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 10th frame to speed up (same as notebook)
            if frame_count % 10 != 0:
                continue
            
            try:
                # Preprocess frame
                input_tensor = preprocess_frame(frame)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    predictions.append(pred)
                
                # Log progress every 50 processed frames
                if len(predictions) % 50 == 0:
                    print(f"Processed {len(predictions)} frames...")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        
        # Clean up temporary file
        import os
        os.unlink(temp_path)
        
        if not predictions:
            raise ValueError("No frames could be processed from the video")
        
        # Aggregate predictions (same logic as notebook)
        fake_count = sum(predictions)
        total_count = len(predictions)
        fake_percentage = (fake_count / total_count) * 100
        
        print(f"Video analysis complete: {total_count} frames analyzed, {fake_count} detected as fake ({fake_percentage:.1f}%)")
        
        # Video is fake if more than 50% of frames are fake
        is_fake = fake_percentage > 50
        result_text = 'DEEPFAKE' if is_fake else 'REAL'
        confidence = fake_percentage if is_fake else (100 - fake_percentage)
        
        return {
            'result': result_text,
            'confidence': round(confidence, 2),
            'frames_analyzed': total_count,
            'fake_frames': fake_count
        }
        
    except Exception as e:
        print(f"Error in process_video: {e}")
        raise
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
            # Check file properties
            print(f"Processing file: {file.filename}, Content-Type: {file.content_type}")
            
            # Reset stream position to beginning (in case it was read before)
            file.stream.seek(0)
            
            # Determine if it's an image or video
            is_video = file.content_type and file.content_type.startswith('video/')
            is_image = file.content_type and file.content_type.startswith('image/')
            
            # Also check by file extension if content-type is not reliable
            if not (is_video or is_image):
                filename_lower = file.filename.lower()
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
                
                is_video = any(filename_lower.endswith(ext) for ext in video_extensions)
                is_image = any(filename_lower.endswith(ext) for ext in image_extensions)
            
            if is_video:
                print("Processing as video...")
                result = process_video(file.stream)
                print(f"Video prediction successful. Result: {result['result']}, Confidence: {result['confidence']:.2f}%")
                return jsonify(result)
                
            elif is_image:
                print("Processing as image...")
                # Preprocess the image
                input_tensor = preprocess_image(file.stream)
                
                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(output, dim=1)
                    # Get the predicted class (0=REAL, 1=FAKE)
                    pred = torch.argmax(output, dim=1).item()
                    # Get the confidence score for the predicted class
                    confidence = probabilities[0][pred].item() * 100
                
                result_text = 'DEEPFAKE' if pred == 1 else 'REAL'
                
                print(f"Image prediction successful. Result: {result_text}, Confidence: {confidence:.2f}%")

                return jsonify({
                    'result': result_text,
                    'confidence': round(confidence, 2)
                })
            else:
                return jsonify({'error': 'Unsupported file type. Please upload an image or video file.'}), 400
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500
# =================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

