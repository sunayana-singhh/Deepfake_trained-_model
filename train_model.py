import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import time

# --- Configuration ---
# IMPORTANT: Change this path to where your dataset is located.
# If your 'dataset' folder is inside 'veritasai-project', this path is correct.
DATASET_PATH = './dataset'
IMAGE_SIZE = (128, 128) # The size to resize frames to
MAX_FRAMES_PER_VIDEO = 10 # To keep processing time reasonable, we'll extract 10 frames per video.
EPOCHS = 10 # How many times the model will go through the data. Increase for better results later.
BATCH_SIZE = 32

# --- This is the preprocessing logic that you MUST copy to app.py ---
def preprocess_frame(frame):
    """
    Takes a single video frame and prepares it for the model.
    """
    # 1. Resize the image to the standard size
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    # 2. Normalize pixel values to be between 0 and 1 (from 0-255)
    normalized_frame = resized_frame / 255.0
    return normalized_frame
# --------------------------------------------------------------------

def load_data(dataset_path):
    """
    Loads videos from the dataset, extracts frames, preprocesses them, and assigns labels.
    """
    print("Starting data loading process...")
    data = []
    labels = []
    
    real_path = os.path.join(dataset_path, 'real')
    fake_path = os.path.join(dataset_path, 'fake')

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"Error: Dataset folders not found at '{real_path}' and '{fake_path}'")
        print("Please make sure you have created a 'dataset' folder with 'real' and 'fake' subfolders.")
        return None, None

    # Process REAL videos (label 0)
    for video_file in os.listdir(real_path):
        video_path = os.path.join(real_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened() and frame_count < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = preprocess_frame(frame)
            data.append(processed_frame)
            labels.append(0) # 0 for REAL
            frame_count += 1
        cap.release()
        print(f"Processed REAL video: {video_file}")

    # Process FAKE videos (label 1)
    for video_file in os.listdir(fake_path):
        video_path = os.path.join(fake_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened() and frame_count < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = preprocess_frame(frame)
            data.append(processed_frame)
            labels.append(1) # 1 for FAKE
            frame_count += 1
        cap.release()
        print(f"Processed FAKE video: {video_file}")

    print("Data loading complete.")
    return np.array(data), np.array(labels)

def build_model():
    """
    Builds a simple Convolutional Neural Network (CNN) for image classification.
    This is a standard architecture for tasks like this.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent the model from just memorizing the data
        Dense(1, activation='sigmoid') # Sigmoid is used for binary (Real/Fake) classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Model built successfully.")
    model.summary() # Prints a table summarizing the model's architecture
    return model

# --- Main Training Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Load the data from the folders
    X, y = load_data("/Users/tm/veritasai-project/dataset")

    
    if X is None or len(X) == 0:
        print("Could not load any data. Please check your dataset path and folder structure.")
    else:
        # 2. Split data into training and testing sets (80% for training, 20% for testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

        # 3. Build the model
        model = build_model()
        
        # 4. Train the model on the data
        print("\nStarting model training...")
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test))
        print("Model training complete.")

        # 5. Evaluate the model's performance on the test data it has never seen
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")

        # 6. Save the final model - this creates the all-important .h5 file
        print("\nSaving the trained model as 'deepfake_model.h5'...")
        model.save('deepfake_model.h5')
        print("✅ Model saved successfully!")

    end_time = time.time()
    print(f"\nTotal script time: {(end_time - start_time) / 60:.2f} minutes.")

