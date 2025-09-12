# Deepfake Detection Project

This project implements a deepfake detection model trained on the **Deep Fake Detection (DFD) Original Dataset**. You can run inference using the pre-trained model weights, so training from scratch is not required.

---

## Project Structure

Deepfake-GitHub/
│
├── MProjectNew.ipynb          # Main Colab notebook for inference
├── requirements.txt           # Python dependencies
├── README.md                  # Project description
└── deepfake_data/             # Kaggle dataset folder

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

---

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset

Place the extracted dataset in:

Deepfake-GitHub/deepfake_data/

---

## Pre-trained Models

Download the pre-trained model weights from Google Drive:

- https://drive.google.com/file/d/1CuPB-pHumMdR17qBCf6Ltk8m0mIa1YvP/view?usp=drive_link
- https://drive.google.com/file/d/1iU4HH6RE3AAfwiK87094z2F1lvHd7ppZ/view?usp=drive_link
- https://drive.google.com/file/d/1P2AxK2C7YCyipT8m3zTl4wpnPrylIIdd/view?usp=drive_link
- https://drive.google.com/file/d/11YiwDzfyCv1A_1NIWkdqquPpXCEQ5Wmu/view?usp=drive_link

These files allow running inference without training from scratch.

---

## Running Inference

1. Open `MProjectNew.ipynb` in Google Colab.  
2. Make sure dataset and model paths are correct.  
3. Run the notebook cells to perform inference.

---

## Notes

- Large files like videos and `.pth` weights are **not included** in the repo; download via the links above.  
- Training from scratch is optional but will take hours.  
- Ensure sufficient GPU memory in Colab for smooth inference.
