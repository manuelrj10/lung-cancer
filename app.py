from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "final_vgg16_model_fixed.keras"
model = load_model(MODEL_PATH, compile=False)

# Class labels (same order as in training)
CLASS_NAMES = [
    "Lung Adenocarcinoma",
    "Lung Normal (Benign Tissue)",
    "Lung Squamous Cell Carcinoma"
]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Model prediction
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    # Render result
    return render_template(
        'index.html',
        prediction_text=f"Predicted Class: {predicted_class} ({confidence}%)",
        image_path=file_path
    )

if __name__ == "__main__":
    app.run(debug=True)
