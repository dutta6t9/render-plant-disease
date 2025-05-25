from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os
import gdown

app = Flask(__name__)

# Model download setup
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_recog_model_pwp.keras")
GOOGLE_DRIVE_FILE_ID = "1BsxReif2gMxQ6kVy_WBQDelKvTQ06XA0"  # <-- Replace with your actual file ID

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Labels
label = [ ... ]  # You can keep your label list as-is here

# Load disease mapping
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image_path = f'{temp_name}_{image.filename}'
        image.save(image_path)
        prediction = model_predict(image_path)
        return render_template('home.html', result=True, imagepath=f'/{image_path}', prediction=prediction)
    else:
        return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)