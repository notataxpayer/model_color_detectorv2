from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import zipfile
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Konfigurasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Konfigurasi model
MODEL_URL = "https://drive.google.com/uc?id=1J0jRFjJNRAlMKgxw642irZ6FNxPEslqT"
MODEL_ZIP = "best_model.zip"
MODEL_PATH = "best_model.h5"

# Unduh & ekstrak model jika belum tersedia
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model ZIP from Google Drive...")
    gdown.download(MODEL_URL, MODEL_ZIP, quiet=False, use_cookies=True)
    if not os.path.exists(MODEL_ZIP):
        raise Exception("❌ Failed to download model ZIP.")
    print("📦 Extracting model...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall()

# Validasi ukuran model
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
    raise Exception("❌ Extracted model file too small or corrupt.")

# Load model
model = load_model(MODEL_PATH)
target_size = (150, 150)
class_labels = ['autumn', 'spring', 'summer', 'winter']

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess gambar
    img = load_img(filepath, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4)
    })

# Jalankan app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
