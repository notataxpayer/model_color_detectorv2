from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Konfigurasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Konfigurasi model
MODEL_URL = "https://github.com/notataxpayer/model_color_detectorv2/releases/download/v1/best_model_newest.h5"
MODEL_PATH = "best_model_newest.h5"
MIN_FILE_SIZE = 100000  # byte, sekitar 100 KB, asumsi model pasti lebih besar dari ini

# Fungsi download model dari GitHub Releases
def download_model(url, output_path):
    print("🔽 Downloading model...")
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("✅ Download complete.")

# Unduh model jika belum tersedia
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Validasi ukuran file model
if os.path.getsize(MODEL_PATH) < MIN_FILE_SIZE:
    raise Exception("❌ Model file too small or corrupt. Possibly HTML page.")

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

# Jalankan server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
