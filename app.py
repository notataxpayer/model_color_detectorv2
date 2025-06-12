from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Konfigurasi Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Info Model ===
MODEL_URL = "https://github.com/notataxpayer/model_color_detectorv2/releases/download/v1/best_model_newest.h5"
MODEL_PATH = "best_model_newest.h5"
TARGET_SIZE = (150, 150)
CLASS_LABELS = ['autumn', 'spring', 'summer', 'winter']

# === Unduh model jika belum ada ===
if not os.path.exists(MODEL_PATH):
    print("📦 Downloading model with wget...")
    os.system(f"wget --content-disposition {MODEL_URL} -O {MODEL_PATH}")

# === Validasi model ===
if not os.path.exists(MODEL_PATH):
    raise Exception("❌ Failed to download model file.")

# Cek ukuran file
file_size = os.path.getsize(MODEL_PATH)
print(f"📁 Model file size: {file_size} bytes")

# Cek konten awal file (pastikan bukan HTML)
with open(MODEL_PATH, "rb") as f:
    preview = f.read(256)
    if b"<html" in preview or file_size < 50000:
        raise Exception("❌ Model file too small or corrupt. Possibly HTML page.")

# === Load model ===
print("✅ Loading model...")
model = load_model(MODEL_PATH)

# === Endpoint prediksi ===
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
    img = load_img(filepath, target_size=TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_label = CLASS_LABELS[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4)
    })

# === Run app ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
