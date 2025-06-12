from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Konfigurasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================
# Konfigurasi model dari GDrive
# ============================
MODEL_PATH = "best_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1J0jRFjJNRAlMKgxw642irZ6FNxPEslqT"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, use_cookies=True)

# Debug ukuran file
if os.path.exists(MODEL_PATH):
    size = os.path.getsize(MODEL_PATH)
    print("📏 Model size:", size, "bytes")
    if size < 100000:
        with open(MODEL_PATH, "r", errors="ignore") as f:
            print("🔍 File content (truncated):\n", f.read(500))
        raise Exception("❌ Model file too small or corrupt. Possibly HTML page.")
else:
    raise Exception("❌ Model file failed to download.")

# ====================
# Load model
# ====================
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise

# Konfigurasi input
target_size = (150, 150)
class_labels = ['autumn', 'spring', 'summer', 'winter']

# ====================
# Endpoint prediksi
# ====================
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

# ====================
# Run server
# ====================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
