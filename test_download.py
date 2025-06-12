# test_download.py
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Ikce9LHNpi1WjthcA3xcHDchhdlcKYcO"
MODEL_PATH = "best_model.h5"

# Download
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, use_cookies=True)

# Cek ukuran
if os.path.exists(MODEL_PATH):
    size = os.path.getsize(MODEL_PATH)
    print("✅ File downloaded. Size:", size, "bytes")
    if size < 100000:
        print("❌ Warning: file may be corrupted (too small).")
else:
    print("❌ File not found after download.")
