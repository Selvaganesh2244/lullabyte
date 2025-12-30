from flask import Flask, request, jsonify, render_template
import os, json
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment 
from datetime import datetime
from src.inference.predict import predict_from_folder
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = "sampletest"
RESULTS_FILE = "results.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🍼 Recommendation mapping
RECOMMENDATIONS = {
    "hungry": "Your baby might be hungry. Try feeding them.",
    "tired": "Your baby seems tired. Help them sleep comfortably.",
    "cooing": "Your baby is cooing happily — no concern needed!",
    "babbling": "Your baby is playful, interact and talk with them.",
    "discomfort": "Your baby is uncomfortable. Check diaper or clothing.",
    "belly_pain": "Your baby may have stomach pain. Try gentle tummy rub or burping.",
    "unknown": "Unable to determine. Please check again later."
}

# 🧹 Delete older audio files beyond 10
def cleanup_audio_folder(folder, keep_last=10):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)], key=os.path.getctime)
    for old_file in files[:-keep_last]:
        try:
            os.remove(old_file)
            print(f"🗑️ Deleted: {os.path.basename(old_file)}")
        except Exception as e:
            print(f"⚠️ Failed to delete {old_file}: {e}")

@app.route('/')
def index():
    return render_template('live_graph.html')

# 📊 For dashboard history
@app.route('/api/results')
def get_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
                # Return newest first
                results = results[::-1]
                return jsonify(results)
        except json.JSONDecodeError:
            return jsonify([])
    return jsonify([])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Overwrite results.json with an empty list
        with open(RESULTS_FILE, "w") as f:
            json.dump([], f, indent=4)
        return jsonify({"success": True})
    except Exception as e:
        print("Error clearing history:", e)
        return jsonify({"success": False})

# 📈 Live Graph API
@app.route("/get_data")
def get_data():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"time": [], "state": [], "emotion_index": []})

    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"time": [], "state": [], "emotion_index": []})

    # Extract last 20 entries
    times, states, indices = [], [], []
    for entry in data[-20:]:
        times.append(entry.get("datetime", datetime.now().strftime("%H:%M:%S")))
        states.append(entry.get("result", "Unknown"))
        indices.append(entry.get("emotion_index", -1))

    return jsonify({"time": times, "state": states, "emotion_index": indices})


def preprocess_audio(y, sr):
    # Normalize amplitude
    y = librosa.util.normalize(y)
    # Pre-emphasis filter
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    # Optional: Noise reduction (can be commented out if too slow)
    y = nr.reduce_noise(y=y, sr=sr)
    return y

# 🎙️ Handle uploaded audio
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    original_filename = file.filename
    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # --- Step 1️⃣ Convert any audio to WAV first ---
        if original_filename.lower().endswith(".3gp"):
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_input.3gp")
            file.save(temp_path)
            audio = AudioSegment.from_file(temp_path, format="3gp")
            audio.export(filepath, format="wav")
            os.remove(temp_path)
            print(f"✅ Converted 3GP to WAV: {filepath}")
        else:
            # Directly save if already WAV
            file.save(filepath)

        # --- Step 2️⃣ Preprocess audio after conversion ---
        y, sr = librosa.load(filepath, sr=44100)
        y = preprocess_audio(y, sr)
        sf.write(filepath, y, sr)
        print(f"✅ Preprocessed and saved: {filepath}")

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        file.save(filepath)  # fallback raw save

    # --- Step 3️⃣ Cleanup old files ---
    cleanup_audio_folder(UPLOAD_FOLDER)

    # --- Step 4️⃣ Prediction ---
    try:
        predictions = predict_from_folder(filepath)
        if isinstance(predictions, dict):
            raw_label = predictions.get("label", "")
        elif isinstance(predictions, list) and len(predictions) > 0:
            first = predictions[0]
            raw_label = first.get("label", "") if isinstance(first, dict) else str(first)
        else:
            raw_label = str(predictions)
        raw_label = (raw_label or "").strip().lower()
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raw_label = "unknown"

    # --- Step 5️⃣ Normalize labels & recommend ---
    synonym_map = {
    "hungry": "hungry", "hunger": "hungry",
    "tired": "tired", "sleepy": "tired",
    "cooing": "cooing", "coo": "cooing",
    "babbling": "babbling",
    "discomfort": "discomfort",
    "belly_pain": "belly_pain", "bellypain": "belly_pain",
    "unknown": "unknown"
}

    raw_label = (raw_label or "").strip().lower().replace(" ", "_")
    canonical = synonym_map.get(raw_label, "unknown")
    recommendation = RECOMMENDATIONS.get(canonical, RECOMMENDATIONS["unknown"])
    display_label = canonical.replace("_", " ").title()

    # --- Step 6️⃣ Save result to JSON ---
    emotion_labels = ["Cooing", "Babbling", "Hungry", "Belly Pain", "Discomfort", "Tired"]

# Find correct y-axis index for plotting
    emotion_index = emotion_labels.index(display_label) if display_label in emotion_labels else -1

    entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": display_label,
        "recommendation": recommendation,
        "emotion_index": emotion_index
    }

    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []

    results.append(entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return jsonify({"status": "ok", "prediction": display_label, "recommendation": recommendation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
