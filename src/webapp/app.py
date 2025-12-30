from flask import Flask, request, jsonify, render_template
import os, json, threading
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from datetime import datetime
from src.inference.predict import predict_from_folder
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ================= CONFIG =================
UPLOAD_FOLDER = "sampletest"
RESULTS_FILE = "results.json"

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= RECOMMENDATIONS =================
RECOMMENDATIONS = {
    "hungry": "Your baby might be hungry. Try feeding them.",
    "tired": "Your baby seems tired. Help them sleep comfortably.",
    "cooing": "Your baby is cooing happily — no concern needed!",
    "babbling": "Your baby is playful, interact and talk with them.",
    "discomfort": "Your baby is uncomfortable. Check diaper or clothing.",
    "belly_pain": "Your baby may have stomach pain. Try gentle tummy rub or burping.",
    "unknown": "Unable to determine. Please check again later."
}

SYNONYM_MAP = {
    "hungry": "hungry",
    "tired": "tired",
    "sleepy": "tired",
    "coo": "cooing",
    "cooing": "cooing",
    "babbling": "babbling",
    "discomfort": "discomfort",
    "bellypain": "belly_pain",
    "belly_pain": "belly_pain",
}

EMOTION_LABELS = ["Cooing", "Babbling", "Hungry", "Belly Pain", "Discomfort", "Tired"]

# ================= UTILITIES =================
def cleanup_audio_folder(folder, keep_last=10):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getctime
    )
    for old_file in files[:-keep_last]:
        try:
            os.remove(old_file)
        except:
            pass

def preprocess_audio(y, sr):
    y = librosa.util.normalize(y)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    y, _ = librosa.effects.trim(y, top_db=20)
    y = nr.reduce_noise(y=y, sr=sr)
    return y

# ================= BACKGROUND TASK =================
def process_audio(filepath):
    try:
        y, sr = librosa.load(filepath, sr=44100)
        y = preprocess_audio(y, sr)
        sf.write(filepath, y, sr)

        predictions = predict_from_folder(filepath)
        raw_label = predictions.get("label", "unknown") if isinstance(predictions, dict) else "unknown"

    except Exception as e:
        print("❌ Processing error:", e)
        raw_label = "unknown"

    canonical = SYNONYM_MAP.get(raw_label.lower().replace(" ", "_"), "unknown")
    display_label = canonical.replace("_", " ").title()
    recommendation = RECOMMENDATIONS.get(canonical)

    emotion_index = EMOTION_LABELS.index(display_label) if display_label in EMOTION_LABELS else -1

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
        except:
            results = []

    results.append(entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    cleanup_audio_folder(UPLOAD_FOLDER)

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template("live_graph.html")

@app.route('/api/results')
def get_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return jsonify(json.load(f)[::-1])
    return jsonify([])

@app.route('/get_data')
def get_data():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"time": [], "state": [], "emotion_index": []})

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    last = data[-20:]
    return jsonify({
        "time": [x["datetime"] for x in last],
        "state": [x["result"] for x in last],
        "emotion_index": [x["emotion_index"] for x in last]
    })

# ================= UPLOAD (FAST RESPONSE) =================
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        if file.filename.lower().endswith(".3gp"):
            temp = filepath.replace(".wav", ".3gp")
            file.save(temp)
            audio = AudioSegment.from_file(temp, format="3gp")
            audio.export(filepath, format="wav")
            os.remove(temp)
        else:
            file.save(filepath)
    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({"error": "Upload failed"}), 500

    # 🔥 START BACKGROUND PROCESS
    threading.Thread(target=process_audio, args=(filepath,), daemon=True).start()

    # 🚀 INSTANT RESPONSE TO MOBILE
    return jsonify({"status": "uploaded"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
