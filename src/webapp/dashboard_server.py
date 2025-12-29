from flask import Flask, request, jsonify, render_template
import os, json
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
    "belly_pain": "Your baby may have stomach pain. Try gentle tummy rub or burping.",
    "tired": "Your baby seems tired. Help them sleep comfortably.",
    "babbling": "Your baby is playful, interact and talk with them.",
    "discomfort": "Your baby is uncomfortable. Check diaper or clothing.",
    "cooing": "Your baby is cooing happily — no concern needed!",
    "unknown": "Unable to classify sound. Try recording again clearly."
}

# 🧹 Delete older audio files beyond 10
def cleanup_audio_folder(folder_path="sampletest", max_files=10):
    folder = Path(folder_path)
    audio_files = sorted(folder.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old_file in audio_files[max_files:]:
        try:
            old_file.unlink()
            print(f"🗑️ Deleted: {old_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to delete {old_file.name}: {e}")

@app.route('/')
def index():
    return render_template('live_graph.html')

@app.route('/api/results')
def get_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
                return jsonify(results)
        except json.JSONDecodeError:
            return jsonify([])
    return jsonify([])

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cleanup_audio_folder(UPLOAD_FOLDER)

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

    synonym_map = {
        "hungry": "hungry", "hunger": "hungry",
        "tired": "tired", "sleepy": "tired",
        "coo": "cooing", "cooing": "cooing",
        "babbling": "babbling",
        "discomfort": "discomfort",
        "bellypain": "belly_pain", "belly_pain": "belly_pain",
        "unknown": "unknown"
    }
    canonical = synonym_map.get(raw_label, "unknown")
    recommendation = RECOMMENDATIONS.get(canonical, RECOMMENDATIONS["unknown"])
    display_label = canonical.replace("_", " ").title()

    entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": display_label,
        "recommendation": recommendation
    }

    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []

    results.insert(0, entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return jsonify({"status": "ok", "prediction": display_label, "recommendation": recommendation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
