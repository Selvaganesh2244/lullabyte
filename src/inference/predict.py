import torch
import numpy as np
from pathlib import Path
from src.models.efficientnet_lstm import build_model
from src.data_preprocessing.audio_loader import sliding_windows
from src.data_preprocessing.feature_extractor import mel_spectrogram, to_image_tensor
from src.config import SR, DEVICE, CLASSES
import librosa


# ------------------------------------------------
# Split Audio into 7-second segments (fixed)
# ------------------------------------------------
def split_audio(file_path, segment_duration=7.0):
    """Split audio into 7-second segments."""
    y, sr = librosa.load(file_path, sr=SR)
    segment_samples = int(segment_duration * sr)
    clips = []
    total_segments = int(np.ceil(len(y) / segment_samples))

    for i in range(total_segments):
        start = int(i * segment_samples)
        end = int(min(len(y), (i + 1) * segment_samples))
        clip = y[start:end]
        clips.append((clip, f"{Path(file_path).stem}_part{i+1}.wav"))
    return clips, sr


# ------------------------------------------------
# Convert waveform → tensor for model
# ------------------------------------------------
def prepare_seq_from_waveform(y, sr=SR, window_seconds=2.0, hop_seconds=1.0, target_size=(224, 224)):
    """Convert waveform array into a sequence tensor."""
    windows = sliding_windows(y, sr=sr, window_seconds=window_seconds, hop_seconds=hop_seconds)
    imgs = []
    for w in windows:
        S = mel_spectrogram(w, sr=sr, from_file=False)
        img = to_image_tensor(S, target_size=target_size)
        imgs.append(img)
    if len(imgs) == 0:
        return None
    seq = torch.stack(imgs, dim=0).unsqueeze(0)
    return seq


# ------------------------------------------------
# Label Recommendations
# ------------------------------------------------
def get_recommendation(label):
    """Provide recommendations for each baby sound."""
    recommendations = {
        "cooing": "Your baby is happy and comfortable.",
        "babbling": "Your baby is playful, interact and talk with them.",
        "hungry": "Feed your baby. They might be hungry.",
        "discomfort": "Your baby may be uncomfortable. Check diaper or clothing.",
        "belly_pain": "Your baby may have stomach pain. Try gentle tummy rub or burping.",
        "tired": "Your baby is sleepy. Try soothing and putting them to sleep."
    }
    return recommendations.get(label.lower(), "Unable to determine baby's condition.")


# ------------------------------------------------
# Predict from a single waveform
# ------------------------------------------------
def predict_single_waveform(y, model, device):
    """Predict label from a single waveform array."""
    seq = prepare_seq_from_waveform(y)
    if seq is None:
        return None, None
    seq = seq.to(device)
    with torch.no_grad():
        logits = model(seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        label = CLASSES[pred_idx]
    return label, probs


# ------------------------------------------------
# Predict from file or folder
# ------------------------------------------------
def predict_from_folder(path="sampletest", checkpoint_path="checkpoints/best_epoch_30.pth", device=DEVICE):
    """
    Predict baby cry labels from a single audio file or all audio files in a folder.
    Returns list of predictions with file, label, probability, and recommendation.
    """
    device = torch.device(device if torch.cuda.is_available() and device.lower() == "cuda" else "cpu")

    # Load model
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    path = Path(path)
    results = []

    # If path is a single audio file
    if path.is_file() and path.suffix.lower() in [".wav", ".mp3"]:
        audio_files = [path]
    else:
        audio_files = list(path.glob("*.wav")) + list(path.glob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in {path}")
        return []

    for file_path in audio_files:
        clips, sr = split_audio(file_path, segment_duration=7.0)
        print(f"\n🔊 Processing {file_path.name} ({len(clips)} clips)")

        file_probs = []
        for clip, clip_name in clips:
            label, probs = predict_single_waveform(clip, model, device)
            if label is None:
                continue
            file_probs.append((label, probs))

        if not file_probs:
            results.append({"file": file_path.name, "label": "unknown", "recommendation": get_recommendation("unknown")})
            continue

        # Majority voting for multi-clip file
        labels = [lbl for lbl, _ in file_probs]
        final_label = max(set(labels), key=labels.count)

        recommendation = get_recommendation(final_label)
        results.append({
            "file": file_path.name,
            "label": final_label,
            "recommendation": recommendation
        })

        print(f"✅ {file_path.name} → {final_label}")

    return results
