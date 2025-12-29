import librosa
import numpy as np
import noisereduce as nr
from ..config import SR

def preprocess_audio(y, sr=SR):
    """Preprocess audio before saving."""
    y = librosa.util.buf_to_float(y, n_bytes=2)

    # 1️⃣ Noise reduction
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)

    # 2️⃣ Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3️⃣ Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 4️⃣ Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    return y
