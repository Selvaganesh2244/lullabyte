# src/data_preprocessing/audio_loader.py
import librosa
import numpy as np
from pathlib import Path
from src.config import SR, WINDOW_SECONDS, HOP_SECONDS

def load_audio(filepath, sr=SR, mono=True):
    y, _sr = librosa.load(str(filepath), sr=sr, mono=mono)
    return y

def sliding_windows(y, sr=SR, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
    wlen = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    if len(y) <= wlen:
        # pad
        pad_len = wlen - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")
        return [y]
    windows = []
    for start in range(0, max(1, len(y) - wlen + 1), hop):
        windows.append(y[start:start + wlen])
    # last segment if not covered
    if (len(y) - wlen) % hop != 0 and (len(y) > wlen):
        windows.append(y[-wlen:])
    return windows
