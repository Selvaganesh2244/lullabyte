# src/inference/realtime.py
"""
A very simple real-time demo using sounddevice to record short snippets
and run the model on them in a loop. This is demo code — for production
use buffering, threading, and low-latency optimizations.
"""
import sounddevice as sd
import numpy as np
import torch
from ..inference.predict import prepare_seq_from_file
from ..models.efficientnet_lstm import build_model
from ..config import SR, DEVICE
import time
import tempfile
import soundfile as sf
import argparse
from pathlib import Path
from ..data_preprocessing.feature_extractor import mel_spectrogram, to_image_tensor
from ..data_preprocessing.audio_loader import sliding_windows

def record_snippet(duration=2.0, sr=SR):
    print("Recording snippet...")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return rec.flatten()

def run_live(args):
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()
    try:
        while True:
            y = record_snippet(duration=args.window_seconds)
            windows = sliding_windows(y, sr=SR, window_seconds=args.window_seconds, hop_seconds=args.hop_seconds)
            imgs = []
            for w in windows:
                S = mel_spectrogram(w, sr=SR)
                img = to_image_tensor(S)
                imgs.append(img)
            seq = torch.stack(imgs, dim=0).unsqueeze(0).to(args.device)
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
            print(f"Pred: {pred} probs: {probs}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--window_seconds", type=float, default=2.0)
    parser.add_argument("--hop_seconds", type=float, default=1.0)
    args = parser.parse_args()
    run_live(args)
