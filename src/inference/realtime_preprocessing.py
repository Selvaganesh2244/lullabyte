import pyaudio
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from collections import deque
import noisereduce as nr
import time

from src.data_preprocessing.feature_extractor import pre_emphasis, extract_mfcc


class RealTimeAudioProcessor:
    def __init__(self, sr=16000, chunk_duration=1.0, model=None, device='cpu'):
        """
        Real-time infant cry audio processor.
        - sr: sample rate
        - chunk_duration: audio chunk size in seconds
        - model: trained model (EfficientNet-LSTM)
        - device: cpu or cuda
        """
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.model = model
        self.device = device
        self.stream = None
        self.audio_buffer = deque(maxlen=self.chunk_samples * 5)

    def start_stream(self):
        """Start the PyAudio stream."""
        p = pyaudio.PyAudio()
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
        )
        print("🎙️ Listening for infant cries in real-time...")

    def stop_stream(self):
        """Stop audio stream safely."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("🛑 Stream stopped.")

    def read_audio_chunk(self):
        """Capture live audio chunk."""
        data = self.stream.read(self.chunk_samples, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def preprocess_chunk(self, y):
        """Apply preprocessing: noise reduction, pre-emphasis, MFCC extraction."""
        y = nr.reduce_noise(y=y, sr=self.sr)
        y = pre_emphasis(y)
        mfcc = extract_mfcc(y, self.sr, n_mfcc=40)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(1).to(self.device)
        return mfcc

    def predict(self, mfcc):
        """Run prediction through model and return label + confidence."""
        with torch.no_grad():
            logits = self.model(mfcc)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        return pred_class, confidence

    def run(self, label_map):
        """
        Start real-time inference loop.
        label_map: dict -> {0: 'Babbling', 1: 'Discomfort', ...}
        """
        self.start_stream()

        try:
            while True:
                chunk = self.read_audio_chunk()
                self.audio_buffer.extend(chunk)

                # Process 1-sec audio chunk
                if len(self.audio_buffer) >= self.chunk_samples:
                    y = np.array(self.audio_buffer)
                    mfcc = self.preprocess_chunk(y)

                    pred_class, conf = self.predict(mfcc)
                    label = label_map.get(pred_class, "Unknown")

                    print(f"[{time.strftime('%H:%M:%S')}] 🎧 {label} ({conf*100:.1f}%)")

                    time.sleep(1.0)  # prevent over-polling
        except KeyboardInterrupt:
            self.stop_stream()
            print("✅ Real-time monitoring stopped.")
