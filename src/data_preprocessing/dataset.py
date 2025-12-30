import os
import torch
from torch.utils.data import Dataset
import random
import librosa
import numpy as np
from src.config import SR, CLASSES
from src.data_preprocessing.feature_extractor import mel_spectrogram, to_image_tensor

class InfantVocalDataset(Dataset):
    def __init__(self, root_dir="data/raw", target_size=(224, 224), augment=False):
        self.audio_list = []
        self.labels = []
        self.target_size = target_size
        self.augment = augment
        self.labels_map = {idx: class_name for idx, class_name in enumerate(CLASSES)}

        for idx, class_name in enumerate(CLASSES):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if fname.lower().endswith((".wav", ".mp3")):
                    self.audio_list.append(os.path.join(class_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_path = self.audio_list[idx]
        label = self.labels[idx]

        # Load audio
        y, sr = librosa.load(audio_path, sr=SR)

        # -----------------------
        # 🔧 Data Augmentation
        # -----------------------
        if self.augment:
            aug_type = random.choice(["none", "pitch", "stretch", "noise"])
            if aug_type == "pitch":
                y = librosa.effects.pitch_shift(y, sr, n_steps=random.choice([-2, -1, 1, 2]))
            elif aug_type == "stretch":
                rate = random.uniform(0.8, 1.2)
                y = librosa.effects.time_stretch(y, rate)
            elif aug_type == "noise":
                noise = np.random.randn(len(y)) * 0.005
                y = y + noise

        # Compute mel-spectrogram
        S = mel_spectrogram_from_array(y, sr)  # New helper for raw array
        img = to_image_tensor(S, target_size=self.target_size)

        # Ensure channel dimension (C,H,W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # Add sequence dimension for LSTM: (1, C, H, W)
        img_seq = img.unsqueeze(0)
        return img_seq, label


# Helper to create mel-spectrogram from raw audio array
def mel_spectrogram_from_array(y, sr=SR):
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
