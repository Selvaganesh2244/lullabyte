# src/data_preprocessing/feature_extractor.py
import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from src.config import SR, N_MELS, HOP_LENGTH, N_FFT

# feature_extractor.py
def mel_spectrogram(y, sr=SR, from_file=True):
    """
    y: either a file path (from_file=True) or a numpy array of audio (from_file=False)
    """
    import librosa
    import numpy as np

    if from_file:
        y, sr = librosa.load(y, sr=sr)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db



def to_image_tensor(S, target_size=(224, 224)):
    """
    Convert a spectrogram numpy array to a 3-channel PyTorch tensor.
    """
    import torch
    import torch.nn.functional as F

    # Convert NumPy array to PyTorch tensor
    tensor = torch.tensor(S, dtype=torch.float32)

    # Normalize (optional, helps training)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)

    # Ensure channel dimension (C,H,W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # (1,H,W)

    # Repeat channels to make 3-channel image
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)  # (3,H,W)

    # Resize to target_size
    tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

    return tensor



def process_all_audio(raw_dir="data/raw", spec_dir="data/spectrograms", target_size=(224,224)):
    """
    Convert all audio files in raw_dir → 3-channel spectrogram tensors saved as .pt
    """
    os.makedirs(spec_dir, exist_ok=True)
    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_dir = os.path.join(spec_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)
        print(f"📂 Processing class: {class_name}")

        for idx, fname in enumerate(os.listdir(class_path)):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue
            audio_path = os.path.join(class_path, fname)
            spec_tensor = to_image_tensor(mel_spectrogram(audio_path), target_size=target_size)
            save_path = os.path.join(save_class_dir, f"{class_name}_{idx}.png")
            torch.save(spec_tensor, save_path)

        print(f"✅ Finished class {class_name}")
    
if __name__ == "__main__":
    raw_dir = "data/raw"  # Change if your raw audio is stored elsewhere
    spec_dir = "data/spectrograms"
    os.makedirs(spec_dir, exist_ok=True)
    process_all_audio(raw_dir, spec_dir)


