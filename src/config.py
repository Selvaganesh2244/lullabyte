import os
import torch
from pathlib import Path

# -----------------------
# Audio / feature parameters
# -----------------------
SR = 44100           # Sample rate
N_MELS = 128         # Number of Mel bands
N_FFT = 2048         # FFT size
HOP_LENGTH = 512     # Hop length for spectrogram

# Sliding window for inference
WINDOW_SECONDS = 2.0  # Window size in seconds
HOP_SECONDS = 1.0     # Hop size in seconds

# -----------------------
# Training parameters
# -----------------------
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Classes / labels
# -----------------------
CLASSES = ["cooing", "babbling", "hunger", "discomfort", "belly_pain", "tired"]
NUM_CLASSES = len(CLASSES)

# -----------------------
# Model parameters
# -----------------------
EFF_NET = "tf_efficientnet_b0"
EMBED_DIM = 128
LSTM_HIDDEN = 64
LSTM_LAYERS = 1

# -----------------------
# Paths
# -----------------------
CHECKPOINTS_DIR = Path("checkpoints")   # Folder to save model checkpoints
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)  # Create folder if it doesn't exist
