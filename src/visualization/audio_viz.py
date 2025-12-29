# src/visualization/audio_viz.py
import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(y, sr):
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.show()

def plot_spectrogram(S_db):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_db, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()
