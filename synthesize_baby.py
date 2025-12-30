import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F

# --------
# 1. Dataset: returns mel-spectrograms and waveform pairs
# --------
class BabyMelDataset(Dataset):
    def __init__(self, folder, sample_rate=16000, sample_len=98304, n_mels=80, n_fft=1024, hop_length=256):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        self.sample_len = sample_len
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, center=True)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        wav = wav.mean(dim=0)
        wav = F.pad(wav, (0, max(0, self.sample_len - wav.shape[-1])))
        wav = wav[:self.sample_len]
        wav = (wav - wav.mean()) / (wav.std() + 1e-7) # normalize
        mel = self.mel_transform(wav)
        return mel, wav.unsqueeze(0)

# --------
# 2. Generator: U-Net style (MelGAN-inspired but simplified)
# --------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_mels=80, mel_steps=384):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_mels * mel_steps)
        self.unet = nn.Sequential(
            nn.Conv1d(n_mels, 128, 7, padding=3), nn.ReLU(),
            nn.Conv1d(128, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, n_mels, 7, padding=3), nn.Tanh(),
        )
    def forward(self, z):
        batch = z.size(0)
        x = self.fc(z).view(batch, 80, -1)
        x = self.unet(x)
        return x

# --------
# 3. Discriminator: PatchGAN on Mel-spectrogram
# --------
class Discriminator(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.cnet = nn.Sequential(
            nn.Conv1d(n_mels, 128, 7, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 7, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, 7, padding=3)
        )
    def forward(self, x):
        return self.cnet(x).mean(dim=-1)

# --------
# 4. Spectral (feature) loss
# --------
def spectral_loss(gen_mel, real_mel):
    # gen_mel and real_mel are already Mel-spectrograms
    min_time = min(gen_mel.size(-1), real_mel.size(-1))
    gen_mel = gen_mel[..., :min_time]
    real_mel = real_mel[..., :min_time]
    return F.l1_loss(gen_mel, real_mel)



# --------
# 5. Griffin-Lim vocoder to synthesize waveform from mel
# --------
def mel_to_audio(mel, n_fft=1024, hop_length=256, n_iter=32, n_mels=80):
    # Inverse Mel
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft//2 + 1,
        n_mels=n_mels
    )(mel)
    
    # Griffin-Lim
    waveform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter
    )(inv_mel)
    
    return waveform


# --------
# 6. Training loop
# --------
def train(dataset_path, epochs=50, batch_size=8, latent_dim=100, lr=0.0002, sample_rate=16000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        n_mels=80,
        win_length=1024,
        hop_length=256
    ).to(device)
    dataset = BabyMelDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    for epoch in range(epochs):
        for real_mel, real_wav in loader:
            real_mel = real_mel.to(device)
            # --- Train D
            z = torch.randn(real_mel.size(0), latent_dim, device=device)
            fake_mel = G(z).detach()
            lossD = (
                F.binary_cross_entropy_with_logits(D(real_mel), torch.ones(real_mel.size(0), 1, device=device)) +
                F.binary_cross_entropy_with_logits(D(fake_mel), torch.zeros(real_mel.size(0), 1, device=device))
            )
            optD.zero_grad(); lossD.backward(); optD.step()
            # --- Train G
            z = torch.randn(real_mel.size(0), latent_dim, device=device)
            fake_mel = G(z)
            adv_loss = F.binary_cross_entropy_with_logits(D(fake_mel), torch.ones(real_mel.size(0), 1, device=device))
            spec_loss = spectral_loss(fake_mel, real_mel)
            lossG = adv_loss + 2.0 * spec_loss # weight spectral
            optG.zero_grad(); lossG.backward(); optG.step()
        print(f"Epoch {epoch+1}/{epochs} LossD={lossD.item():.4f} LossG={lossG.item():.4f}")
        # -- Save sample audio every 10 epochs
        if (epoch+1)%10 == 0:
            z_sample = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                gen_mel = G(z_sample).cpu().squeeze(0)
                gen_wav = mel_to_audio(gen_mel)
                torchaudio.save(f"samples/sample_epoch{epoch+1}.wav", gen_wav.unsqueeze(0), sample_rate)
    torch.save(G.state_dict(), "generator_final.pth")
    print("Model saved as generator_final.pth")
    return G

# --------
# 7. Generate long audio (joined baby sounds)
# --------
def generate_long_audio(G, latent_dim=100, minutes=7, sample_rate=16000):
    device = next(G.parameters()).device
    os.makedirs("samples", exist_ok=True)
    num_clips = int((minutes*60)/6)
    clips = []
    for i in range(num_clips):
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            gen_mel = G(z).cpu().squeeze(0)
            gen_wav = mel_to_audio(gen_mel, n_mels=80)
            clips.append(gen_wav)
            torchaudio.save(f"samples/generated_part{i+1}.wav", gen_wav.unsqueeze(0), sample_rate)
    full_audio = torch.cat(clips, dim=-1)
    torchaudio.save("samples/long_baby_audio.wav", full_audio.unsqueeze(0), sample_rate)
    print("✅ Saved 7-minute audio at samples/long_baby_audio.wav")

# --------
# Main
# --------
if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    G = train("./data/raw/belly_pain", epochs=50)
    generate_long_audio(G)
