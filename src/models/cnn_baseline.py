# src/models/cnn_baseline.py
import torch
import torch.nn as nn
from ..config import DEVICE

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq):
        # x_seq: (B, S, C, H, W) -> average pool per frame then average frames
        B, S, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * S, C, H, W)
        feat = self.features(x_flat)
        feat = feat.view(B, S, -1)  # (B, S, feat)
        feat = feat.mean(dim=1)     # average over time
        logits = self.classifier(feat)
        return logits

def build_cnn(num_classes=6):
    return SimpleCNN(num_classes=num_classes).to(DEVICE)
