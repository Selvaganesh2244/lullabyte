# src/models/efficientnet_lstm.py
import torch
import torch.nn as nn
import timm
from ..config import EFF_NET, EMBED_DIM, LSTM_HIDDEN, LSTM_LAYERS, NUM_CLASSES, DEVICE

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name=EFF_NET, embed_dim=EMBED_DIM, pretrained=True):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # output feature dim depends on model; create a linear to embed_dim
        feat_dim = self.net.num_features
        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        feat = self.net(x)  # (B, feat_dim)
        emb = self.fc(feat)
        return emb

class EfficientNetLSTM(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, num_classes=NUM_CLASSES, bidirectional=True):
        super().__init__()
        self.backbone = EfficientNetBackbone()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq):
        """
        x_seq: list or tensor
        If tensor: shape (B, seq_len, C, H, W)
        We'll flatten batch*seq to process via backbone, then reshape
        """
        B, S, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * S, C, H, W)
        emb = self.backbone(x_flat)  # (B*S, embed_dim)
        emb = emb.view(B, S, -1)     # (B, S, embed_dim)
        lstm_out, _ = self.lstm(emb)  # (B, S, hidden*directions)
        # pool over time dimension
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, hidden, S)
        pooled = self.pool(lstm_out).squeeze(-1)  # (B, hidden)
        logits = self.classifier(pooled)
        return logits

def build_model(num_classes=NUM_CLASSES):
    model = EfficientNetLSTM(num_classes=num_classes)
    return model.to(DEVICE)
