# src/training/utils.py
import torch
import os
from ..config import CHECKPOINTS_DIR, DEVICE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import torch.nn.functional as F

def save_checkpoint(state, name="checkpoint.pth"):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / name
    torch.save(state, path)
    return path

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1_macro, "confusion_matrix": cm}
