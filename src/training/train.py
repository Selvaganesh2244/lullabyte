# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

# ✅ Local imports (fixed paths)
from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
from src.data_preprocessing.dataset import InfantVocalDataset
from src.models.efficientnet_lstm import build_model
from src.training.utils import save_checkpoint, compute_metrics


def collate_fn(batch):
    """
    Collate function for batching.
    Dataset returns (1, C, H, W), label.
    We just stack into (B, 1, C, H, W).
    """
    seqs, labels = zip(*batch)

    # stack tensors into shape (B, 1, C, H, W)
    seqs_tensor = torch.stack(seqs, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return seqs_tensor, labels_tensor


def train(args):
    ds = InfantVocalDataset(args.data_root)
    num_classes = len(ds.labels_map)

    # ✅ Build model
    model = build_model(num_classes=num_classes)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # ⚠️ keep 0 on Windows
        collate_fn=collate_fn
    )

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        losses = []

        for x_seq, y in tqdm(dl, desc=f"Train Epoch {epoch+1}/{args.epochs}"):
            x_seq = x_seq.to(args.device)
            y = y.to(args.device)

            logits = model(x_seq)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1} loss {avg_loss:.4f}")

        # ✅ quick evaluation
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x_seq, y in dl:
                x_seq = x_seq.to(args.device)
                logits = model(x_seq)
                p = logits.argmax(dim=1).cpu().numpy()
                ys.extend(y.numpy().tolist())
                preds.extend(p.tolist())

        metrics = compute_metrics(ys, preds)
        print("Train metrics:", metrics)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            save_checkpoint(
                {"model_state": model.state_dict(), "epoch": epoch+1},
                name=f"best_epoch_{epoch+1}.pth"
            )

    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    train(args)
