import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing.dataset import InfantVocalDataset
from src.models.efficientnet_lstm import build_model
from src.training.utils import compute_metrics
import argparse


def collate_fn(batch):
    seqs, labels = zip(*batch)
    max_s = max([s.shape[0] for s in seqs])
    B = len(seqs)
    C, H, W = seqs[0].shape[1:]
    out = torch.zeros((B, max_s, C, H, W), dtype=seqs[0].dtype)

    for i, s in enumerate(seqs):
        L = s.shape[0]
        out[i, :L] = s
        if L < max_s:
            out[i, L:] = s[-1].unsqueeze(0).repeat(max_s - L, 1, 1, 1)

    labels = torch.tensor(labels, dtype=torch.long)
    return out, labels


def plot_confusion_matrix(cm, labels, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    print(f"\n📁 Confusion Matrix saved as: {save_path}")
    plt.show()


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = InfantVocalDataset(args.data_root)
    num_classes = len(ds.labels_map)

    # Convert label names to string list
    label_names = list(ds.labels_map.values())
    label_names = [str(c) for c in label_names]

    print("\n🧠 Loading model...\n")
    model = build_model(num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    print("🔍 Evaluating model...")
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for x_seq, y in dl:
            x_seq = x_seq.to(device)
            logits = model(x_seq)
            p = logits.argmax(dim=1).cpu().numpy()

            ys.extend(y.numpy().tolist())
            preds.extend(p.tolist())

    # Metrics
    acc = accuracy_score(ys, preds)
    precision = precision_score(ys, preds, average='macro', zero_division=0)
    recall = recall_score(ys, preds, average='macro', zero_division=0)
    f1 = f1_score(ys, preds, average='macro', zero_division=0)

    print("\n📊 Evaluation Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Classification Report
    report = classification_report(ys, preds, target_names=label_names, zero_division=0)
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(ys, preds)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(args)
