# src/visualization/plot_metrics.py
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(losses, savepath=None):
    plt.figure()
    plt.plot(losses, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_confusion_matrix(cm, labels, savepath=None):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
