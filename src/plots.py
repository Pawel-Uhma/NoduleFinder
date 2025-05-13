import os
import matplotlib.pyplot as plt
from config_loader import load_config

def plot_loss(training_loss, val_loss=None, title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", dir="./plots/"):
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, "loss.png")

    epochs = range(1, len(training_loss) + 1)
    plt.figure()
    plt.plot(epochs, training_loss, label="Training Loss", marker="o")

    # --- NEW: keep both curves aligned even if val list is shorter -------------
    if val_loss:
        val_epochs = range(1, len(val_loss) + 1)
        plt.plot(val_epochs, val_loss, label="Validation Loss", marker="o")
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Loss plot saved to {filename}")


def plot_map_accuracy(map_history, accuracy_history, title="Validation mAP and Accuracy", xlabel="Epoch", ylabel="Value", dir="./plots/"):
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, "mAP_accuracy.png")

    plt.figure()
    plt.plot(map_history, label="mAP@[.50:.95]", marker='o')
    plt.plot(accuracy_history, label="Accuracy", marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ mAP vs Accuracy plot saved to {filename}")


def plot_iou_trend(mean_iou_history, title="Mean IoU Trend", xlabel="Epoch", ylabel="Mean IoU", dir="./plots/"):
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, "IoU_trend.png")

    plt.figure()
    plt.plot(mean_iou_history, label="Mean IoU", marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ IoU trend plot saved to {filename}")


def plot_precision_recall_curve(precision, recall, dir, filename="pr_curve.png"):
    os.makedirs(dir, exist_ok=True)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"✅ PR curve saved to {path}")


def plot_f1_curve(f1, thresholds, dir, filename="f1_curve.png"):
    os.makedirs(dir, exist_ok=True)
    plt.figure()
    plt.plot(thresholds, f1[:-1])
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Confidence Threshold")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"✅ F1 curve saved to {path}")


def plot_roc_curve(fpr, tpr, roc_auc, dir, filename="roc_curve.png"):
    os.makedirs(dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"✅ ROC curve saved to {path}")


def plot_map_vs_iou(per_iou_map, dir="./plots/", filename="map_vs_iou.png"):
    os.makedirs(dir, exist_ok=True)
    thresholds = list(per_iou_map.keys())
    aps = list(per_iou_map.values())
    plt.figure()
    plt.plot(thresholds, aps, marker='o')
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.title("AP vs IoU Threshold")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"✅ AP vs IoU plot saved to {path}")
