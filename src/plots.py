#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
from config_loader import load_config

def plot_loss(training_loss, val_loss=None, title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", filename="./plots/loss_plot.png"):
    plt.figure()
    plt.plot(training_loss, label="Training Loss", marker='o')
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss", marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to {filename}")


def plot_map_accuracy(map_history, accuracy_history, title="Validation mAP and Accuracy", xlabel="Epoch", ylabel="Value", filename="./plots/map_accuracy_plot.png"):
    plt.figure()
    plt.plot(map_history, label="mAP", marker='o')
    plt.plot(accuracy_history, label="Accuracy", marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to {filename}")

def plot_iou_trend(mean_iou_history, title="Mean IoU Trend", xlabel="Epoch", ylabel="Mean IoU", filename="./plots/mean_iou_trend.png"):
    plt.figure()
    plt.plot(mean_iou_history, label="Mean IoU", marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to {filename}")

