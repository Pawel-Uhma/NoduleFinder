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
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to {filename}")



