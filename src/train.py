#!/usr/bin/env python
import torch
from data_loader import get_dataloaders
from model import load_or_train_model
import evaluate
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
annotations_csv = os.path.join(BASE_DIR, "data", "annotations_coco.csv")
images_dir = os.path.join(BASE_DIR, "data", "training_nodules")
model_file = os.path.join(BASE_DIR, "models", "nodule_detector.pth")
img_dim = 256
batch_size =  4
train_split =  0.8
num_workers =  4
num_epochs = 40
num_classes = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(annotations_csv, images_dir, img_dim, batch_size, train_split, num_workers)
    model = load_or_train_model(model_file, num_classes, train_loader, device, num_epochs)
    print("Training completed.")
    evaluate.evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
