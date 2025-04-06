#!/usr/bin/env python
import os
import torch
import torchvision
import logging
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(num_classes: int):
        print("🔧 Initializing model...")
        print(f"→ Loading pretrained Faster R-CNN model for {num_classes} classes")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        print("→ Pretrained weights loaded")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        print(f"→ Replacing classifier head with {num_classes} output classes (input features: {in_features})")
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("✅ Model customized successfully")
        return model

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []

    def train(self, dataloader, num_epochs):
        print("🚀 Starting training loop...")
        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            print(f"\n🔁 Epoch {epoch+1}/{num_epochs}...")
            epoch_loss = 0.0
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"📉 Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            self.loss_history.append(avg_loss)

def load_or_train_model(model_file: str, num_classes: int, train_dataloader, device, num_epochs=5) -> torch.nn.Module:
    print(f"📂 Checking for model at: {model_file}")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    if os.path.exists(model_file):
        print("📦 Model file found. Loading saved model...")
        model = ModelFactory.get_model(num_classes)
        model.load_state_dict(torch.load(model_file))
        print("✅ Model loaded from disk")
    else:
        print("❌ Model not found. Starting training...")
        model = ModelFactory.get_model(num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        trainer = Trainer(model, optimizer, device)
        trainer.train(train_dataloader, num_epochs)

        torch.save(model.state_dict(), model_file)
        print(f"💾 Model trained and saved to: {model_file}")

    print("✅ Model ready to use")
    return model
