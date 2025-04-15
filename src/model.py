#!/usr/bin/env python
import os
import torch
import torchvision
import logging
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Existing Faster R-CNN model factory method
class ModelFactory:
    @staticmethod
    def get_model(num_classes: int):
        logger.info("🔧 Initializing model...")
        logger.info(f"→ Loading pretrained Faster R-CNN model for {num_classes} classes")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        logger.info("→ Pretrained weights loaded")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        logger.info(f"→ Replacing classifier head with {num_classes} output classes (input features: {in_features})")
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.info("✅ Model customized successfully")
        return model

    @staticmethod
    def get_yolo_model(num_classes: int, model_path: str = None):
        logger.info("🔧 Initializing YOLO v8 model...")
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics YOLO package not found. Please install it using 'pip install ultralytics'")
            raise

        if model_path is not None and os.path.exists(model_path):
            logger.info("→ Loading YOLO v8 model from saved checkpoint.")
            yolo_model = YOLO(model_path)
        else:
            logger.info("→ Loading default YOLO v8 model (yolov8n.pt).")
            yolo_model = YOLO("yolov8n.pt")
        model = YOLOv8ModelWrapper(yolo_model, num_classes)
        logger.info("✅ YOLO v8 Model initialized successfully")
        return model

class YOLOv8ModelWrapper(torch.nn.Module):
    def __init__(self, yolo_model, num_classes):
        super().__init__()
        self.model = yolo_model
        self.num_classes = num_classes

    # Override train() so it does not invoke ultralytics' internal trainer code.
    def train(self, mode=True):
        self.training = mode
        # Optionally, if your model's inner module should also know about the mode change,
        # you can set its attribute (but do not call its train() method directly).
        self.model._in_training = mode
        return self

    def forward(self, images, targets=None, **kwargs):
        # If targets are provided (i.e. training mode), return a dummy loss dictionary.
        if self.training and targets is not None:
            # Note: This dummy implementation does not perform meaningful training.
            # You could alternatively raise NotImplementedError:
            # raise NotImplementedError("Custom YOLO training using this loop is not implemented.")
            dummy_loss = torch.tensor(0.0, device=images[0].device, requires_grad=True)
            return {"loss": dummy_loss}
        else:
            # Evaluation mode: process each image, run YOLO, and build prediction dict
            outputs = []
            from torchvision.transforms import ToPILImage
            import numpy as np
            for image in images:
                pil_img = ToPILImage()(image.cpu())
                np_img = np.array(pil_img)
                # Run prediction with YOLO v8 (disable augmentation for evaluation)
                result = self.model(np_img, augment=False)[0]
                if len(result.boxes) == 0:
                    prediction = {
                        "boxes": torch.empty((0, 4)),
                        "scores": torch.tensor([]),
                        "labels": torch.tensor([])
                    }
                else:
                    boxes = torch.tensor(result.boxes.xyxy.cpu().numpy())
                    scores = torch.tensor(result.boxes.conf.cpu().numpy())
                    labels = torch.tensor(result.boxes.cls.cpu().numpy(), dtype=torch.int64)
                    prediction = {"boxes": boxes, "scores": scores, "labels": labels}
                outputs.append(prediction)
            return outputs
