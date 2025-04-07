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

