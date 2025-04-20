import torch
import torchvision
import logging
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(num_classes: int):
        logger.info("🔧 Initializing Faster R-CNN model...")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.info("✅ Model customized successfully")
        return model