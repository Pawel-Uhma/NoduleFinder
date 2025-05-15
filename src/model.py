import torch
import torchvision
import logging
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor
import torch.nn as nn
from loguru import logger

class ModelFactory:
    @staticmethod
    def get_model(
        num_classes: int,
        trainable_backbone_layers: int = 2,   # only C4 and C5 trainable
        detection_score_thresh: float = 0.1,  # drop low-score boxes
        detections_per_img: int = 50          # fewer high-quality detections
    ):
        logger.info("🔧 Initializing Faster R-CNN model with limited backbone tuning...")
        # load COCO‐pretrained weights but freeze early layers
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # replace the head for our # of classes
        in_feats = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        # re-init head weights for stable start
        nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
        nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, 0)
        nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.001)
        nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, 0)

        # tighten detection thresholds to reduce false positives
        model.roi_heads.score_thresh = detection_score_thresh
        model.roi_heads.detections_per_img = detections_per_img

        logger.info(
            "✅ Model ready: frozen backbone layers except last "
            f"{trainable_backbone_layers}, new head initialized, "
            f"score_thresh={detection_score_thresh}, max_detections={detections_per_img}"
        )
        return model
