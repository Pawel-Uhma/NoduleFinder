import torchvision
from torchvision.models.detection import FasterRCNN                    # core detector
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn


class ModelFactory:
    @staticmethod
    def get_model(
        num_classes: int,
        trainable_backbone_layers: int = 2,
        detection_score_thresh: float = 0.1,
        detections_per_img: int = 50,
    ):
        """
        Faster R-CNN with a ResNet-18 + FPN backbone.

        • Uses ImageNet-1K weights for the backbone (no COCO detector
          checkpoint exists for ResNet-18), so you usually start
          fine-tuning from epoch 0.
        • `trainable_backbone_layers` behaves exactly like the
          ResNet-50 helper: 0 → freeze all, 5 → train all. 
        """

        # -----------------------------------------------------------
        # 1. Backbone (ResNet-18 + FPN)
        # -----------------------------------------------------------
        try:
            # torchvision 0.13 + : use the new "weights=" API
            weights_backbone = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet_fpn_backbone(
                backbone_name="resnet18",
                weights=weights_backbone,
                trainable_layers=trainable_backbone_layers,
            )  # :contentReference[oaicite:0]{index=0}
        except TypeError:
            # Older torchvision (<0.13): falls back to the legacy "pretrained=True"
            backbone = resnet_fpn_backbone(
                backbone_name="resnet18",
                pretrained=True,
                trainable_layers=trainable_backbone_layers,
            )

        # `resnet_fpn_backbone` always exposes the number of channels it outputs
        # via the `out_channels` attribute (256 by default), which FasterRCNN needs
        # to size its first heads correctly.
        model = FasterRCNN(backbone=backbone, num_classes=num_classes)

        # -----------------------------------------------------------
        # 2. Replace and (optionally) re-initialise the detection head
        # -----------------------------------------------------------
        in_feats = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
        nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, 0)
        nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.001)
        nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, 0)

        # -----------------------------------------------------------
        # 3. Post-processing tweaks (same as your original code)
        # -----------------------------------------------------------
        model.roi_heads.score_thresh = detection_score_thresh
        model.roi_heads.detections_per_img = detections_per_img

        return model
