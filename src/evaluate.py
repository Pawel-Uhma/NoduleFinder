#!/usr/bin/env python
import os, torch, numpy as np, logging, shutil
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from plots import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2Area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def compute_map(model, dataloader, device, iou_threshold=0.5, confidence_threshold=0.5):
    model.eval()
    detections = []  
    total_gts = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                total_gts += 1 
                gt_box = target["boxes"][0].cpu().numpy()
                if len(pred["boxes"]) > 0:
                    scores = pred["scores"].cpu().numpy()
                    boxes = pred["boxes"].cpu().numpy()
                    best_idx = scores.argmax()
                    best_score = scores[best_idx]
                    if best_score < confidence_threshold:
                        detections.append({'score': best_score, 'is_tp': 0})
                    else:
                        pred_box = boxes[best_idx]
                        iou = compute_iou(gt_box, pred_box)
                        is_tp = 1 if iou >= iou_threshold else 0
                        detections.append({'score': best_score, 'is_tp': is_tp})
                else:
                    detections.append({'score': 0.0, 'is_tp': 0})
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    cum_tp = 0
    cum_fp = 0
    precisions = []
    recalls = []
    for det in detections:
        if det['is_tp']:
            cum_tp += 1
        else:
            cum_fp += 1
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / total_gts if total_gts > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def get_all_detections(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    detections = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for pred, target in zip(preds, targets):
                gt_box = target['boxes'][0].cpu().numpy()
                scores = pred['scores'].cpu().numpy() if len(pred['scores'])>0 else np.array([])
                boxes = pred['boxes'].cpu().numpy() if len(pred['boxes'])>0 else np.empty((0,4))
                for score, box in zip(scores, boxes):
                    iou = compute_iou(gt_box, box)
                    is_tp = 1 if iou >= iou_threshold else 0
                    detections.append({'score': float(score), 'is_tp': is_tp})
    return detections


def compute_coco_map(model, dataloader, device, iou_min=0.5, iou_max=0.95, iou_step=0.05, conf_thres=0.0):
    thresholds = np.arange(iou_min, iou_max + 1e-9, iou_step)
    aps = []
    for t in thresholds:
        ap = compute_map(model, dataloader, device, iou_threshold=t, confidence_threshold=conf_thres)
        aps.append(ap)
    mAP = float(np.mean(aps))
    return mAP, dict(zip(thresholds, aps))


def compute_pr_roc(detections):
    y_scores = np.array([d['score'] for d in detections])
    y_true = np.array([d['is_tp'] for d in detections])
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    return {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

