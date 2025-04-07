#!/usr/bin/env python
import os, torch, numpy as np, logging
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

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

def compute_map(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    all_detections = []
    total_gts = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                total_gts += len(gt_boxes)
                pred_boxes = pred["boxes"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                if len(pred_boxes) == 0:
                    continue
                order = scores.argsort()[::-1]
                pred_boxes = pred_boxes[order]
                scores = scores[order]
                detected = [False] * len(gt_boxes)
                for score, box in zip(scores, pred_boxes):
                    match_found = False
                    for idx, gt_box in enumerate(gt_boxes):
                        iou = compute_iou(gt_box, box)
                        if iou >= iou_threshold and not detected[idx]:
                            match_found = True
                            detected[idx] = True
                            break
                    is_tp = 1 if match_found else 0
                    all_detections.append({'score': score, 'is_tp': is_tp})
    all_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)
    cum_tp = 0
    cum_fp = 0
    precisions = []
    recalls = []
    for det in all_detections:
        if det['is_tp']:
            cum_tp += 1
        else:
            cum_fp += 1
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / total_gts if total_gts > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    # Interpolate precision-recall curve
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def evaluate_model(model, dataloader, device, predictions_dir, save_predictions=True, iou_threshold=0.5):
    model.eval()
    ious = []
    correct = 0
    total = 0
    if save_predictions:
        os.makedirs(predictions_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            predictions = model(images)
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                gt_box = target["boxes"][0].cpu().numpy()
                if len(pred["boxes"]) > 0:
                    scores = pred["scores"].cpu().numpy()
                    boxes = pred["boxes"].cpu().numpy()
                    best_idx = scores.argmax()
                    pred_box = boxes[best_idx]
                    iou = compute_iou(gt_box, pred_box)
                else:
                    iou = 0.0
                    pred_box = None
                total += 1
                if iou >= iou_threshold:
                    correct += 1
                logger.info(f"Evaluated Image {batch_idx * len(images) + i}: IoU = {iou:.4f}")
                ious.append(iou)
                if save_predictions:
                    image_tensor = images[i].cpu()
                    image_pil = to_pil_image(image_tensor)
                    draw = ImageDraw.Draw(image_pil)
                    if pred_box is not None:
                        draw.rectangle(pred_box.tolist(), outline="red", width=2)
                    draw.rectangle(gt_box.tolist(), outline="green", width=2)
                    file_name = target.get("file_name", f"image_{batch_idx * len(images) + i}")
                    file_base = os.path.splitext(file_name)[0]
                    save_path = os.path.join(predictions_dir, f"{file_base}_iou_{iou:.4f}.jpg")
                    image_pil.save(save_path)
                    logger.info(f"Saved evaluated image with boxes to {save_path}")
    mean_iou = np.mean(ious) if ious else 0.0
    accuracy = correct / total if total > 0 else 0.0
    ap = compute_map(model, dataloader, device, iou_threshold)
    logger.info(f"Mean IoU on test set: {mean_iou:.4f}")
    logger.info(f"Accuracy on test set (IoU threshold {iou_threshold}): {accuracy:.4f}")
    logger.info(f"mAP on test set (IoU threshold {iou_threshold}): {ap:.4f}")
    return ious, mean_iou, accuracy, ap

