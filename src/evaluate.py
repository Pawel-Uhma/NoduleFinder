import os
import shutil
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from plots import (
    plot_precision_recall_curve,
    plot_f1_curve,
    plot_roc_curve,
)

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

EPS = 1e-8  # small number to keep us away from divide‑by‑zero land

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two *xyxy* boxes given as `[x1, y1, x2, y2]`."""
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter_area == 0.0:
        return 0.0
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (
        box2[2] - box2[0]
    ) * (box2[3] - box2[1]) - inter_area
    return inter_area / max(union_area, EPS)


def compute_map(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> float:
    """Classical 11‑point interpolated AP for a single IoU threshold."""
    model.eval()
    detections: List[Dict[str, float]] = []
    total_gts = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for pred, target in zip(preds, targets):
                total_gts += 1  # single‑object assumption
                gt_box = target["boxes"][0].cpu().numpy()

                if len(pred["boxes" ] ):
                    scores = pred["scores"].cpu().numpy()
                    boxes  = pred["boxes" ].cpu().numpy()
                    best_i = scores.argmax()
                    best_s, best_b = scores[best_i], boxes[best_i]

                    if best_s < confidence_threshold:
                        detections.append({"score": float(best_s), "is_tp": 0})
                    else:
                        iou = compute_iou(gt_box, best_b)
                        detections.append({"score": float(best_s), "is_tp": int(iou >= iou_threshold)})
                else:
                    detections.append({"score": 0.0, "is_tp": 0})

    # Sort by score desc
    detections.sort(key=lambda d: d["score"], reverse=True)

    cum_tp = cum_fp = 0
    precisions, recalls = [], []
    for det in detections:
        if det["is_tp"]:
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / max(cum_tp + cum_fp, EPS))
        recalls.append(cum_tp / max(total_gts, EPS))

    # 11‑point interpolation (but really the exact integral)
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in range(1, len(mrec)))
    return ap


def get_all_detections(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Return list of `{score, is_tp}` for *every* predicted box above 0 conf."""
    model.eval()
    dets: List[Dict[str, float]] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            preds  = model(images)
            for pred, target in zip(preds, targets):
                gt_box = target["boxes"][0].cpu().numpy()
                scores = pred["scores"].cpu().numpy() if len(pred["scores" ]) else np.empty(0)
                boxes  = pred["boxes" ].cpu().numpy() if len(pred["boxes" ]) else np.empty((0, 4))
                for s, b in zip(scores, boxes):
                    dets.append({"score": float(s), "is_tp": int(compute_iou(gt_box, b) >= iou_threshold)})
    return dets


def compute_coco_map(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_min: float = 0.5,
    iou_max: float = 0.95,
    iou_step: float = 0.05,
    conf_thres: float = 0.0):
    thresholds = np.arange(iou_min, iou_max + 1e-9, iou_step)
    aps = [compute_map(model, dataloader, device, t, conf_thres) for t in thresholds]
    return float(np.mean(aps)), dict(zip(thresholds, aps))


def compute_pr_roc(dets: List[Dict[str, float]]) -> Dict[str, np.ndarray | float]:
    """Precision‑recall and ROC data for a list returned by `get_all_detections`."""
    y_scores = np.array([d["score"] for d in dets])
    y_true   = np.array([d["is_tp"] for d in dets])

    precision, recall, pr_thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
    fpr, tpr, roc_thr = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    return {
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thr,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    predictions_dir: str,
    plots_dir: str,
    *,
    save_predictions: bool = True,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, float | np.ndarray | Dict[float, float]]:
    """Run evaluation, optionally saving visualisations and plots.

    Returns a dictionary with the key metrics expected by the trainer.
    """
    model.eval()

    tp = fp = fn = 0
    ious: List[float] = []

    if save_predictions and verbose:
        if os.path.exists(predictions_dir):
            shutil.rmtree(predictions_dir)
        os.makedirs(predictions_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            preds  = model(images)

            for i, (pred, target) in enumerate(zip(preds, targets)):
                gt_box = target["boxes"][0].cpu().numpy()

                if len(pred["boxes"]):
                    scores = pred["scores"].cpu().numpy()
                    boxes  = pred["boxes" ].cpu().numpy()
                    best_i = scores.argmax()
                    best_s, best_b = float(scores[best_i]), boxes[best_i]

                    if best_s < confidence_threshold:
                        # considered "no prediction" above threshold
                        fn += 1
                        pred_box = None
                        iou_val = 0.0
                    else:
                        iou_val = compute_iou(gt_box, best_b)
                        pred_box = best_b
                        if iou_val >= iou_threshold:
                            tp += 1
                        else:
                            fp += 1
                            fn += 1  # miss as well (single‑object assumption)
                else:
                    fn += 1
                    pred_box = None
                    iou_val = 0.0

                ious.append(iou_val)

                # ─── visualise ────────────────────────────────────────────────
                if save_predictions and verbose:
                    img_pil = to_pil_image(images[i].cpu())
                    draw = ImageDraw.Draw(img_pil)
                    draw.rectangle(gt_box.tolist(), outline="green", width=2)
                    if pred_box is not None:
                        draw.rectangle(pred_box.tolist(), outline="red", width=2)

                    orig_name = target.get("file_name", f"image_{batch_idx*len(images)+i}.jpg")
                    base, ext = os.path.splitext(os.path.basename(orig_name))
                    save_name = f"{base}_iou_{iou_val:.4f}{ext or '.jpg'}"
                    img_pil.save(os.path.join(predictions_dir, save_name))

    # ─── single‑threshold metrics ────────────────────────────────────────────
    total = max(tp + fn, 1)  # safeguard
    mean_iou = float(np.mean(ious)) if ious else 0.0
    accuracy = tp / total
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    ap_50 = compute_map(model, dataloader, device, iou_threshold, confidence_threshold)

    # ─── multi‑threshold metrics ────────────────────────────────────────────
    detections = get_all_detections(model, dataloader, device, iou_threshold)
    prroc = compute_pr_roc(detections)
    map_50_95, per_iou_map = compute_coco_map(
        model, dataloader, device, 0.5, 0.95, 0.05, confidence_threshold
    )

    # ─── plots ──────────────────────────────────────────────────────────────
    if verbose and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plot_precision_recall_curve(prroc["precision"], prroc["recall"], plots_dir)
        plot_f1_curve(prroc["f1"], prroc["pr_thresholds"], plots_dir)
        plot_roc_curve(prroc["fpr"], prroc["tpr"], prroc["roc_auc"], plots_dir)

        cm_arr = np.array([[tp, fn], [fp, 0]])  # TN is undefined in single‑class
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_arr, display_labels=["Positive", "Negative"]
        )
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format="d")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close(fig)
        logger.info("✅ Confusion matrix saved.")

        # console summary
        logger.info(f"TP: {tp} | FP: {fp} | FN: {fn}")
        for thr, val in per_iou_map.items():
            logger.info(f"AP@{thr:.2f}: {val:.4f}")
        logger.info(
            "Mean IoU: %.4f | Acc: %.4f | AP@0.50: %.4f | mAP@0.50:0.95: %.4f | "
            "Precision: %.4f | Recall: %.4f | F1: %.4f | ROC‑AUC: %.4f",
            mean_iou,
            accuracy,
            ap_50,
            map_50_95,
            precision,
            recall,
            f1,
            prroc["roc_auc"],
        )

    # ─── package & return ───────────────────────────────────────────────────
    return {
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "ap_50": ap_50,
        "mAP_50_95": map_50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": prroc["roc_auc"],
        "confusion_matrix": np.array([[tp, fn], [fp, 0]]),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "per_iou_map": per_iou_map,
    }
