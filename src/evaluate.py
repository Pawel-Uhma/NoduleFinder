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

def evaluate_model(model, dataloader, device):
    model.eval()
    ious = []
    results_dir = "../predictions"
    os.makedirs(results_dir, exist_ok=True)
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
                global_idx = batch_idx * len(images) + i
                print(f"Evaluated Image {global_idx}: IoU = {iou:.4f}")
                ious.append(iou)
                image_tensor = images[i].cpu()
                image_pil = to_pil_image(image_tensor)
                draw = ImageDraw.Draw(image_pil)
                if pred_box is not None:
                    draw.rectangle(pred_box.tolist(), outline="red", width=2)
                draw.rectangle(gt_box.tolist(), outline="green", width=2)
                # Use original filename if available; otherwise default to a generated name.
                file_name = target.get("file_name", f"image_{global_idx}")
                file_base = os.path.splitext(file_name)[0]
                save_path = os.path.join(results_dir, f"{file_base}_iou_{iou:.4f}.jpg")
                image_pil.save(save_path)
                print(f"Saved evaluated image with boxes to {save_path}")
    mean_iou = np.mean(ious) if ious else 0.0
    print(f"Mean IoU on test set: {mean_iou:.4f}")
    return ious, mean_iou
