from data_loader import get_dataloaders
from train import load_or_train_model, load_or_train_yolo_model
import evaluate
from config_loader import load_config
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    annotations_csv = cfg["annotations_coco_csv"]
    images_dir = cfg["training_images_path"]
    model_file = cfg["model_file"]
    plots_dir = cfg["plots_dir"]
    predictions_dir = cfg["predictions_dir"]

    # New YOLO-specific configuration
    yolo_model_file = cfg["yolo_model_file"]
    yolo_predictions_dir = cfg["yolo_predictions_dir"]
    yolo_plots_dir = cfg["yolo_plots_dir"]

    seed = cfg.get("seed", 42)
    img_dim = cfg.get("dimension", 256)
    batch_size = cfg.get("batch_size", 4)
    train_split = cfg.get("train_split", 0.8)
    num_workers = cfg.get("num_workers", 4)
    num_epochs = cfg.get("num_epochs", 40)
    num_classes = cfg.get("num_classes", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        annotations_csv, images_dir, img_dim, batch_size, train_split, num_workers, seed
    )
    logger.info(f"Train loader contains {len(train_loader.dataset)} images in {len(train_loader)} batches")
    logger.info(f"Test loader contains {len(test_loader.dataset)} images in {len(test_loader)} batches")

    # Load or train Faster R-CNN model
    model = load_or_train_model(
        model_file,
        num_classes,
        train_loader,
        device,
        num_epochs,
        plots_dir
    )
    logger.info("✅ Faster R-CNN Training completed.")
    evaluate.evaluate_model(model, test_loader, device, predictions_dir)

    # Load or train YOLO v8 model
    yolo_model = load_or_train_yolo_model(
        yolo_model_file,
        num_classes,
        train_loader,
        device,
        num_epochs,
        yolo_plots_dir
    )
    logger.info("✅ YOLO v8 Training completed.")
    evaluate.evaluate_model(yolo_model, test_loader, device, yolo_predictions_dir)

if __name__ == "__main__":
    main()
