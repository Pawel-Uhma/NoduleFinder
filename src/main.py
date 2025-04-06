from data_loader import get_dataloaders
from train import load_or_train_model
import evaluate
from config_loader import load_config
import torch


def main():
    cfg = load_config()
    annotations_csv = cfg["annotations_coco_csv"]
    images_dir = cfg["training_images_path"]
    model_file = cfg["model_file"]
    plots_file = cfg["plots_file"]

    img_dim = cfg.get("dimension", 256)
    batch_size = cfg.get("batch_size", 4)
    train_split = cfg.get("train_split", 0.8)
    num_workers = cfg.get("num_workers", 4)
    num_epochs = cfg.get("num_epochs", 40)
    num_classes = cfg.get("num_classes", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        annotations_csv, images_dir, img_dim, batch_size, train_split, num_workers
    )

    model = load_or_train_model(
        model_file,
        num_classes,
        train_loader,
        device,
        num_epochs,
        plots_file)
    print("✅ Training completed.")

    evaluate.evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
