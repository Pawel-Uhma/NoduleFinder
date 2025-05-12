import torch
import os
import logging
from model import ModelFactory
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from plots import plot_loss, plot_map_accuracy, plot_iou_trend
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []
        self.map_history = []
        self.accuracy_history = []
        self.mean_iou_history = []

    def train(
        self,
        dataloader,
        num_epochs,
        scheduler=None,
        eval_dataloader=None,
        predictions_dir="",
        plots_dir="./plots/",
    ):
        logger.info("🚀 Starting training loop.")
        self.model.to(self.device)

        for epoch in range(num_epochs):
            logger.info(f"🔁 Epoch {epoch+1}/{num_epochs}.")
            self.model.train()
            epoch_loss = 0.0

            for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                # move inputs
                images = [img.to(self.device) for img in images]
                # only move tensor values; keep strings (e.g. file_name) intact
                targets = [
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()}
                    for t in targets
                ]

                # forward + backward
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # record epoch loss
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📉 Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            self.loss_history.append(avg_loss)

            # scheduler step
            if scheduler:
                scheduler.step()

            # --- evaluate on validation set at end of this epoch ---
            if eval_dataloader is not None:
                self.model.eval()
                metrics = evaluate_model(
                    self.model,
                    eval_dataloader,
                    self.device,
                    predictions_dir=predictions_dir,
                    plots_dir=plots_dir,
                    save_predictions=False,
                    verbose=False     # <-- this prevents plotting during evaluation
                )
                # append one point PER epoch
                self.mean_iou_history.append(metrics["mean_iou"])
                self.accuracy_history.append(metrics["accuracy"])
                self.map_history.append(metrics["mAP_50_95"])
                # back to train for next epoch
                self.model.train()

        # --- At the end of training, plot the entire history ---
        plot_loss(self.loss_history, title="Training Loss", dir=plots_dir)
        plot_map_accuracy(self.map_history, self.accuracy_history, dir=plots_dir)
        plot_iou_trend(self.mean_iou_history, dir=plots_dir)

        return self.loss_history


def load_or_train_model(model_file: str, num_classes: int, train_dataloader, device, num_epochs, plots_dir) -> torch.nn.Module:
    logger.info(f"📂 Checking for model at: {model_file}")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    if os.path.exists(model_file):
        logger.info("📦 Model file found. Loading saved model...")
        model = ModelFactory.get_model(num_classes)
        model.load_state_dict(torch.load(model_file))
        model.to(device)
        logger.info("✅ Model loaded from disk and moved to device")
    else:
        logger.info("❌ Model not found. Starting training...")
        model = ModelFactory.get_model(num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        trainer = Trainer(model, optimizer, device)
        loss_history = trainer.train(
            train_dataloader,
            num_epochs,
            scheduler,
            eval_dataloader=train_dataloader
        )

        plot_loss(loss_history, title="Training Loss", dir=plots_dir)
        plot_map_accuracy(trainer.map_history, trainer.accuracy_history, dir=plots_dir)
        plot_iou_trend(trainer.mean_iou_history, dir=plots_dir)

        torch.save(model.state_dict(), model_file)
        logger.info(f"💾 Model trained and saved to: {model_file}")

    logger.info("✅ Model ready to use")
    return model
