import os
import logging
from typing import List, Tuple, Optional, Dict

import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import ModelFactory
from plots import (
    plot_loss,
    plot_map_accuracy,
    plot_iou_trend,
    plot_map_vs_iou,
)
from evaluate import evaluate_model, compute_coco_map

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    """Wrapper that handles the full training / validation / evaluation loop."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # History containers – guaranteed to have the same length (num_epochs)
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.map_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.mean_iou_history: List[float] = []

    def _compute_validation_loss(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Compute average validation loss over the provided dataloader."""
        # Ensure model is in train mode to compute losses (detection models only return losses in training mode)
        was_training = self.model.training
        self.model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                    for t in targets
                ]
                loss_dict = self.model(images, targets)
                # Expecting a dict of loss components
                batch_loss = sum(loss for loss in loss_dict.values())
                val_loss += batch_loss.item()
        avg_val_loss = val_loss / len(dataloader)
        # Restore original training/eval mode if needed
        if not was_training:
            self.model.eval()
        return avg_val_loss

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        predictions_dir: str = "",  # forwarded to evaluate_model
        plots_dir: str = "./plots/",
        evaluate_each_epoch: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """Main training loop.

        Args:
            dataloader: Training dataloader.
            num_epochs: Number of epochs.
            scheduler: Optional LR scheduler stepped once per epoch **after** all
                training / validation work is done.
            eval_dataloader: DataLoader used for metrics (IoU / mAP / etc.). If
                ``None``, no metrics or validation loss will be computed.
            predictions_dir: Passed straight through to ``evaluate_model`` –
                can stay empty if you do not save predictions.
            plots_dir: Destination folder for PNG plots.
            evaluate_each_epoch: If ``True`` compute detection metrics (IoU /
                mAP / TP-FP-FN) every epoch; otherwise compute them **once** at
                the very end. Validation *loss* is always computed each epoch
                because it is cheap and required for the two-line loss plot.
        Returns:
            Tuple of (training_loss_history, validation_loss_history).
        """

        logger.info("🚀 Starting training loop.")
        self.model.to(self.device)
        os.makedirs(plots_dir, exist_ok=True)

        last_per_iou_map = None  # type: ignore
        last_metrics = None      # type: ignore

        for epoch in range(num_epochs):
            logger.info(f"🔁 Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0.0

            # ─────── Training pass ───────────────────────────────────────
            for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                images = [img.to(self.device) for img in images]
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]
                loss_dict = self.model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_train_loss)
            logger.info(f"📉 Training loss: {avg_train_loss:.4f}")

            # ─────── Validation loss (always) ────────────────────────────
            if eval_dataloader is not None:
                avg_val_loss = self._compute_validation_loss(eval_dataloader)
                self.val_loss_history.append(avg_val_loss)
                logger.info(f"📉 Validation loss: {avg_val_loss:.4f}")
            else:
                self.val_loss_history.append(float('nan'))

            # ─────── Detection metrics (optional per-epoch) ──────────────
            if eval_dataloader is not None and evaluate_each_epoch:
                self.model.eval()
                last_metrics = evaluate_model(
                    self.model,
                    eval_dataloader,
                    self.device,
                    predictions_dir,
                    plots_dir,
                    save_predictions=False,
                    verbose=False,
                )
                self.mean_iou_history.append(last_metrics["mean_iou"])
                self.accuracy_history.append(last_metrics["accuracy"])
                self.map_history.append(last_metrics["mAP_50_95"])
                last_per_iou_map = last_metrics["per_iou_map"]
                self.model.train()
            else:
                # Maintain equal-length histories for plotting later.
                if eval_dataloader is not None:
                    self.mean_iou_history.append(float("nan"))
                    self.accuracy_history.append(float("nan"))
                    self.map_history.append(float("nan"))

            # ─────── Scheduler step (end of epoch) ───────────────────────
            if scheduler is not None:
                scheduler.step()

        # ─────────────────────────────────────────────────────────────────┐
        # Post-training final evaluation (if not done every epoch)        │
        # ─────────────────────────────────────────────────────────────────┘
        if eval_dataloader is not None and not evaluate_each_epoch:
            last_metrics = evaluate_model(
                self.model,
                eval_dataloader,
                self.device,
                predictions_dir,
                plots_dir,
                save_predictions=False,
                verbose=True,  # Console prints TP / FP / FN during eval
            )
            last_per_iou_map = last_metrics["per_iou_map"]
            # Append *real* metrics for the final epoch position
            self.mean_iou_history[-1] = last_metrics["mean_iou"]
            self.accuracy_history[-1] = last_metrics["accuracy"]
            self.map_history[-1] = last_metrics["mAP_50_95"]

        # ─────────────────────────────────────────────────────────────────┐
        # Console summary                                                 │
        # ─────────────────────────────────────────────────────────────────┘
        if last_metrics is not None:
            logger.info(
                f"TP: {last_metrics['TP']}  "
                f"FP: {last_metrics['FP']}  "
                f"FN: {last_metrics['FN']}"
            )
            for thr, ap in last_metrics["per_iou_map"].items():
                logger.info(f"AP@{thr:.2f}: {ap:.4f}")

        # ─────────────────────────────────────────────────────────────────┐
        # Plots                                                          │
        # ─────────────────────────────────────────────────────────────────┘
        plot_loss(self.loss_history, self.val_loss_history, dir=plots_dir)
        plot_map_accuracy(self.map_history, self.accuracy_history, dir=plots_dir)
        plot_iou_trend(self.mean_iou_history, dir=plots_dir)

        if eval_dataloader is not None and last_per_iou_map is None:
            _, last_per_iou_map = compute_coco_map(
                self.model,
                eval_dataloader,
                self.device,
                iou_min=0.5,
                iou_max=0.95,
                iou_step=0.05,
                conf_thres=0.0,
            )
        if last_per_iou_map is not None:
            plot_map_vs_iou(last_per_iou_map, dir=plots_dir)

        return self.loss_history, self.val_loss_history


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper – train or load from file
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_model(
    model_file: str,
    num_classes: int,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    num_epochs: int,
    plots_dir: str,
    evaluate_each_epoch: bool = True,
) -> torch.nn.Module:
    """Load a saved model if it exists; otherwise train from scratch."""

    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    if os.path.exists(model_file):
        logger.info("📦 Found saved model – loading from disk …")
        model = ModelFactory.get_model(num_classes)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        logger.info("✅ Model loaded and moved to device")
        return model

    logger.info("❌ No saved model found – starting fresh training …")
    model = ModelFactory.get_model(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, optimizer, device)
    trainer.train(
        dataloader=train_dataloader,
        num_epochs=num_epochs,
        scheduler=scheduler,
        eval_dataloader=eval_dataloader,
        predictions_dir="",
        plots_dir=plots_dir,
        evaluate_each_epoch=evaluate_each_epoch,
    )

    torch.save(model.state_dict(), model_file)
    logger.info(f"💾 Model trained and saved to: {model_file}")

    return model
