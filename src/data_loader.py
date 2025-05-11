import csv
import logging
import os
import random
from typing import Callable, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)

class Compose:
    """Compose a list of transforms that accept **and return** `(image, target)`."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flip the image horizontally with probability *p* and remap the boxes."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: dict) -> Tuple[Image.Image, dict]:
        if random.random() < self.p:
            image = F.hflip(image)
            w = image.width
            boxes = target["boxes"].clone()
            # swap xmin / xmax
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class ToTensorTransform:
    """`torchvision.transforms.ToTensor` wrapper that leaves *target* intact."""

    def __init__(self):
        self._totensor = ToTensor()

    def __call__(self, image: Image.Image, target: dict):
        return self._totensor(image), target


class NoduleDataset(Dataset):
    def __init__(
        self,
        annotations_csv: str,
        images_dir: str,
        img_dim: int = 256,
        transform: Callable = None,
    ):
        self.images_dir = images_dir
        self.img_dim = img_dim  # kept for compatibility but no longer used in __getitem__

        # Ensure we always have a transform that expects (img, target)
        self.transform = (
            transform if transform is not None else Compose([ToTensorTransform()])
        )

        self.data = []
        with open(annotations_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(
                    {
                        "filename": row["filename"],
                        "x_min": float(row["x_min"]),
                        "y_min": float(row["y_min"]),
                        "x_max": float(row["x_max"]),
                        "y_max": float(row["y_max"]),
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item["filename"])
        image = Image.open(img_path).convert("RGB")

        w, h = image.size  # actual image resolution
        box = torch.tensor(
            [
                [
                    item["x_min"] * w,
                    item["y_min"] * h,
                    item["x_max"] * w,
                    item["y_max"] * h,
                ]
            ],
            dtype=torch.float32,
        )
        target = {
            "boxes": box,
            "labels": torch.tensor([1], dtype=torch.int64),
            "file_name": item["filename"]
        }

        image, target = self.transform(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def _split_indices(total_size: int, train_ratio: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=gen).tolist()
    split = int(train_ratio * total_size)
    return indices[:split], indices[split:]


def get_dataloaders(
    annotations_csv: str,
    images_dir: str,
    img_dim: int,
    batch_size: int,
    train_split: float,
    num_workers: int,
    seed: int,
):
    """Return **augmented** train and standard val/test loaders."""

    # Transforms -------------------------------------------------------------
    train_transform = Compose([RandomHorizontalFlip(p=0.5), ToTensorTransform()])
    test_transform = Compose([ToTensorTransform()])

    # Build datasets ---------------------------------------------------------
    full_dataset = NoduleDataset(
        annotations_csv, images_dir, img_dim, transform=None  # placeholder
    )
    total = len(full_dataset)
    train_idx, test_idx = _split_indices(total, train_split, seed)

    train_dataset = torch.utils.data.Subset(
        NoduleDataset(annotations_csv, images_dir, img_dim, transform=train_transform),
        train_idx,
    )
    test_dataset = torch.utils.data.Subset(
        NoduleDataset(annotations_csv, images_dir, img_dim, transform=test_transform),
        test_idx,
    )

    logger.info(f"Total dataset size: {total}")
    logger.info(f"Training   size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader
