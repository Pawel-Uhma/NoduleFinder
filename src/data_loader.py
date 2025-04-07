#!/usr/bin/env python
import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor


class NoduleDataset(Dataset):
    def __init__(self, annotations_csv, images_dir,
                 img_dim=256, transform=None):
        self.images_dir = images_dir
        self.img_dim = img_dim
        self.transform = transform if transform is not None else ToTensor()
        self.data = []
        with open(annotations_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    "filename": row["filename"],
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        box = torch.tensor([[item["x_min"] * self.img_dim, item["y_min"] * self.img_dim,
                             item["x_max"] * self.img_dim, item["y_max"] * self.img_dim]], dtype=torch.float32)
        target = {"boxes": box, "labels": torch.tensor([1], dtype=torch.int64)}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(annotations_csv, images_dir, img_dim, batch_size, train_split, num_workers):
    dataset = NoduleDataset(annotations_csv, images_dir, img_dim)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Total dataset size: {total_size} images")
    logger.info(f"Training dataset size: {train_size} images")
    logger.info(f"Test dataset size: {test_size} images")

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn)
    return train_loader, test_loader
