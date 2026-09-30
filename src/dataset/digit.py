import os

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2


class DigitDataset(Dataset):
    def __init__(self, data_dir: str, transform: v2.Compose = None):
        self.transform = transform

        filepath = os.path.join(data_dir, "train.pt")
        if os.path.exists(filepath):
            data = torch.load(filepath)
            self.features = data["features"]
            self.labels = data["labels"]
        else:
            try:
                df = pd.read_csv(os.path.join(data_dir, "train.csv"))

                self.features = torch.tensor(
                    df.drop(columns=["label"]).values.astype("float32")
                ).reshape(-1, 1, 28, 28)
                self.labels = torch.tensor(df["label"].values, dtype=torch.int64)
                torch.save({"features": self.features, "labels": self.labels}, filepath)
            except FileNotFoundError:
                print(f"CSV file not found at {os.path.join(data_dir, 'train.csv')}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)
        
        return features, self.labels[idx]

class DigitTestDataset(Dataset):
    def __init__(self, data_dir: str, transform: v2.Compose = None):
        self.transform = transform

        filepath = os.path.join(data_dir, "test.pt")
        if os.path.exists(filepath):
            data = torch.load(filepath)
            self.features = data["features"]
        else:
            try:
                df = pd.read_csv(os.path.join(data_dir, "test.csv"))
                self.features = torch.tensor(df.values.astype("float32")).reshape(-1, 1, 28, 28)
                torch.save({"features": self.features}, filepath)
            except FileNotFoundError:
                print(f"CSV file not found at {os.path.join(data_dir, 'test.csv')}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)

        return features

class DigitDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            *,
            batch_size: int = 32,
            num_workers: int = 4
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        DigitDataset(data_dir=self.data_dir)
        DigitTestDataset(data_dir=self.data_dir) 

    def setup(self, stage: str | None = None):
        if hasattr(self, "train_dataset"):
            return
        
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = DigitDataset(data_dir=self.data_dir, transform=transform)
        self.test_dataset = DigitTestDataset(data_dir=self.data_dir, transform=transform)

        train_dataset_len = int(0.9 * len(self.train_dataset))
        val_dataset_len = len(self.train_dataset) - train_dataset_len
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_dataset_len, val_dataset_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )