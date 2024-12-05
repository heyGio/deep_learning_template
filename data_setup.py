import numpy as np
import os
import torch
import pandas as pd
import imageio.v3 as iio 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose



class imageDataset(Dataset):
    def __init__(self, paths_csv, transform=None):
        """
        Arguments:
            data_dir (string): Directory containing uv textures for each subject(subfolders)
            tranform (callable, optional): Transform to be applied on sample
        """
        super().__init__()
        self.images_paths_csv = pd.read_csv(paths_csv)
        self.num_classes = len(self.images_paths_csv['class'].unique())
        self.transform = transform


    def __len__(self):
        return len(self.images_paths_csv) 
    
    def __getitem__(self, idx):
        row = self.images_paths_csv.iloc[idx]
        image = iio.imread(row['path'])
        # image = np.random.rand(256, 256, 3).astype(dtype=np.float32)

        # y = torch.zeros(self.num_classes, dtype=torch.float32)
        # y[row['class']] = 1
        y = torch.tensor(row["class"])

        if self.transform:
            image = self.transform(image)

        return image, y


def create_dataloeaders(
        dataset_csv: str,
        transform: Compose,
        batch_size: int,
        num_workers: int,
        train_val_ratio: float,
):
    
    dataset = imageDataset(dataset_csv, transform=transform)
    train_size = int(train_val_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f'Train: {train_size} -- Val: {val_size}' )

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader 
