"""
Dataset construction utilities for multimodal galaxy property estimation.

This module provides:
- Data loading (images, labels, tabular catalog features)
- Train/validation/test splitting
- PyTorch Dataset and DataLoader construction
- Optional image data augmentation
- Reproducible data loading setup

The dataset integrates:
(1) Multi-band galaxy images
(2) Photometric catalog features
(3) Target physical properties (e.g., SFR, stellar mass, metallicity)

Last updated: 2026-04-20
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import random


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import train_test_split
from data.load_utils import read_config, load_data, load_catalog,resolve_path



# ------------------------------------------------------------
# Image augmentation (applied only to training set if enabled)
# ------------------------------------------------------------
def image_Augmentation():
    """
    Define basic geometric augmentations for galaxy images.
    These transformations preserve physical morphology invariance.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90, fill=(0,))  # fill empty pixels with 0
    ])


# ------------------------------------------------------------
# Ensure reproducibility for multi-worker DataLoader
# ------------------------------------------------------------
def worker_init_fn(worker_id):
    """
    Initialize random seeds for each dataloader worker to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Dataset splitting
# ------------------------------------------------------------
def data_split(indices, data, config):
    """
    Split dataset into train / validation / test sets.

    Parameters
    ----------
    indices : array-like
        Indices of the dataset
    data : list
        List of arrays/tensors to be split consistently
    config : dict
        Configuration dictionary

    Returns
    -------
    train_data, valid_data, test_data : list
        Split datasets corresponding to input data list
    """
    if config['Data']['TEST_SIZE'] == 1.:
        # Use entire dataset as test set
        indices_train = []
        indices_valid = []
        indices_test = indices
    else:
        # Train / Test split
        indices_train, indices_test = train_test_split(
            indices,
            shuffle=True,
            test_size=config['Data']['TEST_SIZE'],
            random_state=42
        )

        # Train / Validation split
        indices_train, indices_valid = train_test_split(
            indices_train,
            shuffle=True,
            test_size=config['Data']['VALIDATION_SIZE'],
            random_state=42
        )

    # Apply split to all data arrays
    train_data, valid_data, test_data = [], [], []
    for d in data:
        train_data.append(d[indices_train])
        valid_data.append(d[indices_valid])
        test_data.append(d[indices_test])

    print(f"training set size:{len(indices_train)} \t validation set size:{len(indices_valid)} \t test set size:{len(indices_test)}")

    return train_data, valid_data, test_data


# ------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------
class AstroDataset(Dataset):
    """
    PyTorch Dataset for multimodal galaxy data.

    Each sample includes:
    - image (multi-band tensor)
    - label (target physical property)
    - catalog_data (tabular features)
    - indice (original index for traceability)
    """

    def __init__(self, images, labels, catalog_data, indices,
                 augmentation=None, transform=None):
        self.images = images
        self.labels = labels
        self.catalog_data = catalog_data
        self.indices = indices
        self.augmentation = augmentation
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Apply augmentation (train only)
        if self.augmentation:
            image = self.augmentation(image)

        # Additional transforms if provided
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        catalog_data = self.catalog_data[idx]
        indice = self.indices[idx]

        return (
            image.to(torch.float32),
            label.to(torch.float32),
            torch.Tensor(catalog_data),
            indice
        )


# ------------------------------------------------------------
# Main dataset builder
# ------------------------------------------------------------
def make_dataset(config):
    """
    Build PyTorch DataLoaders for training, validation, and testing.

    Returns
    -------
    train_loader, valid_loader, test_loader : DataLoader
    catalog : pandas.DataFrame
        Original catalog (for analysis/validation)
    Nbins : int
        Output dimension (currently 1 for regression)
    channels : int
        Number of image channels
    features_num : int
        Number of catalog features
    """

    # ----------- Load image and label -----------
    images, labels = load_data(
        resolve_path(config['Data']['IMG_PATH']),
        resolve_path(config['Data']['LABEL_PATH']),
    )

    # Convert images to PyTorch tensor (N, C, H, W)
    images = torch.tensor(images).permute(0, 3, 1, 2)

    # Convert labels to float32
    labels = torch.Tensor(labels).to(torch.float32)

    # ----------- Load catalog features -----------
    # Columns are selected automatically based on Property_NAME in config
    catalog_data, catalog = load_catalog(
        resolve_path(config['Data']['CATALOG_PATH']),
        config=config
    )

    # Generate indices for splitting
    indices = np.arange(len(images))

    # Print shapes for sanity check
    print(images.shape, labels.shape, catalog_data.shape, indices.shape)

    # ----------- Split dataset -----------
    train_data, valid_data, test_data = data_split(
        indices,
        [images, labels, catalog_data, indices],
        config
    )

    Nbins = 1  # regression output

    # ----------- Build datasets & loaders -----------

    augmentation = image_Augmentation() if config['Data']['DATA_AUGMENTATION'] else None

    train_dataset = AstroDataset(
        images=train_data[0],
        labels=train_data[1],
        catalog_data=train_data[2],
        indices=train_data[3],
        augmentation=augmentation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['Train']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    valid_dataset = AstroDataset(
        images=valid_data[0],
        labels=valid_data[1],
        catalog_data=valid_data[2],
        indices=valid_data[3]
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['Train']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    test_dataset = AstroDataset(
        images=test_data[0],
        labels=test_data[1],
        catalog_data=test_data[2],
        indices=test_data[3]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['Train']['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # ----------- Dataset metadata -----------
    channels = int(images.shape[1])
    features_num = int(catalog_data.shape[1])

    return train_loader, valid_loader, test_loader, catalog, Nbins, channels, features_num


# ------------------------------------------------------------
# Debug / standalone run
# ------------------------------------------------------------
if __name__ == '__main__':
    config = read_config(
        resolve_path('config/config.yaml')
    )

    train_loader, valid_loader, test_loader, catalog, Nbins, channels, features_num = make_dataset(config)