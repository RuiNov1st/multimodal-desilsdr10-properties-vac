"""
Utility functions for data loading and configuration handling.

This module provides:
- Loading image and label arrays
- Loading catalog (tabular) data with flexible feature selection
- Reading YAML configuration files

The catalog loader supports:
(1) Direct feature list from config
(2) External feature list file

Last updated: 2026-04-20
"""

import torch
import numpy as np
import yaml
import pandas as pd
import os
import time

# ------------------------------------------------------------
# Convert relative path to absolute path based on project root
# ------------------------------------------------------------
# project root (important for relative paths)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

# ------------------------------------------------------------
# Load image and label data
# ------------------------------------------------------------
def load_data(img_path, label_path):
    """
    Load image and label arrays from .npy files.

    Parameters
    ----------
    img_path : str
        Path to image numpy file
    label_path : str
        Path to label numpy file

    Returns
    -------
    images : np.ndarray
    labels : np.ndarray
    """
    images = np.load(img_path)
    labels = np.load(label_path)

    return images, labels


# ------------------------------------------------------------
# Load catalog (tabular features)
# ------------------------------------------------------------
def load_catalog(catalog_path, config=None, colsname=None):
    """
    Load catalog data and select feature columns.

    Feature selection can be specified in two ways:
    1) Directly via `colsname` (list or file path)
    2) Automatically from config:
       config['Data']['Property_NAME'] determines which feature set to use

    Parameters
    ----------
    catalog_path : str
        Path to catalog CSV file
    config : dict, optional
        Configuration dictionary (used for automatic feature selection)
    colsname : list or str, optional
        List of column names OR path to a text file containing column names

    Returns
    -------
    arr : np.ndarray
        Selected feature array (N_samples, N_features)
    df : pandas.DataFrame
        Full catalog dataframe
    """
    time1 = time.time()

    # ----------- Determine feature columns -----------
    if colsname is None:
        if config is None:
            raise ValueError("Either colsname or config must be provided")

        property_name = config['Data']['Property_NAME']
        column_config = config['Data']['CatalogColumns'][property_name]

        if column_config['type'] == 'list':
            colsname = column_config['value']

        elif column_config['type'] == 'file':
            # Load feature names from file (one per line)
            with open(resolve_path(column_config['value']), 'r') as f:
                colsname = [line.strip() for line in f if line.strip()]

        else:
            raise ValueError(f"Unknown column type: {column_config['type']}")

    # Backward compatibility: colsname as file path
    elif isinstance(colsname, str):
        with open(resolve_path(colsname), 'r') as f:
            colsname = [line.strip() for line in f if line.strip()]

    # ----------- Load catalog file -----------
    df = pd.read_csv(catalog_path)

    # ----------- Sanity check -----------
    missing_cols = [col for col in colsname if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in catalog: {missing_cols}")

    # ----------- Extract features -----------
    arr = df[colsname].to_numpy()

    print(f"Loaded catalog with {len(colsname)} features in {time.time() - time1:.2f}s")

    return arr, df


# ------------------------------------------------------------
# Read YAML configuration
# ------------------------------------------------------------
def read_config(config_path="./config.yaml"):
    """
    Read YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file

    Returns
    -------
    config : dict
        Configuration dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config