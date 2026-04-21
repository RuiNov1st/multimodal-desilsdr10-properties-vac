"""
Utility functions for experiment management.

This module provides:
- Configuration handling
- Reproducibility utilities (random seed)
- Device (GPU/CPU) setup
- Experiment logging and result saving

These utilities support a config-driven training pipeline
and ensure reproducibility and traceability of experiments.

Last updated: 2026-04-20
"""

import numpy as np
import yaml
import torch
import random
import os
from types import SimpleNamespace



# ------------------------------------------------------------
# Convert relative path to absolute path based on project root
# ------------------------------------------------------------
# project root (important for relative paths)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ------------------------------------------------------------
# Config utilities
# ------------------------------------------------------------
def read_config(config_path="./config.yaml"):
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config file

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def write_config(config):
    """
    Save a copy of the config file to the output directory.

    This ensures experiment reproducibility and record keeping.
    """
    output_path = f"./output/{config['Experiment']['Run_name']}/config.yaml"

    with open(output_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file, allow_unicode=True)

    print("write config.yaml successfully!")


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def seed_everything(seed=42):
    """
    Set random seed for reproducibility.

    Covers:
    - Python random
    - NumPy
    - PyTorch (CPU & CUDA)

    Also enforces deterministic behavior in cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------
def set_gpu(device_list=None):
    """
    Configure GPU device(s) for training.

    Parameters
    ----------
    device_list : list or None
        List of GPU indices to use

    Returns
    -------
    list or torch.device
        GPU device IDs or CPU device
    """
    if torch.cuda.is_available():
        if device_list is not None:
            device_ids = device_list
        else:
            device_ids = list(range(torch.cuda.device_count()))

        print(f"Using GPU. CUDA NUMBER: {len(device_ids)}")
        return device_ids
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device


# ------------------------------------------------------------
# Output utilities
# ------------------------------------------------------------
def write_result(true_res, pred_res, indice_arr, rep_arr, config):
    """
    Save prediction results to a compressed file.

    Stored data includes:
    - true values
    - predicted values
    - sample indices
    - learned representations (for analysis)

    Output format: .npz
    """
    file_name = f"./output/{config['Experiment']['Run_name']}/{config['Experiment']['Run_name']}_res.npz"

    np.savez(
        file_name,
        x_true=true_res,
        x_pred=pred_res,
        indices=indice_arr,
        representation=rep_arr
    )

    print(f"save {file_name} success!")


def log_result(mse, rmse, nrmse, mae, std, bias, outlier, nmad, r, config, name):
    """
    Save evaluation metrics to a text file.

    Metrics include:
    - error metrics (MSE, RMSE, MAE)
    - statistical properties (std, bias)
    - robustness indicators (outlier, NMAD)
    - correlation (Pearson R)
    """
    log_path = f"./output/{config['Experiment']['Run_name']}/log_result_{name}.txt"

    with open(log_path, 'w') as log_file:
        log_file.write(
            f"MSE: {mse:.3f}\n"
            f"RMSE: {rmse:.3f}\n"
            f"NRMSE: {nrmse:.3f}\n"
            f"MAE: {mae:.3f}\n"
            f"STD: {std:.3f}\n"
            f"Bias: {bias:.3f}\n"
            f"Outlier: {outlier:.3f}\n"
            f"NMAD: {nmad:.3f}\n"
            f"R: {r:.3f}\n"
        )

    print(f"log {log_path} success!")