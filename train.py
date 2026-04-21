"""
Training, validation, and evaluation pipeline.

This script serves as the main entry point for model training and testing.

Main functionalities:
- Build dataloaders from config
- Initialize model and optimizer
- Train with validation and early stopping
- Save best checkpoint
- Evaluate on the test set
- Compute metrics and save predictions

Supported modes:
- training + evaluation
- evaluation only

Last updated: 2026-04-20
"""

import numpy as np
import os
import time
from types import SimpleNamespace

import torch
from torch import nn

from data.dataset import make_dataset
from models.model import (
    Image_only_Model,
    Multimodal_model,
    Catalog_only_Model
)

from utils import (
    set_gpu,
    read_config,
    log_result,
    write_result,
    write_config,
    seed_everything,
    resolve_path
)

from metrics import compute_metrics, make_scatter_plot


# ============================================================
# Evaluation
# ============================================================
def evaluation(test_loader, model_checkpoint, config,
               Nbins, channels, features_num,
               device, model_configuration):
    """
    Evaluate the trained model on the test set.

    Steps:
    1. Load best checkpoint
    2. Run inference
    3. Compute regression metrics
    4. Save predictions and latent representations
    5. Generate scatter plot
    """

    run_name = config['Experiment']['Run_name']

    # -------- Model initialization --------
    # Switch model type here if needed
    # model = Catalog_only_Model(
    #     config['Model']['CatalogEncoder_NAME'],
    #     config['Model']['EstimateModel_NAME'],
    #     features_num,
    #     Nbins,
    #     model_configuration
    # )
    # model = Image_only_Model(
        # config['Model']['ImageEncoder_NAME'],
        # config['Model']['EstimateModel_NAME'],
        # channels,
        # features_num,
        # Nbins,
        # model_configuration
        # ) 
        
    model = Multimodal_model(
        config['Model']['ImageEncoder_NAME'],
        config['Model']['CatalogEncoder_NAME'],
        config['Model']['FusionModel_NAME'],
        config['Model']['EstimateModel_NAME'],
        channels,
        features_num,
        Nbins,
        model_configuration
        )

    model.load_state_dict(torch.load(model_checkpoint))
    model = model.to(device[0])

    if len(device) > 1:
        model = torch.nn.DataParallel(model, device_ids=device)

    loss_fn = nn.MSELoss()

    output_arr = []
    label_arr = []
    indice_arr = []
    rep_arr = []

    model.eval()

    # -------- Inference --------
    with torch.no_grad():
        test_loss = 0.

        for idx, data in enumerate(test_loader):
            image, label, catalog_data, indice = data

            image = image.to(device[0])
            label = label.to(device[0])
            catalog_data = catalog_data.to(device[0])

            output, rep = model(image, catalog_data)

            loss = loss_fn(output.view(-1), label)
            test_loss += loss.item()

            output_arr.append(output)
            label_arr.append(label)
            indice_arr.append(indice)
            rep_arr.append(rep)

        print(f"loss/test: {test_loss / (idx + 1)}")

    # -------- Concatenate outputs --------
    pred_res = np.array(torch.cat(output_arr, dim=0).cpu().view(-1))
    true_res = np.array(torch.cat(label_arr, dim=0).cpu())
    indice_arr = np.array(torch.cat(indice_arr, dim=0).cpu())
    rep_arr = np.array(torch.cat(rep_arr, dim=0).cpu())

    # -------- Metrics --------
    mse, rmse, nrmse, mae, std, bias, outlier, nmad, r = \
        compute_metrics(pred_res, true_res)

    print(
        f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, "
        f"NRMSE: {nrmse:.3f}, MAE: {mae:.3f}, "
        f"STD: {std:.3f}, Bias: {bias:.3f}, "
        f"Outlier: {outlier:.3f}, NMAD: {nmad:.3f}, R:{r:.3f}"
    )

    # -------- Save outputs --------
    make_scatter_plot(
        pred_res,
        true_res,
        std,
        outlier,
        config,
        name=config['Data']['Property_NAME']
    )

    log_result(
        mse, rmse, nrmse, mae,
        std, bias, outlier, nmad, r,
        config,
        name=config['Data']['Property_NAME']
    )

    write_result(true_res, pred_res, indice_arr, rep_arr, config)


# ============================================================
# Early stopping
# ============================================================
class EarlyStopper:
    """
    Early stopping based on validation loss.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ============================================================
# Training
# ============================================================
def train(train_dataloader, valid_dataloader,
          config, Nbins, channels, features_num,
          device, model_configuration):
    """
    Train model with validation and checkpoint saving.
    """

    # -------- Model initialization --------
    # Switch model type here if needed
    # model = Catalog_only_Model(
    #     config['Model']['CatalogEncoder_NAME'],
    #     config['Model']['EstimateModel_NAME'],
    #     features_num,
    #     Nbins,
    #     model_configuration
    # )
    # model = Image_only_Model(
        # config['Model']['ImageEncoder_NAME'],
        # config['Model']['EstimateModel_NAME'],
        # channels,
        # features_num,
        # Nbins,
        # model_configuration
        # ) 
        
    model = Multimodal_model(
        config['Model']['ImageEncoder_NAME'],
        config['Model']['CatalogEncoder_NAME'],
        config['Model']['FusionModel_NAME'],
        config['Model']['EstimateModel_NAME'],
        channels,
        features_num,
        Nbins,
        model_configuration
    )

    model = model.to(device[0])

    if len(device) > 1:
        model = torch.nn.DataParallel(model, device_ids=device)

    # -------- Save path --------
    os.makedirs("./weights", exist_ok=True)
    model_save_path = f"./weights/{config['Experiment']['Run_name']}.pth"

    # -------- Optimization --------
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['Train']['LEARNING_RATE'],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    best_vloss = np.inf
    early_stopper = EarlyStopper(patience=10)

    # ========================================================
    # Training loop
    # ========================================================
    for e in range(config['Train']['EPOCH']):
        print(f"Epoch {e + 1}")

        model.train()
        running_loss = 0.
        start_time = time.time()

        for idx, data in enumerate(train_dataloader):
            image, label, catalog_data, indice = data

            image = image.to(device[0])
            label = label.to(device[0])
            catalog_data = catalog_data.to(device[0])

            optimizer.zero_grad()

            output, _ = model(image, catalog_data)

            loss = loss_fn(output.view(-1), label)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"loss/train: {running_loss / (idx + 1)}")

        # ====================================================
        # Validation
        # ====================================================
        model.eval()

        with torch.no_grad():
            val_loss = 0.

            for idx, data in enumerate(valid_dataloader):
                image, label, catalog_data, indice = data

                image = image.to(device[0])
                label = label.to(device[0])
                catalog_data = catalog_data.to(device[0])

                voutput, _ = model(image, catalog_data)

                vloss = loss_fn(voutput.view(-1), label)

                val_loss += vloss.item()

            val_loss /= (idx + 1)

            print(f"loss/valid: {val_loss}")

            # -------- Save best model --------
            if val_loss < best_vloss:
                best_vloss = val_loss

                if len(device) > 1:
                    torch.save(
                        model.module.state_dict(),
                        model_save_path
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        model_save_path
                    )

                print(f"Best validation loss: {best_vloss}")

            # -------- Early stopping --------
            if early_stopper.early_stop(val_loss):
                print("Early stopping triggered.")
                break


# ============================================================
# Main
# ============================================================
def main(only_test=False):
    """
    Main experiment pipeline.
    """

    # -------- Reproducibility --------
    device = set_gpu(device_list=[0])
    seed_everything(42)

    # -------- Read config --------
    config = read_config('./config/config.yaml')

    # -------- Create output dir --------
    output_dir = f"./output/{config['Experiment']['Run_name']}"
    os.makedirs(output_dir, exist_ok=True)

    # -------- Load task-specific hyperparameters --------
    property_name = config['Data']['Property_NAME']

    model_configuration = SimpleNamespace(
        **config['Model']['HParams'][property_name]
    )

    # -------- Build dataloaders --------
    train_loader, val_loader, test_loader, catalog, \
    Nbins, channels, features_num = make_dataset(config)

    print("Dataset loaded successfully.")

    # -------- Train --------
    if not only_test:
        train(
            train_loader,
            val_loader,
            config,
            Nbins,
            channels,
            features_num,
            device,
            model_configuration
        )

    # -------- Evaluate --------
    model_checkpoint = f"./weights/{config['Experiment']['Run_name']}.pth"

    evaluation(
        test_loader,
        model_checkpoint,
        config,
        Nbins,
        channels,
        features_num,
        device,
        model_configuration
    )

    # -------- Save config snapshot --------
    write_config(config)


if __name__ == '__main__':
    main(only_test=False)