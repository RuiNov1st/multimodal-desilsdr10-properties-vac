"""
Model definitions for multimodal galaxy property estimation.

This module implements a modular architecture that supports:
- Image-only models (CNN-based)
- Catalog-only models (MLP-based)
- Multimodal models (image + tabular fusion)

Design principles:
- Flexible model composition via configuration
- Dynamic module loading (importlib) for extensibility
- Separation of encoders, fusion, and estimator components

Each model consists of:
(1) Encoder(s): extract representations from each modality
(2) Fusion module (optional): combine multimodal features
(3) Estimator: map features to target physical properties

Last updated: 2026-04-20
"""

import torch
import torch.nn as nn
import importlib
import sys
import os
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ------------------------------------------------------------
# Module registry (maps config names → actual implementations)
# ------------------------------------------------------------
Image_Encoder_Paths = {
    "ResNet50": "models.ResNet.ResNet",
    "ResNet34": "models.ResNet.ResNet",
    "ResNet18": "models.ResNet.ResNet",
    "ResNet101": "models.ResNet.ResNet",
}

Catalog_Encoder_Paths = {
    "MLP": "models.MLP.MLP",
}

Fusion_Model_Paths = {
    "concat_model": "models.fusion.concat",
}

Estimate_Model_Paths = {
    "FCN": "models.estimator.FCN",
}


# ------------------------------------------------------------
# Image Encoder (CNN backbone)
# ------------------------------------------------------------
class ImageEncoder(nn.Module):
    """
    Wrapper for image encoders (e.g., ResNet variants).
    Dynamically loads model based on configuration.
    """

    def __init__(self, model_name, input_channels, output_dim):
        super(ImageEncoder, self).__init__()

        if model_name not in Image_Encoder_Paths:
            raise ValueError(f"Unsupported model: {model_name}")

        module = importlib.import_module(Image_Encoder_Paths[model_name])
        model_class = getattr(module, model_name)

        self.image_encoder = model_class(input_channels, output_dim)

    def forward(self, X):
        return self.image_encoder(X)


# ------------------------------------------------------------
# Catalog Encoder (MLP)
# ------------------------------------------------------------
class CatalogEncoder(nn.Module):
    """
    Encoder for tabular photometric features.
    Typically implemented as an MLP.
    """

    def __init__(self, model_name, feature_num, hyparameter_config):
        super(CatalogEncoder, self).__init__()

        if model_name not in Catalog_Encoder_Paths:
            raise ValueError(f"Unsupported model: {model_name}")

        module = importlib.import_module(Catalog_Encoder_Paths[model_name])
        model_class = getattr(module, model_name)

        self.catalog_encoder = model_class(feature_num, hyparameter_config)

    def forward(self, X):
        return self.catalog_encoder(X)


# ------------------------------------------------------------
# Fusion Module
# ------------------------------------------------------------
class FusionModel(nn.Module):
    """
    Combine representations from different modalities.
    Current implementation: simple concatenation.
    """

    def __init__(self, model_name, modal1_dim, modal2_dim):
        super(FusionModel, self).__init__()

        if model_name not in Fusion_Model_Paths:
            raise ValueError(f"Unsupported model: {model_name}")

        module = importlib.import_module(Fusion_Model_Paths[model_name])
        model_class = getattr(module, model_name)

        if model_name == 'concat_model':
            self.fusion_model = model_class()

    def forward(self, modal1_rep, modal2_rep):
        return self.fusion_model(modal1_rep, modal2_rep)


# ------------------------------------------------------------
# Estimator (prediction head)
# ------------------------------------------------------------
class EstimateModel(nn.Module):
    """
    Final prediction head that maps latent features to outputs.
    """

    def __init__(self, model_name, input_dim, output_dim, hidden_dim=1024):
        super(EstimateModel, self).__init__()

        if model_name not in Estimate_Model_Paths:
            raise ValueError(f"Unsupported model: {model_name}")

        module = importlib.import_module(Estimate_Model_Paths[model_name])
        model_class = getattr(module, model_name)

        if model_name == 'FCN':
            self.estimator = model_class(
                input_dim=input_dim,
                output_dim=output_dim
            )

    def forward(self, X):
        output, rep = self.estimator(X)
        return output, rep


# ------------------------------------------------------------
# Image-only model
# ------------------------------------------------------------
class Image_only_Model(nn.Module):
    """
    Model using only image data.
    Pipeline: Image → CNN → Estimator
    """

    def __init__(self, image_model_name, estimate_model_name,
                 channels, features_num, Nbins, hyparameter_config):
        super(Image_only_Model, self).__init__()

        self.channels = channels
        self.Nbins = Nbins

        self.cnn_hparams, self.mlp_hparams = self.get_hparam(hyparameter_config)

        self.image_encoder = ImageEncoder(
            image_model_name,
            channels,
            output_dim=self.cnn_hparams.fc_layer_size
        )

        self.estimate_model = EstimateModel(
            estimate_model_name,
            input_dim=self.cnn_hparams.fc_layer_size,
            output_dim=self.Nbins
        )

    def get_hparam(self, config):
        """
        Convert config to structured hyperparameters.
        """
        cnn_hparams = SimpleNamespace(fc_layer_size=config.cnn_fc_layer_size)

        mlp_hparams = SimpleNamespace(
            fc_layers=config.mlp_fc_layers,
            fc_layer_size=config.mlp_fc_layer_size,
            dropout=config.mlp_dropout,
            activation=config.mlp_activation,
            use_bn=config.mlp_use_bn
        )

        return cnn_hparams, mlp_hparams

    def forward(self, images, values):
        images = images[:, :self.channels, :, :]
        image_value = self.image_encoder(images)
        output, rep = self.estimate_model(image_value)
        return output, rep


# ------------------------------------------------------------
# Catalog-only model
# ------------------------------------------------------------
class Catalog_only_Model(nn.Module):
    """
    Model using only tabular catalog features.
    Pipeline: Features → MLP → Estimator
    """

    def __init__(self, catalog_model_name, estimate_model_name,
                 features_num, Nbins, hyparameter_config):
        super(Catalog_only_Model, self).__init__()

        self.Nbins = Nbins

        self.cnn_hparams, self.mlp_hparams = self.get_hparam(hyparameter_config)

        self.catalog_encoder = CatalogEncoder(
            catalog_model_name,
            features_num,
            self.mlp_hparams
        )

        self.estimate_model = EstimateModel(
            estimate_model_name,
            input_dim=self.mlp_hparams.fc_layer_size,
            output_dim=self.Nbins
        )

    def get_hparam(self, config):
        cnn_hparams = SimpleNamespace(fc_layer_size=config.cnn_fc_layer_size)

        mlp_hparams = SimpleNamespace(
            fc_layers=config.mlp_fc_layers,
            fc_layer_size=config.mlp_fc_layer_size,
            dropout=config.mlp_dropout,
            activation=config.mlp_activation,
            use_bn=config.mlp_use_bn
        )

        return cnn_hparams, mlp_hparams

    def forward(self, images, values):
        catalog_value = self.catalog_encoder(values)
        output, rep = self.estimate_model(catalog_value)
        return output, rep


# ------------------------------------------------------------
# Multimodal model
# ------------------------------------------------------------
class Multimodal_model(nn.Module):
    """
    Multimodal model combining image and catalog features.

    Pipeline:
    Image → CNN → feature
                         → Fusion → Estimator → output
    Catalog → MLP → feature
    """

    def __init__(self, image_model_name, catalog_model_name,
                 fusion_model_name, estimate_model_name,
                 channels, features_num, Nbins, hyparameter_config):
        super(Multimodal_model, self).__init__()

        self.channels = channels
        self.Nbins = Nbins

        self.cnn_hparams, self.mlp_hparams = self.get_hparam(hyparameter_config)

        self.image_encoder = ImageEncoder(
            image_model_name,
            channels,
            output_dim=self.cnn_hparams.fc_layer_size
        )

        self.catalog_encoder = CatalogEncoder(
            catalog_model_name,
            features_num,
            self.mlp_hparams
        )

        self.fusion_model = FusionModel(
            fusion_model_name,
            modal1_dim=self.cnn_hparams.fc_layer_size,
            modal2_dim=self.mlp_hparams.fc_layer_size
        )

        self.estimate_model = EstimateModel(
            estimate_model_name,
            input_dim=self.cnn_hparams.fc_layer_size + self.mlp_hparams.fc_layer_size,
            output_dim=self.Nbins
        )

    def load_weights(self, finetune=False,
                     image_weights=None, catalog_weights=None,
                     freeze=False):
        """
        Load pretrained weights for encoders (optional fine-tuning).
        """
        if finetune:
            image_dict = torch.load(image_weights)
            image_dict = model_load_weights(
                image_dict, 'image_encoder', self.image_encoder.state_dict()
            )
            self.image_encoder.load_state_dict(image_dict)

            catalog_dict = torch.load(catalog_weights)
            catalog_dict = model_load_weights(
                catalog_dict, 'catalog_encoder', self.catalog_encoder.state_dict()
            )
            self.catalog_encoder.load_state_dict(catalog_dict)

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.catalog_encoder.parameters():
                param.requires_grad = False

    def get_hparam(self, config):
        cnn_hparams = SimpleNamespace(fc_layer_size=config.cnn_fc_layer_size)

        mlp_hparams = SimpleNamespace(
            fc_layers=config.mlp_fc_layers,
            fc_layer_size=config.mlp_fc_layer_size,
            dropout=config.mlp_dropout,
            activation=config.mlp_activation,
            use_bn=config.mlp_use_bn
        )

        return cnn_hparams, mlp_hparams

    def forward(self, images, values):
        images = images[:, :self.channels, :, :]
        image_value = self.image_encoder(images)
        catalog_value = self.catalog_encoder(values)

        fused = self.fusion_model(image_value, catalog_value)
        output, rep = self.estimate_model(fused)

        return output, rep


# ------------------------------------------------------------
# Utility: load partial weights with prefix matching
# ------------------------------------------------------------
def model_load_weights(pretrained_dict, prefix, current_dict):
    """
    Load weights by matching prefix (used for encoder extraction).
    """
    new_dict = {
        k[len(prefix)+1:]: v
        for k, v in pretrained_dict.items()
        if k.startswith(prefix)
    }

    assert (new_dict.keys() == current_dict.keys())
    current_dict.update(new_dict)
    return current_dict


# ------------------------------------------------------------
# Debug / standalone test
# ------------------------------------------------------------
if __name__ == '__main__':
    channels = 7
    
    model_configuration =  {
            "cnn_fc_layer_size": 1024,
            "mlp_fc_layers": 4,
            "mlp_fc_layer_size": 256,
            "mlp_dropout": 0.1,
            "mlp_activation": "GELU",
            "mlp_use_bn": False,
            "Nbins": 1,
    }
    model_configuration = SimpleNamespace(** model_configuration)


    # model = Catalog_only_Model("MLP","FCN",features_num=28,Nbins=1,hyparameter_config=model_configuration)
    # model = Image_only_Model("ResNet50","FCN",channels,features_num=28,Nbins=1,hyparameter_config=model_configuration)
    model = Multimodal_model("ResNet50","MLP","concat_model","FCN",channels,features_num=28,Nbins=1,hyparameter_config= model_configuration)

   
    image_data = torch.randn(64,7,64,64)
    catalog_data = torch.randn(64,28)

    output,rep = model(image_data,catalog_data)
    
    print(model)
    print(output.shape)
    print(rep.shape)
    
    