"""
MLP-based encoder for tabular (catalog) features.

This module implements a configurable multi-layer perceptron (MLP)
used to encode photometric catalog features into latent representations.

Key characteristics:
- Fully configurable architecture via hyperparameters
- Optional Batch Normalization
- Flexible activation functions
- Dropout for regularization

The output representation is used for:
- standalone prediction (catalog-only model)
- multimodal fusion with image features

Last updated: 2026-04-20
"""

import torch
import torch.nn as nn
from types import SimpleNamespace


class MLP(nn.Module):
    """
    Multi-layer perceptron for tabular feature encoding.

    Parameters
    ----------
    features_num : int
        Number of input features
    hyparameter_config : namespace
        Configuration containing:
        - fc_layers       : number of layers
        - fc_layer_size   : hidden dimension
        - dropout         : dropout rate
        - activation      : activation function name (string)
        - use_bn          : whether to use BatchNorm

    Output
    -------
    Latent feature representation of size (batch_size, fc_layer_size)
    """

    def __init__(self, features_num, hyparameter_config):
        super(MLP, self).__init__()

        # ----------- Hyperparameters -----------
        self.activ = getattr(nn, hyparameter_config.activation)()
        self.dropout_rate = hyparameter_config.dropout
        self.num_layers = hyparameter_config.fc_layers
        self.layer_size = hyparameter_config.fc_layer_size
        self.use_bn = getattr(hyparameter_config, "use_bn", True)

        # ----------- Layers -----------
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if self.use_bn else None

        in_dim = features_num
        for _ in range(self.num_layers):
            out_dim = self.layer_size

            # Linear layer
            self.layers.append(nn.Linear(in_dim, out_dim))

            # Optional BatchNorm
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(out_dim))

            in_dim = out_dim

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Forward pass through MLP encoder.
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)

            if self.use_bn:
                x = self.bns[i](x)

            x = self.activ(x)
            x = self.dropout(x)

        return x


# ------------------------------------------------------------
# Debug / standalone test
# ------------------------------------------------------------
if __name__ == '__main__':
    config = {
        "fc_layers": 3,
        "fc_layer_size": 512,
        "dropout": 0.3,
        "activation": "ReLU",
        "use_bn": True,
    }

    config = SimpleNamespace(**config)

    model = MLP(features_num=28, hyparameter_config=config)

    x = torch.rand(5, 28)
    out = model(x)

    print(model)
    print(out.shape)