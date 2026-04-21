"""
Fully Connected Networks (FCN) for final prediction.

This module implements simple feed-forward networks used as
the estimator (prediction head) in the multimodal framework.

Given latent representations from encoders (image / catalog / fused),
the FCN maps features to target physical properties (e.g., SFR, stellar mass, metallicity).

FCN
- Simple 2-layer fully connected network used as the prediction head.

Returns:
- output : final prediction
- rep    : intermediate latent representation (useful for analysis or visualization)

Last updated: 2026-04-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Basic FCN (2-layer)
# ------------------------------------------------------------
class FCN(nn.Module):
    """
    Simple fully connected network.

    Structure:
    input → Linear → ReLU → Linear → output

    Suitable for:
    - lightweight regression tasks
    - baseline or fast inference
    """

    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super(FCN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # latent representation
        rep = F.relu(self.fc1(X))

        # final prediction
        output = self.fc2(rep)

        return output, rep

