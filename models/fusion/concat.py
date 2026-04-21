"""
Fusion module: concatenation-based multimodal fusion.

This module implements a simple feature-level fusion strategy by
concatenating representations from different modalities (e.g., image and catalog).

Fusion strategy:
    fused = [image_features ; catalog_features]

This serves as a baseline fusion method and can be replaced by more
advanced approaches (e.g., attention-based or gated fusion).

Last updated: 2026-04-20
"""

import torch
import torch.nn as nn


class concat_model(nn.Module):
    """
    Concatenation-based fusion.

    Inputs:
    - image_value   : feature representation from image encoder
    - catalog_value : feature representation from catalog encoder

    Output:
    - fused feature vector (concatenated along feature dimension)
    """

    def __init__(self):
        super(concat_model, self).__init__()

    def forward(self, image_value, catalog_value):
        # Concatenate along feature dimension (dim=1)
        output = torch.cat([image_value, catalog_value], dim=1)
        return output


# ------------------------------------------------------------
# Debug / standalone test
# ------------------------------------------------------------
if __name__ == '__main__':
    model = concat_model()

    image_rep = torch.randn((64, 1024))
    catalog_rep = torch.randn((64, 1024))

    output = model(image_rep, catalog_rep)

    print(output.shape)