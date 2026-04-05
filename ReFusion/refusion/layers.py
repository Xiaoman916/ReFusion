"""Linear per-modality classification head."""

import torch
import torch.nn as nn


class ModalityClassifier(nn.Module):
    """Maps latent vectors to class logits."""

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
