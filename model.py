"""
model.py
--------
Arquitectura base: un MLP de pocas capas (nn.Sequential) para
clasificación multiclase.
"""

import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)
