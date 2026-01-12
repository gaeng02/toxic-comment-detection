from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EmbeddingNNConfig :
    input_dim : int
    hidden_dim1 : int = 64
    hidden_dim2 : int = 32
    dropout : float = 0.0


class EmbeddingNN(nn.Module) :

    def __init__(self, config: EmbeddingNNConfig) :
        super().__init__()
        layers = []

        layers.append(nn.Linear(config.input_dim, config.hidden_dim1))
        layers.append(nn.ReLU())
        if (config.dropout > 0.0) :
            layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(config.hidden_dim1, config.hidden_dim2))
        layers.append(nn.ReLU())
        if (config.dropout > 0.0) :
            layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(config.hidden_dim2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        out = self.net(x)
        return out.squeeze(-1)

