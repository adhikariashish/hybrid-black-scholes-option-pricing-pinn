from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional

import torch
import torch.nn as nn

class PricingPINN(nn.Module):
    """
        MLP Pricing PINN: (S, t) -> V(S, t)

    """

    def __init__(
            self,
            in_dimension: int = 2,
            hidden_sizes: Sequence[int] = (128, 128, 128),
            activation: str = 'tanh',
            out_dimension: int = 1,
    ) -> None:
        super().__init__()

        act = activation.lower()

        if act == 'tanh':
            act_layer = nn.Tanh
        elif act == 'relu':
            act_layer = nn.ReLU
        elif act == 'gelu':
            act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        layers: list[nn.Module] = []
        prev = in_dimension

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_layer())
            prev = h
        layers.append(nn.Linear(prev, out_dimension))

        self.net = nn.Sequential(*layers)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2) with columns (S, t)
        return self.net(x)