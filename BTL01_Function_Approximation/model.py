"""Configurable PyTorch MLP for Gaussian function approximation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """MLP mapping 2D coordinates to a scalar Gaussian-density estimate."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (10,),
        activation: str = "tanh",
        input_scale: tuple[float, float] = (2.0, 4.0),
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain one or more positive widths")
        if activation not in {"relu", "tanh", "sigmoid"}:
            raise ValueError("activation must be one of: relu, tanh, sigmoid")
        if len(input_scale) != 2 or any(scale <= 0 for scale in input_scale):
            raise ValueError("input_scale must contain two positive values")

        activation_layers: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        layers: list[nn.Module] = []
        input_size = 2
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), activation_layers[activation]()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation_name = activation
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError(f"inputs must have shape (batch_size, 2); got {tuple(inputs.shape)}")
        return self.network(inputs / self.input_scale)
