"""Direct spatial mask: one logit per pixel (same as exp-cam-spatial-direct.py)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelMask(nn.Module):
    """Learnable B×1×H×W mask logits. Slot i is the mask for image i only."""

    def __init__(self, height: int, width: int, batch_size: int = 1, init: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(
            torch.full((int(batch_size), 1, int(height), int(width)), float(init))
        )

    def forward(self, x=None):
        return self.logits
