"""Direct spatial mask: one logit per pixel (same as exp-cam-spatial-direct.py)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelMask(nn.Module):
    """Learnable 1×H×W mask logits. No U-Net / CSAE."""

    def __init__(self, height: int, width: int, init: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((1, 1, int(height), int(width)), float(init)))

    def forward(self, x=None):
        return self.logits
