from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, img_global: torch.Tensor, txt_global: torch.Tensor):
        zi = F.normalize(self.img_proj(img_global), dim=-1)
        zt = F.normalize(self.txt_proj(txt_global), dim=-1)
        return zi, zt