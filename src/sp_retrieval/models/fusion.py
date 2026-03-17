from __future__ import annotations
import torch
import torch.nn as nn


class ConservativeFusion(nn.Module):
    def __init__(self, init_logits=(2.2, 0.8), eps: float = 1e-6):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
        self.eps = eps

    def _norm(self, s: torch.Tensor) -> torch.Tensor:
        return (s - s.mean()) / (s.std() + self.eps)

    def get_weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self, s_clip: torch.Tensor, s_shared: torch.Tensor) -> torch.Tensor:
        w = self.get_weights()
        s_clip_n = self._norm(s_clip)
        s_shared_n = self._norm(s_shared)
        return w[0] * s_clip_n + w[1] * s_shared_n
