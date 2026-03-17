from __future__ import annotations
import torch
import torch.nn.functional as F


def info_nce_loss(zx: torch.Tensor, zy: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    zx = F.normalize(zx, dim=-1)
    zy = F.normalize(zy, dim=-1)
    logits = zx @ zy.t() / temperature
    labels = torch.arange(zx.shape[0], device=zx.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def orthogonality_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Dimension-agnostic orthogonality / decorrelation loss.

    Supports a.shape = [B, Da], b.shape = [B, Db] with Da != Db.
    We whiten each branch feature-wise, then penalize cross-correlation.
    """
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    a = a / (a.std(dim=0, keepdim=True) + 1e-6)
    b = b / (b.std(dim=0, keepdim=True) + 1e-6)

    c = (a.T @ b) / max(a.shape[0] - 1, 1)   # [Da, Db]
    return c.pow(2).mean()


def variance_loss(x: torch.Tensor, min_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0) + eps)
    return F.relu(min_std - std).mean()
