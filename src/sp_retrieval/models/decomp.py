from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLP
from .losses import info_nce_loss, orthogonality_loss, variance_loss


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    shared_dim: int
    private_dim: int
    temperature: float
    lambda_recon: float
    lambda_private: float
    lambda_ortho: float
    lambda_var: float


class SharedPrivateRetrievalModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.img_shared = MLP(cfg.input_dim, cfg.shared_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)
        self.txt_shared = MLP(cfg.input_dim, cfg.shared_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)
        self.img_private = MLP(cfg.input_dim, cfg.private_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)
        self.txt_private = MLP(cfg.input_dim, cfg.private_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)
        self.img_recon = MLP(cfg.shared_dim + cfg.private_dim, cfg.input_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)
        self.txt_recon = MLP(cfg.shared_dim + cfg.private_dim, cfg.input_dim, cfg.hidden_dim, num_layers=2, dropout=0.1)

    def encode_and_decompose(self, ux: torch.Tensor, uy: torch.Tensor):
        sx = self.img_shared(ux)
        sy = self.txt_shared(uy)
        px = self.img_private(ux)
        py = self.txt_private(uy)
        ux_rec = self.img_recon(torch.cat([sx, px], dim=-1))
        uy_rec = self.txt_recon(torch.cat([sy, py], dim=-1))
        return {
            'ux': ux,
            'uy': uy,
            'sx': sx,
            'sy': sy,
            'px': px,
            'py': py,
            'ux_rec': ux_rec,
            'uy_rec': uy_rec,
        }

    def compute_losses(self, out):
        loss_align = info_nce_loss(out['sx'], out['sy'], self.cfg.temperature)
        loss_private = 0.5 * (
            info_nce_loss(out['px'], out['py'], self.cfg.temperature) +
            info_nce_loss(out['px'], -out['py'], self.cfg.temperature)
        )
        # Keep private branch from collapsing but do not encourage cross-modal match.
        # Second term prevents trivial alignment by making one view sign-flipped.
        loss_recon = F.mse_loss(out['ux_rec'], out['ux']) + F.mse_loss(out['uy_rec'], out['uy'])
        loss_ortho = orthogonality_loss(out['sx'], out['px']) + orthogonality_loss(out['sy'], out['py'])
        loss_var = variance_loss(out['sx']) + variance_loss(out['sy']) + 0.5 * (variance_loss(out['px']) + variance_loss(out['py']))
        total = (
            loss_align
            + self.cfg.lambda_recon * loss_recon
            + self.cfg.lambda_private * loss_private
            + self.cfg.lambda_ortho * loss_ortho
            + self.cfg.lambda_var * loss_var
        )
        return {
            'loss': total,
            'loss_align': loss_align,
            'loss_recon': loss_recon,
            'loss_private': loss_private,
            'loss_ortho': loss_ortho,
            'loss_var': loss_var,
        }
