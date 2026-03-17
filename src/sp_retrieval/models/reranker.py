from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_adapter import TokenAdapter
from .shared_private_slots import SharedPrivateSlots


class QueryAwarePooling(nn.Module):
    """
    Query-conditioned image token pooling.
    img_tokens: [BK, N, D]
    txt_global: [BK, D]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, img_tokens: torch.Tensor, txt_global: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(txt_global).unsqueeze(1)     # [BK,1,D]
        k = self.k_proj(img_tokens)                  # [BK,N,D]
        v = self.v_proj(img_tokens)                  # [BK,N,D]

        attn = torch.matmul(q, k.transpose(1, 2)) / (img_tokens.size(-1) ** 0.5)  # [BK,1,N]
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v).squeeze(1)   # [BK,D]
        return F.normalize(pooled, dim=-1)


class SharedPrivateReranker(nn.Module):
    def __init__(
        self,
        in_dim: int,
        token_dim: int,
        num_shared_slots: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_redundant_branch: bool = True,
    ):
        super().__init__()
        self.img_adapter = TokenAdapter(in_dim, token_dim, dropout=dropout)
        self.txt_adapter = TokenAdapter(in_dim, token_dim, dropout=dropout)

        self.decomp = SharedPrivateSlots(
            dim=token_dim,
            num_shared_slots=num_shared_slots,
            num_heads=num_heads,
            dropout=dropout,
            use_redundant_branch=use_redundant_branch,
        )

        self.query_pool = QueryAwarePooling(token_dim)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.shared_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

        self.unique_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,       # [B,K,N,D0]
        txt_tokens: torch.Tensor,       # [B,L,D0]
        txt_mask: torch.Tensor | None,  # [B,L]
        backbone_scores: torch.Tensor,  # [B,K]
    ):
        b, k, n, d0 = img_tokens.shape
        _, l, _ = txt_tokens.shape

        img_tokens = self.img_adapter(img_tokens)
        txt_tokens = self.txt_adapter(txt_tokens)

        out = self.decomp(img_tokens, txt_tokens, txt_mask)

        img_tokens_flat = img_tokens.reshape(b * k, n, -1)
        txt_global_flat = out["txt_shared_global"].reshape(b * k, -1)
        pooled_img = self.query_pool(img_tokens_flat, txt_global_flat).reshape(b, k, -1)

        shared_cat = torch.cat([pooled_img, out["txt_shared_global"]], dim=-1)
        unique_cat = torch.cat([out["img_unique"], out["txt_unique"]], dim=-1)

        shared_score = self.shared_gate(shared_cat).squeeze(-1) + (pooled_img * out["txt_shared_global"]).sum(dim=-1)
        unique_score = self.unique_gate(unique_cat).squeeze(-1) + (out["img_unique"] * out["txt_unique"]).sum(dim=-1)

        if out["img_redundant"] is not None and out["txt_redundant"] is not None:
            redundant_penalty = (out["img_redundant"] * out["txt_redundant"]).sum(dim=-1)
        else:
            redundant_penalty = torch.zeros_like(shared_score)

        final_score = backbone_scores + self.alpha * shared_score + self.beta * unique_score - self.gamma * redundant_penalty

        return {
            "backbone_score": backbone_scores,
            "shared_score": shared_score,
            "unique_score": unique_score,
            "redundant_penalty": redundant_penalty,
            "final_score": final_score,
            **out,
        }