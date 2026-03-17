from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPrivateSlots(nn.Module):
    """
    Input:
      img_tokens: [B, K, N, D]
      txt_tokens: [B, L, D]
      txt_mask:   [B, L] or None

    Output:
      dict with shared / unique / redundant branches for each candidate.
    """

    def __init__(
        self,
        dim: int,
        num_shared_slots: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_redundant_branch: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_shared_slots = num_shared_slots
        self.use_redundant_branch = use_redundant_branch

        self.shared_slots = nn.Parameter(torch.randn(num_shared_slots, dim) * 0.02)

        self.img_slot_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.txt_slot_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.img_backproj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.txt_backproj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.unique_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        if self.use_redundant_branch:
            self.redundant_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        else:
            self.redundant_head = None

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _shared_from_tokens(
        self,
        tokens: torch.Tensor,
        slot_attn: nn.MultiheadAttention,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens: [M, T, D]
        return: [M, S, D]
        """
        m = tokens.size(0)
        slots = self.shared_slots.unsqueeze(0).expand(m, -1, -1)
        shared, _ = slot_attn(
            query=slots,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return shared

    def forward(
        self,
        img_tokens: torch.Tensor,   # [B, K, N, D]
        txt_tokens: torch.Tensor,   # [B, L, D]
        txt_mask: torch.Tensor | None = None,
    ):
        b, k, n, d = img_tokens.shape
        _, l, _ = txt_tokens.shape

        img_tokens_flat = img_tokens.reshape(b * k, n, d)
        txt_tokens_rep = txt_tokens.unsqueeze(1).expand(b, k, l, d).reshape(b * k, l, d)

        if txt_mask is not None:
            txt_mask_rep = txt_mask.unsqueeze(1).expand(b, k, l).reshape(b * k, l)
            txt_key_padding_mask = ~txt_mask_rep.bool()
        else:
            txt_mask_rep = None
            txt_key_padding_mask = None

        img_shared_slots = self._shared_from_tokens(img_tokens_flat, self.img_slot_attn, None)
        txt_shared_slots = self._shared_from_tokens(txt_tokens_rep, self.txt_slot_attn, txt_key_padding_mask)

        img_shared_global = img_shared_slots.mean(dim=1)
        if txt_mask_rep is not None:
            txt_shared_global = txt_shared_slots.mean(dim=1)
        else:
            txt_shared_global = txt_shared_slots.mean(dim=1)

        img_shared_back = self.img_backproj(img_shared_global).unsqueeze(1)   # [BK,1,D]
        txt_shared_back = self.txt_backproj(txt_shared_global).unsqueeze(1)   # [BK,1,D]

        img_private_tokens = img_tokens_flat - img_shared_back
        txt_private_tokens = txt_tokens_rep - txt_shared_back

        img_unique = self.unique_head(img_private_tokens.mean(dim=1))
        if txt_mask_rep is not None:
            txt_unique = self.unique_head(self._masked_mean(txt_private_tokens, txt_mask_rep))
        else:
            txt_unique = self.unique_head(txt_private_tokens.mean(dim=1))

        if self.use_redundant_branch:
            img_redundant = self.redundant_head(img_private_tokens.mean(dim=1))
            if txt_mask_rep is not None:
                txt_redundant = self.redundant_head(self._masked_mean(txt_private_tokens, txt_mask_rep))
            else:
                txt_redundant = self.redundant_head(txt_private_tokens.mean(dim=1))
        else:
            img_redundant = None
            txt_redundant = None

        return {
            "img_shared_slots": img_shared_slots.reshape(b, k, self.num_shared_slots, d),
            "txt_shared_slots": txt_shared_slots.reshape(b, k, self.num_shared_slots, d),
            "img_shared_global": F.normalize(img_shared_global.reshape(b, k, d), dim=-1),
            "txt_shared_global": F.normalize(txt_shared_global.reshape(b, k, d), dim=-1),
            "img_unique": F.normalize(img_unique.reshape(b, k, d), dim=-1),
            "txt_unique": F.normalize(txt_unique.reshape(b, k, d), dim=-1),
            "img_redundant": None if img_redundant is None else F.normalize(img_redundant.reshape(b, k, d), dim=-1),
            "txt_redundant": None if txt_redundant is None else F.normalize(txt_redundant.reshape(b, k, d), dim=-1),
        }