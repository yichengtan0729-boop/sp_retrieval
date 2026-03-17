from __future__ import annotations

import torch
import torch.nn.functional as F


def listwise_ce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores: [B,K]
    labels: [B,K] one-hot or multi-hot
    """
    target = labels.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    logp = F.log_softmax(scores, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def margin_ranking_positive_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    Encourage positive > negatives by margin.
    """
    pos = (scores * labels.float()).sum(dim=-1, keepdim=True)  # [B,1]
    neg_mask = (1 - labels.float()).bool()
    neg = scores.masked_fill(~neg_mask, float("-inf"))
    hardest_neg = neg.max(dim=-1, keepdim=True).values
    loss = F.relu(margin - pos + hardest_neg)
    return loss.mean()


def shared_alignment_loss(img_shared: torch.Tensor, txt_shared: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sim = (img_shared * txt_shared).sum(dim=-1)
    pos = (sim * labels.float()).sum(dim=-1)
    return (1.0 - pos).mean()


def unique_margin_loss(img_unique: torch.Tensor, txt_unique: torch.Tensor, labels: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    sim = (img_unique * txt_unique).sum(dim=-1)
    pos = (sim * labels.float()).sum(dim=-1, keepdim=True)
    neg = sim.masked_fill(labels.bool(), float("-inf")).max(dim=-1, keepdim=True).values
    return F.relu(margin - pos + neg).mean()


def redundant_suppression_loss(img_red: torch.Tensor | None, txt_red: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    if img_red is None or txt_red is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    sim = (img_red * txt_red).sum(dim=-1)
    pos = (sim * labels.float()).sum(dim=-1)
    return pos.mean()


def slot_diversity_loss(shared_slots: torch.Tensor) -> torch.Tensor:
    """
    shared_slots: [B,K,S,D] or [M,S,D]
    """
    if shared_slots.dim() == 4:
        x = shared_slots.reshape(-1, shared_slots.size(-2), shared_slots.size(-1))
    else:
        x = shared_slots

    x = F.normalize(x, dim=-1)
    gram = torch.matmul(x, x.transpose(1, 2))  # [M,S,S]
    eye = torch.eye(gram.size(-1), device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def total_stage2_loss(
    outputs: dict,
    labels: torch.Tensor,
    lambda_shared: float = 1.0,
    lambda_unique: float = 0.3,
    lambda_redundant: float = 0.1,
    lambda_slot_div: float = 0.05,
):
    scores = outputs["final_score"]

    loss_rank = listwise_ce_loss(scores, labels) + margin_ranking_positive_loss(scores, labels)
    loss_shared = shared_alignment_loss(outputs["img_shared_global"], outputs["txt_shared_global"], labels)
    loss_unique = unique_margin_loss(outputs["img_unique"], outputs["txt_unique"], labels)
    loss_red = redundant_suppression_loss(outputs["img_redundant"], outputs["txt_redundant"], labels)
    loss_div = slot_diversity_loss(outputs["img_shared_slots"]) + slot_diversity_loss(outputs["txt_shared_slots"])

    total = (
        loss_rank
        + lambda_shared * loss_shared
        + lambda_unique * loss_unique
        + lambda_redundant * loss_red
        + lambda_slot_div * loss_div
    )

    stats = {
        "loss_total": float(total.item()),
        "loss_rank": float(loss_rank.item()),
        "loss_shared": float(loss_shared.item()),
        "loss_unique": float(loss_unique.item()),
        "loss_redundant": float(loss_red.item()),
        "loss_slot_div": float(loss_div.item()),
    }
    return total, stats