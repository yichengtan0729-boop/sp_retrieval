from __future__ import annotations

from pathlib import Path
import csv

import torch
from tqdm import tqdm

from ..models.losses import total_stage2_loss
from ..utils.misc import save_json


def encode_candidate_batch(backbone, batch, device):
    """
    batch["candidate_images"]: [B,K,C,H,W]
    batch["query_texts"]: list[str]
    """
    imgs = batch["candidate_images"].to(device, non_blocking=True)
    texts = batch["query_texts"]
    b, k, c, h, w = imgs.shape

    imgs_flat = imgs.view(b * k, c, h, w)
    texts_rep = []
    for t in texts:
        texts_rep.extend([t] * k)

    out_img = backbone(imgs_flat, texts_rep, device=device, return_tokens=True)

    # query text once per query
    dummy_img = imgs[:, 0]  # [B,C,H,W]
    out_txt = backbone(dummy_img, texts, device=device, return_tokens=True)

    img_tokens = out_img["image_tokens"]
    if img_tokens is None:
        raise ValueError("Stage2 requires image token outputs, but backbone returned None.")
    txt_tokens = out_txt["text_tokens"]
    if txt_tokens is None:
        raise ValueError("Stage2 requires text token outputs, but backbone returned None.")

    img_tokens = img_tokens.view(b, k, img_tokens.size(1), img_tokens.size(2))
    txt_tokens = out_txt["text_tokens"]
    txt_mask = out_txt.get("text_mask", None)

    backbone_scores = batch["candidate_scores"].to(device)
    labels = batch["labels"].to(device)

    return {
        "img_tokens": img_tokens,
        "txt_tokens": txt_tokens,
        "txt_mask": txt_mask,
        "backbone_scores": backbone_scores,
        "labels": labels,
    }


def train_one_epoch_stage2(backbone, reranker, loader, optimizer, device, cfg):
    backbone.eval()
    reranker.train()

    total = 0.0
    n = 0
    agg = {
        "loss_total": 0.0,
        "loss_rank": 0.0,
        "loss_shared": 0.0,
        "loss_unique": 0.0,
        "loss_redundant": 0.0,
        "loss_slot_div": 0.0,
    }

    for batch in tqdm(loader, desc="Train Stage2", leave=False):
        with torch.no_grad():
            enc = encode_candidate_batch(backbone, batch, device)

        outputs = reranker(
            img_tokens=enc["img_tokens"],
            txt_tokens=enc["txt_tokens"],
            txt_mask=enc["txt_mask"],
            backbone_scores=enc["backbone_scores"],
        )

        loss, stats = total_stage2_loss(
            outputs,
            enc["labels"],
            lambda_shared=cfg["loss"]["lambda_shared"],
            lambda_unique=cfg["loss"]["lambda_unique"],
            lambda_redundant=cfg["loss"]["lambda_redundant"],
            lambda_slot_div=cfg["loss"]["lambda_slot_div"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = enc["labels"].size(0)
        n += bs
        total += loss.item() * bs
        for k, v in stats.items():
            agg[k] += v * bs

    out = {"loss": total / max(n, 1)}
    for k, v in agg.items():
        out[k] = v / max(n, 1)
    return out


@torch.no_grad()
def evaluate_stage2(backbone, reranker, loader, device):
    backbone.eval()
    reranker.eval()

    rows = []
    hits_backbone_1 = []
    hits_backbone_5 = []
    hits_backbone_10 = []
    hits_rerank_1 = []
    hits_rerank_5 = []
    hits_rerank_10 = []

    for batch in tqdm(loader, desc="Eval Stage2", leave=False):
        enc = encode_candidate_batch(backbone, batch, device)

        outputs = reranker(
            img_tokens=enc["img_tokens"],
            txt_tokens=enc["txt_tokens"],
            txt_mask=enc["txt_mask"],
            backbone_scores=enc["backbone_scores"],
        )

        b_scores = outputs["backbone_score"].cpu()
        r_scores = outputs["final_score"].cpu()
        labels = enc["labels"].cpu()

        for i in range(labels.size(0)):
            pos_idx = labels[i].argmax().item()

            rank_b = torch.argsort(b_scores[i], descending=True)
            rank_r = torch.argsort(r_scores[i], descending=True)

            hits_backbone_1.append(int(pos_idx in rank_b[:1].tolist()))
            hits_backbone_5.append(int(pos_idx in rank_b[:5].tolist()))
            hits_backbone_10.append(int(pos_idx in rank_b[:10].tolist()))

            hits_rerank_1.append(int(pos_idx in rank_r[:1].tolist()))
            hits_rerank_5.append(int(pos_idx in rank_r[:5].tolist()))
            hits_rerank_10.append(int(pos_idx in rank_r[:10].tolist()))

            text_id = batch["text_ids"][i]
            pos_image_id = batch["positive_image_ids"][i]
            cand_ids = batch["candidate_image_ids"][i]

            top10_b = rank_b[:10].tolist()
            top10_r = rank_r[:10].tolist()

            for rank, j in enumerate(top10_b, start=1):
                rows.append({
                    "mode": "backbone",
                    "text_id": text_id,
                    "positive_image_id": pos_image_id,
                    "rank": rank,
                    "pred_image_id": cand_ids[j],
                    "score": float(b_scores[i, j].item()),
                })

            for rank, j in enumerate(top10_r, start=1):
                rows.append({
                    "mode": "rerank",
                    "text_id": text_id,
                    "positive_image_id": pos_image_id,
                    "rank": rank,
                    "pred_image_id": cand_ids[j],
                    "score": float(r_scores[i, j].item()),
                })

    metrics = {
        "backbone_t2i_R@1": sum(hits_backbone_1) / max(len(hits_backbone_1), 1),
        "backbone_t2i_R@5": sum(hits_backbone_5) / max(len(hits_backbone_5), 1),
        "backbone_t2i_R@10": sum(hits_backbone_10) / max(len(hits_backbone_10), 1),
        "rerank_t2i_R@1": sum(hits_rerank_1) / max(len(hits_rerank_1), 1),
        "rerank_t2i_R@5": sum(hits_rerank_5) / max(len(hits_rerank_5), 1),
        "rerank_t2i_R@10": sum(hits_rerank_10) / max(len(hits_rerank_10), 1),
    }
    metrics["delta_t2i_R@1"] = metrics["rerank_t2i_R@1"] - metrics["backbone_t2i_R@1"]
    metrics["delta_t2i_R@5"] = metrics["rerank_t2i_R@5"] - metrics["backbone_t2i_R@5"]
    metrics["delta_t2i_R@10"] = metrics["rerank_t2i_R@10"] - metrics["backbone_t2i_R@10"]

    return metrics, rows


def save_eval_rows_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = ["mode", "text_id", "positive_image_id", "rank", "pred_image_id", "score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fit_stage2(backbone, reranker, train_loader, val_loader, optimizer, device, cfg, output_dir: Path):
    best_score = -1.0
    best_path = output_dir / "best_stage2.pt"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch_stage2(backbone, reranker, train_loader, optimizer, device, cfg)
        val_metrics, _ = evaluate_stage2(backbone, reranker, val_loader, device)

        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print("[Train Stage2]")
        for k, v in train_stats.items():
            print(f"  {k}: {v:.4f}")
        print("[Val Stage2]")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        score = val_metrics["rerank_t2i_R@1"] + val_metrics["rerank_t2i_R@5"] + val_metrics["rerank_t2i_R@10"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "reranker": reranker.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to: {best_path}")

    return best_path, best_score