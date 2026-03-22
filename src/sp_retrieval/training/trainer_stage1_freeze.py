from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm


def symmetric_contrastive_loss(img_feat: torch.Tensor, txt_feat: torch.Tensor, temperature: float = 0.07):
    logits = img_feat @ txt_feat.t() / temperature
    labels = torch.arange(img_feat.size(0), device=img_feat.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def _move_images_if_tensor(images, device):
    if torch.is_tensor(images):
        return images.to(device, non_blocking=True)
    return images


@torch.no_grad()
def collect_embeddings(backbone, loader, device, head=None):
    backbone.eval()
    if head is not None:
        head.eval()

    img_all, txt_all = [], []
    image_ids, text_ids, image_paths = [], [], []

    for batch in tqdm(loader, desc="Collect", leave=False):
        images = _move_images_if_tensor(batch["images"], device)
        texts = batch["texts"]

        out = backbone(images, texts, device=device, return_tokens=False)
        img = out["image_global"]
        txt = out["text_global"]

        if head is not None:
            img, txt = head(img, txt)

        img_all.append(F.normalize(img, dim=-1).cpu())
        txt_all.append(F.normalize(txt, dim=-1).cpu())
        image_ids.extend(batch["image_ids"])
        text_ids.extend(batch["text_ids"])
        image_paths.extend(batch["image_paths"])

    return {
        "img": torch.cat(img_all, dim=0),
        "txt": torch.cat(txt_all, dim=0),
        "image_ids": image_ids,
        "text_ids": text_ids,
        "image_paths": image_paths,
    }


def build_unique_image_bank(img_emb, image_ids, image_paths):
    unique_img_ids = []
    unique_img_embs = []
    unique_img_paths = []
    img_id_to_unique = {}

    for idx, img_id in enumerate(image_ids):
        if img_id not in img_id_to_unique:
            img_id_to_unique[img_id] = len(unique_img_ids)
            unique_img_ids.append(img_id)
            unique_img_embs.append(img_emb[idx])
            unique_img_paths.append(image_paths[idx])

    unique_img_embs = torch.stack(unique_img_embs, dim=0)
    return unique_img_embs, unique_img_ids, unique_img_paths, img_id_to_unique


def build_multicap_mapping(image_ids):
    image_to_txts = defaultdict(list)
    for txt_idx, img_id in enumerate(image_ids):
        image_to_txts[img_id].append(txt_idx)
    return image_to_txts


def compute_retrieval_metrics_from_embeddings(feats):
    img_emb = feats["img"]
    txt_emb = feats["txt"]
    image_ids = feats["image_ids"]
    image_paths = feats["image_paths"]

    unique_img_embs, unique_img_ids, unique_img_paths, img_id_to_unique = build_unique_image_bank(
        img_emb, image_ids, image_paths
    )
    image_to_txts = build_multicap_mapping(image_ids)
    txt_to_img = [img_id_to_unique[x] for x in image_ids]

    sim_t2i = txt_emb @ unique_img_embs.t()
    sim_i2t = sim_t2i.t()

    def recall_t2i(k):
        topk = sim_t2i.topk(k, dim=1).indices
        hits = []
        for txt_idx, img_idx in enumerate(txt_to_img):
            hits.append(img_idx in topk[txt_idx].tolist())
        return float(torch.tensor(hits, dtype=torch.float32).mean().item())

    def recall_i2t(k):
        topk = sim_i2t.topk(k, dim=1).indices
        hits = []
        for i, img_id in enumerate(unique_img_ids):
            positives = set(image_to_txts[img_id])
            pred = topk[i].tolist()
            hits.append(any(p in positives for p in pred))
        return float(torch.tensor(hits, dtype=torch.float32).mean().item())

    metrics = {
        "t2i_R@1": recall_t2i(1),
        "t2i_R@5": recall_t2i(5),
        "t2i_R@10": recall_t2i(10),
        "i2t_R@1": recall_i2t(1),
        "i2t_R@5": recall_i2t(5),
        "i2t_R@10": recall_i2t(10),
    }
    metrics["rsum"] = sum(metrics.values())
    metrics["mean_R@1"] = 0.5 * (metrics["t2i_R@1"] + metrics["i2t_R@1"])

    aux = {
        "sim_t2i": sim_t2i,
        "sim_i2t": sim_i2t,
        "unique_img_ids": unique_img_ids,
        "unique_img_paths": unique_img_paths,
        "txt_to_img": txt_to_img,
    }
    return metrics, aux


def save_retrieval_csv_t2i(out_csv: Path, feats, aux, topk: int = 10):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sim_t2i = aux["sim_t2i"]
    unique_img_ids = aux["unique_img_ids"]
    unique_img_paths = aux["unique_img_paths"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", "positive_image_id", "rank", "pred_image_id", "score", "pred_image_path"])
        for txt_idx, text_id in enumerate(feats["text_ids"]):
            pos_img_id = feats["image_ids"][txt_idx]
            vals, inds = sim_t2i[txt_idx].topk(min(topk, sim_t2i.size(1)))
            for rank, (v, j) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
                writer.writerow([
                    text_id,
                    pos_img_id,
                    rank,
                    unique_img_ids[j],
                    float(v),
                    unique_img_paths[j],
                ])


@torch.no_grad()
def evaluate(backbone, loader, device, head=None):
    feats = collect_embeddings(backbone, loader, device, head=head)
    metrics, aux = compute_retrieval_metrics_from_embeddings(feats)
    return metrics, feats, aux


def train_one_epoch(backbone, head, loader, optimizer, device, temperature, grad_clip_norm=0.0):
    freeze_backbone = bool(getattr(backbone, "freeze", True))
    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()

    head.train()

    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = _move_images_if_tensor(batch["images"], device)
        texts = batch["texts"]

        out = backbone(images, texts, device=device, return_tokens=False)
        zi, zt = head(out["image_global"], out["text_global"])

        loss = symmetric_contrastive_loss(zi, zt, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            params_to_clip = list(head.parameters())
            if not freeze_backbone:
                params_to_clip += [p for p in backbone.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)

        optimizer.step()

        bs = len(texts)
        total_loss += loss.item() * bs
        total_n += bs

    return {"loss": total_loss / max(total_n, 1)}


def fit_stage1(backbone, head, train_loader, val_loader, optimizer, device, cfg, output_dir: Path):
    best_score = -1.0
    best_path = output_dir / "best_stage1.pt"
    temperature = cfg["train"].get("temperature", 0.07)
    grad_clip_norm = cfg["train"].get("grad_clip_norm", 0.0)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            device,
            temperature,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics, _, _ = evaluate(backbone, val_loader, device, head=head)

        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print("[Train]")
        for k, v in train_stats.items():
            print(f" {k}: {v:.4f}")
        print("[Val Projected]")
        for k, v in val_metrics.items():
            print(f" {k}: {v:.4f}")

        if val_metrics["rsum"] > best_score:
            best_score = val_metrics["rsum"]
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to: {best_path}")

    return best_path, best_score
