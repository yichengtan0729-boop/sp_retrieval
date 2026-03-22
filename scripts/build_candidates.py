from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sp_retrieval.data.flickr30k import Flickr30KJsonDataset
from sp_retrieval.models.backbones import build_backbone
from sp_retrieval.models.stage1 import Stage1ProjectionHead
from sp_retrieval.utils.config import load_yaml
from sp_retrieval.utils.io import ensure_dir, write_json


@torch.no_grad()
def encode_texts(backbone, head, texts, device, batch_size=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        feat = backbone.encode_text(batch)
        if head is not None:
            feat = head.project_text(feat)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        embs.append(feat.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_images(backbone, head, image_paths, device, batch_size=64):
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        feat = backbone.encode_image(batch)
        if head is not None:
            feat = head.project_image(feat)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        embs.append(feat.cpu())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--use_stage1_head", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") == "auto" else cfg.get("device", "cpu"))

    ds_cfg = cfg["dataset"]
    split_map = {
        "train": ds_cfg["train_split"],
        "val": ds_cfg["val_split"],
        "test": ds_cfg["test_split"],
    }
    split_name = split_map[args.split]

    dataset = Flickr30KJsonDataset(
        annotation_json=ds_cfg["annotation_json"],
        images_root=ds_cfg["images_root"],
        split=split_name,
    )

    backbone = build_backbone(cfg["backbone"]).to(device)
    backbone.eval()

    head = None
    suffix = "backbone"

    if args.use_stage1_head:
        head = Stage1ProjectionHead(
            input_dim=backbone.output_dim,
            proj_dim=cfg["model"]["proj_dim"],
            dropout=cfg["model"].get("dropout", 0.1),
        ).to(device)

        ckpt_path = Path(cfg["output_dir"]) / "best_stage1.pt"
        ckpt = torch.load(ckpt_path, map_location=device)

        if "backbone" in ckpt:
            backbone.load_state_dict(ckpt["backbone"])

        head.load_state_dict(ckpt["head"])
        head.eval()
        suffix = "stage1"

    texts = [s["caption"] for s in dataset.samples]
    image_paths = [str(dataset.images_root / s["image"]) for s in dataset.samples]
    image_ids = [s["image_id"] for s in dataset.samples]

    txt_emb = encode_texts(backbone, head, texts, device)
    img_emb = encode_images(backbone, head, image_paths, device)

    sims = txt_emb @ img_emb.T
    topk = min(args.topk, sims.shape[1])
    topk_idx = torch.topk(sims, k=topk, dim=1).indices

    rows = []
    for i, sample in enumerate(dataset.samples):
        cand_ids = [image_ids[j] for j in topk_idx[i].tolist()]
        rows.append(
            {
                "query_id": sample["id"],
                "caption": sample["caption"],
                "gt_image_id": sample["image_id"],
                "candidate_image_ids": cand_ids,
            }
        )

    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)
    out_path = out_dir / f"candidates_{suffix}_{args.split}_top{args.topk}.json"
    write_json(rows, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()