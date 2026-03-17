import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import set_seed, resolve_device, ensure_dir
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.models.heads import Stage1ProjectionHead
from sp_retrieval.training.trainer_stage1 import collect_embeddings, compute_retrieval_metrics_from_embeddings


def build_topk_records(feats, topk=64):
    metrics, aux = compute_retrieval_metrics_from_embeddings(feats)
    sim_t2i = aux["sim_t2i"]
    unique_img_ids = aux["unique_img_ids"]

    records = []
    for txt_idx, text_id in enumerate(feats["text_ids"]):
        pos_img_id = feats["image_ids"][txt_idx]
        vals, inds = sim_t2i[txt_idx].topk(min(topk, sim_t2i.size(1)))
        records.append({
            "text_id": int(text_id),
            "positive_image_id": int(pos_img_id),
            "candidate_image_ids": [int(unique_img_ids[j]) for j in inds.tolist()],
            "candidate_scores": [float(v) for v in vals.tolist()],
        })
    return records, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    out_dir = ensure_dir(cfg["output_dir"])

    backbone, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    backbone = backbone.to(device)

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    # raw backbone candidates
    raw_feats = collect_embeddings(backbone, loader, device, head=None)
    raw_records, raw_metrics = build_topk_records(raw_feats, topk=args.topk)
    with open(out_dir / f"candidates_backbone_{args.split}_top{args.topk}.json", "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)

    # stage1 projected candidates
    head = Stage1ProjectionHead(
        input_dim=backbone.output_dim,
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    ckpt_path = Path(cfg["output_dir"]) / "best_stage1.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])

    proj_feats = collect_embeddings(backbone, loader, device, head=head)
    proj_records, proj_metrics = build_topk_records(proj_feats, topk=args.topk)
    with open(out_dir / f"candidates_stage1_{args.split}_top{args.topk}.json", "w", encoding="utf-8") as f:
        json.dump(proj_records, f, ensure_ascii=False, indent=2)

    print(f"Saved raw candidates to: {out_dir / f'candidates_backbone_{args.split}_top{args.topk}.json'}")
    print(f"Saved stage1 candidates to: {out_dir / f'candidates_stage1_{args.split}_top{args.topk}.json'}")
    print("[Backbone metrics]")
    for k, v in raw_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("[Stage1 metrics]")
    for k, v in proj_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()