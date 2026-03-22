from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sp_retrieval.data.build import build_stage2_loaders
from sp_retrieval.models.stage2 import SharedPrivateReranker
from sp_retrieval.training.trainer_stage2 import evaluate_stage2, fit_stage2
from sp_retrieval.utils.config import load_yaml
from sp_retrieval.utils.io import ensure_dir, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") == "auto" else cfg.get("device", "cpu"))

    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)

    backbone, train_loader, val_loader, test_loader = build_stage2_loaders(cfg)
    backbone = backbone.to(device)

    stage1_ckpt = cfg.get("stage1_ckpt", None)
    if stage1_ckpt is not None:
        ckpt = torch.load(stage1_ckpt, map_location=device)
        if "backbone" in ckpt:
            backbone.load_state_dict(ckpt["backbone"])
            print(f"Loaded stage1 backbone from: {stage1_ckpt}")
        else:
            print(f"Warning: no backbone weights found in {stage1_ckpt}")

    backbone.eval()

    model = SharedPrivateReranker(
        token_dim=cfg["model"]["token_dim"],
        num_shared_slots=cfg["model"]["num_shared_slots"],
        attn_heads=cfg["model"]["attn_heads"],
        dropout=cfg["model"].get("dropout", 0.1),
        use_redundant_branch=cfg["model"].get("use_redundant_branch", True),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 1e-4),
    )

    best_path, best_val = fit_stage2(
        model=model,
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        out_dir=out_dir,
    )

    print(f"best checkpoint: {best_path}")
    print(f"best val score: {best_val:.6f}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_metrics = evaluate_stage2(model, backbone, val_loader, device, cfg)
    test_metrics = evaluate_stage2(model, backbone, test_loader, device, cfg)

    write_json(val_metrics, out_dir / "metrics_stage2_val.json")
    write_json(test_metrics, out_dir / "metrics_stage2_test.json")

    print("val metrics:")
    print(val_metrics)
    print("test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()