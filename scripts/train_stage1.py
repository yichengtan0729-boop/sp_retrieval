import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import (
    set_seed,
    resolve_device,
    ensure_dir,
    save_json,
    count_parameters,
)
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.models.heads import Stage1ProjectionHead
from sp_retrieval.training.trainer_stage1 import (
    evaluate,
    fit_stage1,
    save_retrieval_csv_t2i,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    output_dir = ensure_dir(cfg["output_dir"])

    backbone, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    backbone = backbone.to(device)

    head = Stage1ProjectionHead(
        input_dim=backbone.output_dim,
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    print(f"Device: {device}")
    print(f"Backbone output dim: {backbone.output_dim}")
    print(f"Head trainable params: {count_parameters(head):,}")

    # raw backbone metrics first
    raw_val_metrics, raw_val_feats, raw_val_aux = evaluate(backbone, val_loader, device, head=None)
    raw_test_metrics, raw_test_feats, raw_test_aux = evaluate(backbone, test_loader, device, head=None)

    save_json(output_dir / "metrics_backbone_val.json", raw_val_metrics)
    save_json(output_dir / "metrics_backbone_test.json", raw_test_metrics)
    save_retrieval_csv_t2i(output_dir / "retrieval_backbone_val.csv", raw_val_feats, raw_val_aux, topk=10)
    save_retrieval_csv_t2i(output_dir / "retrieval_backbone_test.csv", raw_test_feats, raw_test_aux, topk=10)

    torch.save(raw_test_feats["img"], output_dir / "image_embeds_backbone_test.pt")
    torch.save(raw_test_feats["txt"], output_dir / "text_embeds_backbone_test.pt")

    print("\n[Backbone / Val]")
    for k, v in raw_val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Backbone / Test]")
    for k, v in raw_test_metrics.items():
        print(f"  {k}: {v:.4f}")

    best_path, best_score = fit_stage1(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        output_dir=output_dir,
    )

    print(f"\nBest checkpoint: {best_path}")
    print(f"Best val rsum: {best_score:.4f}")

    ckpt = torch.load(best_path, map_location=device)
    head.load_state_dict(ckpt["head"])

    proj_val_metrics, proj_val_feats, proj_val_aux = evaluate(backbone, val_loader, device, head=head)
    proj_test_metrics, proj_test_feats, proj_test_aux = evaluate(backbone, test_loader, device, head=head)

    save_json(output_dir / "metrics_stage1_val.json", proj_val_metrics)
    save_json(output_dir / "metrics_stage1_test.json", proj_test_metrics)
    save_retrieval_csv_t2i(output_dir / "retrieval_stage1_val.csv", proj_val_feats, proj_val_aux, topk=10)
    save_retrieval_csv_t2i(output_dir / "retrieval_stage1_test.csv", proj_test_feats, proj_test_aux, topk=10)

    torch.save(proj_test_feats["img"], output_dir / "image_embeds_stage1_test.pt")
    torch.save(proj_test_feats["txt"], output_dir / "text_embeds_stage1_test.pt")

    comparison = {
        "backbone_test": raw_test_metrics,
        "stage1_test": proj_test_metrics,
        "delta_t2i_R@1": proj_test_metrics["t2i_R@1"] - raw_test_metrics["t2i_R@1"],
        "delta_i2t_R@1": proj_test_metrics["i2t_R@1"] - raw_test_metrics["i2t_R@1"],
        "delta_rsum": proj_test_metrics["rsum"] - raw_test_metrics["rsum"],
    }
    save_json(output_dir / "compare_backbone_vs_stage1.json", comparison)

    print("\n[Stage1 / Val]")
    for k, v in proj_val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Stage1 / Test]")
    for k, v in proj_test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()