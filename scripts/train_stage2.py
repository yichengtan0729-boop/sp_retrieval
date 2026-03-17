import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import set_seed, resolve_device, ensure_dir, save_json, count_parameters
from sp_retrieval.models.backbone import build_backbone
from sp_retrieval.models.reranker import SharedPrivateReranker
from sp_retrieval.data.candidates import CandidateRerankDataset, rerank_collate_fn
from sp_retrieval.training.trainer_stage2 import (
    fit_stage2,
    evaluate_stage2,
    save_eval_rows_csv,
)


def build_stage2_loaders(cfg):
    dcfg = cfg["dataset"]
    backbone = build_backbone(cfg["backbone"])

    train_ds = CandidateRerankDataset(
        annotation_json=dcfg["annotation_json"],
        images_root=dcfg["images_root"],
        candidate_cache=dcfg["candidate_cache_train"],
        image_transform=backbone.preprocess,
        split=dcfg["train_split"],
    )
    val_ds = CandidateRerankDataset(
        annotation_json=dcfg["annotation_json"],
        images_root=dcfg["images_root"],
        candidate_cache=dcfg["candidate_cache_val"],
        image_transform=backbone.preprocess,
        split=dcfg["val_split"],
    )
    test_ds = CandidateRerankDataset(
        annotation_json=dcfg["annotation_json"],
        images_root=dcfg["images_root"],
        candidate_cache=dcfg["candidate_cache_test"],
        image_transform=backbone.preprocess,
        split=dcfg["test_split"],
    )

    kwargs = dict(
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        pin_memory=dcfg.get("pin_memory", True),
        collate_fn=rerank_collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs)

    return backbone, train_loader, val_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    output_dir = ensure_dir(cfg["output_dir"])

    backbone, train_loader, val_loader, test_loader = build_stage2_loaders(cfg)
    backbone = backbone.to(device)
    backbone.eval()

    in_dim = backbone.token_dim if backbone.token_dim is not None else backbone.output_dim

    reranker = SharedPrivateReranker(
        in_dim=in_dim,
        token_dim=cfg["model"]["token_dim"],
        num_shared_slots=cfg["model"]["num_shared_slots"],
        num_heads=cfg["model"].get("attn_heads", 8),
        dropout=cfg["model"].get("dropout", 0.1),
        use_redundant_branch=cfg["model"].get("use_redundant_branch", True),
    ).to(device)

    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    print(f"Device: {device}")
    print(f"Backbone token dim: {in_dim}")
    print(f"Reranker trainable params: {count_parameters(reranker):,}")

    best_path, best_score = fit_stage2(
        backbone=backbone,
        reranker=reranker,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        output_dir=output_dir,
    )

    print(f"\nBest stage2 checkpoint: {best_path}")
    print(f"Best val score: {best_score:.4f}")

    ckpt = torch.load(best_path, map_location=device)
    reranker.load_state_dict(ckpt["reranker"])

    val_metrics, val_rows = evaluate_stage2(backbone, reranker, val_loader, device)
    test_metrics, test_rows = evaluate_stage2(backbone, reranker, test_loader, device)

    save_json(output_dir / "metrics_stage2_val.json", val_metrics)
    save_json(output_dir / "metrics_stage2_test.json", test_metrics)
    save_eval_rows_csv(output_dir / "retrieval_stage2_val.csv", val_rows)
    save_eval_rows_csv(output_dir / "retrieval_stage2_test.csv", test_rows)

    comparison = {
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(output_dir / "compare_backbone_vs_rerank.json", comparison)

    print("\n[Stage2 / Val]")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Stage2 / Test]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()