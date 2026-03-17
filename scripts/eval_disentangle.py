import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import resolve_device
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.training.build import build_model_and_optim


def mean_abs_diag_cos(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).abs().mean().item()


def mean_diag_cos(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).mean().item()


def cross_cov_score(a, b):
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    a = a / (a.std(dim=0, keepdim=True) + 1e-6)
    b = b / (b.std(dim=0, keepdim=True) + 1e-6)

    c = (a.T @ b) / max(a.size(0) - 1, 1)
    return c.abs().mean().item(), c.norm(p="fro").item()


def feature_std(a):
    return a.std(dim=0).mean().item()


@torch.no_grad()
def collect_outputs(model, backbone, loader, device):
    all_ux, all_uy = [], []
    all_sx, all_sy = [], []
    all_px, all_py = [], []

    model.eval()
    backbone.eval()

    for batch in tqdm(loader, desc="collect"):
        images = batch["images"].to(device)
        texts = batch["texts"]

        ux = backbone.encode_image(images)
        uy = backbone.encode_text(texts, device)

        out = model.encode_and_decompose(ux, uy)

        all_ux.append(out["ux"].cpu())
        all_uy.append(out["uy"].cpu())
        all_sx.append(out["sx"].cpu())
        all_sy.append(out["sy"].cpu())
        all_px.append(out["px"].cpu())
        all_py.append(out["py"].cpu())

    return {
        "ux": torch.cat(all_ux, dim=0),
        "uy": torch.cat(all_uy, dim=0),
        "sx": torch.cat(all_sx, dim=0),
        "sy": torch.cat(all_sy, dim=0),
        "px": torch.cat(all_px, dim=0),
        "py": torch.cat(all_py, dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    backbone, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    backbone = backbone.to(device)

    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    model, fusion, optimizer = build_model_and_optim(cfg, backbone.output_dim, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    outputs = collect_outputs(model, backbone, loader, device)

    ux, uy = outputs["ux"], outputs["uy"]
    sx, sy = outputs["sx"], outputs["sy"]
    px, py = outputs["px"], outputs["py"]

    print("\n==== Disentanglement diagnostics ====")

    print("\n[1] shared-private within-modality orthogonality")
    print("img | mean abs cos(sx, px):", mean_abs_diag_cos(sx, px))
    print("txt | mean abs cos(sy, py):", mean_abs_diag_cos(sy, py))

    m1, f1 = cross_cov_score(sx, px)
    m2, f2 = cross_cov_score(sy, py)
    print("img | cross-cov mean abs   :", m1)
    print("img | cross-cov fro norm   :", f1)
    print("txt | cross-cov mean abs   :", m2)
    print("txt | cross-cov fro norm   :", f2)

    print("\n[2] cross-modal alignment")
    print("shared | mean cos(sx, sy):", mean_diag_cos(sx, sy))
    print("private| mean cos(px, py):", mean_diag_cos(px, py))

    print("\n[3] leakage check")
    print("mean cos(sx, py):", mean_diag_cos(sx, py))
    print("mean cos(sy, px):", mean_diag_cos(sy, px))

    print("\n[4] feature variance")
    print("ux std:", feature_std(ux))
    print("uy std:", feature_std(uy))
    print("sx std:", feature_std(sx))
    print("sy std:", feature_std(sy))
    print("px std:", feature_std(px))
    print("py std:", feature_std(py))


if __name__ == "__main__":
    main()