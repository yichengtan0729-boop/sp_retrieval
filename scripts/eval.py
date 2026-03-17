import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import resolve_device
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.training.build import build_model_and_optim
from sp_retrieval.training.eval import retrieval_metrics


def pretty_print_metrics(metrics):
    for name, vals in metrics.items():
        print(f'[{name}]')
        for k, v in vals.items():
            print(f'{k}: {v:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg['device'])
    backbone, _, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == 'val' else test_loader
    backbone = backbone.to(device)
    model, fusion, _ = build_model_and_optim(cfg, backbone.output_dim, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    fusion.load_state_dict(ckpt['fusion'])
    metrics = retrieval_metrics(model, fusion, backbone, loader, device)
    pretty_print_metrics(metrics)


if __name__ == '__main__':
    main()
