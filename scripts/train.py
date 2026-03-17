import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import set_seed, resolve_device, count_parameters
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.training.build import build_model_and_optim
from sp_retrieval.training.trainer import fit
from sp_retrieval.training.eval import retrieval_metrics


def pretty_print_metrics(metrics):
    for name, vals in metrics.items():
        print(f'[{name}]')
        for k, v in vals.items():
            print(f'  {k}: {v:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = resolve_device(cfg['device'])

    backbone, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    backbone = backbone.to(device)

    model, fusion, optimizer = build_model_and_optim(cfg, backbone.output_dim, device)
    print(f'Device: {device}')
    print(f'Backbone output dim: {backbone.output_dim}')
    print(f'Trainable params(model+fusion): {count_parameters(model) + count_parameters(fusion):,}')

    best_path, best_score = fit(model, fusion, backbone, train_loader, val_loader, optimizer, device, cfg)
    print(f'Best checkpoint: {best_path}')
    print(f'Best val fused rsum: {best_score:.4f}')

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    fusion.load_state_dict(ckpt['fusion'])
    print('Best fusion weights:', ckpt.get('fusion_weights'))

    metrics = retrieval_metrics(model, fusion, backbone, test_loader, device)
    print('Test metrics:')
    pretty_print_metrics(metrics)


if __name__ == '__main__':
    main()
