from __future__ import annotations
from pathlib import Path
import torch
from tqdm import tqdm
from ..utils.misc import ensure_dir, append_jsonl
from .eval import retrieval_metrics


def train_one_epoch(model, fusion, backbone, loader, optimizer, device, cfg_train):
    model.train()
    fusion.train()
    running = {
        'loss': 0.0,
        'loss_align': 0.0,
        'loss_recon': 0.0,
        'loss_private': 0.0,
        'loss_ortho': 0.0,
        'loss_var': 0.0,
        'loss_fusion_reg': 0.0,
    }
    n_steps = 0
    pbar = tqdm(loader, desc='train', leave=False)
    for batch in pbar:
        images = batch['images'].to(device, non_blocking=True)
        texts = batch['texts']
        ux = backbone.encode_image(images)
        uy = backbone.encode_text(texts, device)
        out = model.encode_and_decompose(ux, uy)
        losses = model.compute_losses(out)

        s_clip = torch.nn.functional.normalize(ux, dim=-1) @ torch.nn.functional.normalize(uy, dim=-1).t()
        s_shared = torch.nn.functional.normalize(out['sx'], dim=-1) @ torch.nn.functional.normalize(out['sy'], dim=-1).t()
        s_fused = fusion(s_clip, s_shared)
        target = torch.arange(s_fused.size(0), device=device)
        loss_fused = 0.5 * (
            torch.nn.functional.cross_entropy(s_fused / cfg_train['fusion_temperature'], target) +
            torch.nn.functional.cross_entropy(s_fused.t() / cfg_train['fusion_temperature'], target)
        )
        loss_fusion_reg = torch.nn.functional.mse_loss(s_fused, fusion._norm(s_clip))
        total = losses['loss'] + cfg_train['lambda_fused'] * loss_fused + cfg_train['lambda_fusion_reg'] * loss_fusion_reg

        optimizer.zero_grad()
        total.backward()
        if cfg_train.get('grad_clip_norm', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(fusion.parameters()), cfg_train['grad_clip_norm'])
        optimizer.step()

        n_steps += 1
        running['loss'] += float(total.item())
        running['loss_align'] += float(losses['loss_align'].item())
        running['loss_recon'] += float(losses['loss_recon'].item())
        running['loss_private'] += float(losses['loss_private'].item())
        running['loss_ortho'] += float(losses['loss_ortho'].item())
        running['loss_var'] += float(losses['loss_var'].item())
        running['loss_fusion_reg'] += float(loss_fusion_reg.item())
        w = fusion.get_weights().detach().cpu().tolist()
        pbar.set_postfix(loss=f'{total.item():.4f}', w=f'[{w[0]:.3f},{w[1]:.3f}]')
    return {k: v / max(n_steps, 1) for k, v in running.items()}


def fit(model, fusion, backbone, train_loader, val_loader, optimizer, device, cfg):
    output_dir = ensure_dir(cfg['output_dir'])
    ckpt_dir = ensure_dir(Path(output_dir) / 'checkpoints')
    log_path = Path(output_dir) / 'train_log.jsonl'
    best_score = -1.0
    best_path = ckpt_dir / 'best.pt'
    train_cfg = cfg['train']

    for epoch in range(1, train_cfg['epochs'] + 1):
        train_stats = train_one_epoch(model, fusion, backbone, train_loader, optimizer, device, train_cfg)
        val_metrics = retrieval_metrics(model, fusion, backbone, val_loader, device)
        payload = {
            'epoch': epoch,
            'train': train_stats,
            'val': val_metrics,
            'fusion_weights': [float(x) for x in fusion.get_weights().detach().cpu().tolist()],
        }
        append_jsonl(log_path, payload)
        monitor = val_metrics['fused']['rsum']
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'fusion': fusion.state_dict(),
            'monitor': monitor,
            'config': cfg,
            'fusion_weights': payload['fusion_weights'],
        }
        torch.save(ckpt, ckpt_dir / 'last.pt')
        if monitor > best_score:
            best_score = monitor
            torch.save(ckpt, best_path)
    return best_path, best_score
