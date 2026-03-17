from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_embeddings(model, backbone, loader, device):
    model.eval()
    backbone.eval()
    ux_all, uy_all = [], []
    sx_all, sy_all = [], []
    image_ids, text_ids = [], []
    for batch in loader:
        images = batch['images'].to(device, non_blocking=True)
        texts = batch['texts']
        ux = backbone.encode_image(images)
        uy = backbone.encode_text(texts, device)
        out = model.encode_and_decompose(ux, uy)
        ux_all.append(F.normalize(ux, dim=-1).cpu())
        uy_all.append(F.normalize(uy, dim=-1).cpu())
        sx_all.append(F.normalize(out['sx'], dim=-1).cpu())
        sy_all.append(F.normalize(out['sy'], dim=-1).cpu())
        image_ids.extend(batch['image_ids'])
        text_ids.extend(batch['text_ids'])
    return {
        'ux': torch.cat(ux_all, 0),
        'uy': torch.cat(uy_all, 0),
        'sx': torch.cat(sx_all, 0),
        'sy': torch.cat(sy_all, 0),
        'image_ids': image_ids,
        'text_ids': text_ids,
    }


def build_unique_image_text_views(img_emb: torch.Tensor, txt_emb: torch.Tensor, image_ids: List[int]):
    image_to_txts: Dict[int, List[int]] = defaultdict(list)
    image_to_first_idx: Dict[int, int] = {}
    unique_img_ids: List[int] = []
    unique_img_embs = []
    txt_to_unique_img = []
    for txt_idx, img_id in enumerate(image_ids):
        image_to_txts[img_id].append(txt_idx)
        if img_id not in image_to_first_idx:
            image_to_first_idx[img_id] = txt_idx
            unique_img_ids.append(img_id)
            unique_img_embs.append(img_emb[txt_idx])
        txt_to_unique_img.append(None)
    img_id_to_unique = {img_id: i for i, img_id in enumerate(unique_img_ids)}
    for txt_idx, img_id in enumerate(image_ids):
        txt_to_unique_img[txt_idx] = img_id_to_unique[img_id]
    img_to_txts_unique = [image_to_txts[img_id] for img_id in unique_img_ids]
    unique_img_embs = torch.stack(unique_img_embs, dim=0)
    return unique_img_embs, txt_emb, img_to_txts_unique, txt_to_unique_img


def recall_i2t(sim: torch.Tensor, img_to_txts: List[List[int]], k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    hits = []
    for i, positives in enumerate(img_to_txts):
        pos = set(positives)
        pred = topk[i].tolist()
        hits.append(any(p in pos for p in pred))
    return float(torch.tensor(hits, dtype=torch.float32).mean().item())


def recall_t2i(sim: torch.Tensor, txt_to_img: List[int], k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    hits = []
    for txt_idx, img_idx in enumerate(txt_to_img):
        hits.append(img_idx in topk[txt_idx].tolist())
    return float(torch.tensor(hits, dtype=torch.float32).mean().item())


def summarize_sim(sim_i2t: torch.Tensor, img_to_txts: List[List[int]], txt_to_img: List[int], ks=(1, 5, 10)):
    out = {}
    for k in ks:
        out[f'i2t_R@{k}'] = recall_i2t(sim_i2t, img_to_txts, k)
        out[f't2i_R@{k}'] = recall_t2i(sim_i2t.t(), txt_to_img, k)
    out['rsum'] = sum(out.values())
    out['mean_R@1'] = 0.5 * (out['i2t_R@1'] + out['t2i_R@1'])
    return out


@torch.no_grad()
def retrieval_metrics(model, fusion, backbone, loader, device, ks=(1, 5, 10)):
    feats = collect_embeddings(model, backbone, loader, device)

    ux_img, uy_txt, img_to_txts, txt_to_img = build_unique_image_text_views(feats['ux'], feats['uy'], feats['image_ids'])
    sx_img, sy_txt, _, _ = build_unique_image_text_views(feats['sx'], feats['sy'], feats['image_ids'])

    sim_clip = ux_img @ uy_txt.t()
    sim_shared = sx_img @ sy_txt.t()
    sim_fused = fusion(sim_clip.to(device), sim_shared.to(device)).cpu()

    metrics = {
        'clip': summarize_sim(sim_clip, img_to_txts, txt_to_img, ks=ks),
        'shared': summarize_sim(sim_shared, img_to_txts, txt_to_img, ks=ks),
        'fused': summarize_sim(sim_fused, img_to_txts, txt_to_img, ks=ks),
    }
    return metrics
