import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import resolve_device
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.training.build import build_model_and_optim

from nnn import NNNRetriever, NNNRanker


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def recall_from_rank_lists_i2t(rank_lists, img_to_txt, ks=(1, 5, 10)):
    out = {}
    n_img = len(rank_lists)
    for k in ks:
        correct = 0
        for i, preds in enumerate(rank_lists):
            pos = set(img_to_txt[i])
            if any(p in pos for p in preds[:k]):
                correct += 1
        out[k] = correct / n_img
    return out


def recall_from_rank_lists_t2i(rank_lists, txt_to_img, ks=(1, 5, 10)):
    out = {}
    n_txt = len(rank_lists)
    for k in ks:
        correct = 0
        for i, preds in enumerate(rank_lists):
            if txt_to_img[i] in preds[:k]:
                correct += 1
        out[k] = correct / n_txt
    return out


def report(name, i2t_rank_lists, t2i_rank_lists, img_to_txt, txt_to_img):
    i2t = recall_from_rank_lists_i2t(i2t_rank_lists, img_to_txt)
    t2i = recall_from_rank_lists_t2i(t2i_rank_lists, txt_to_img)
    rsum = i2t[1] + i2t[5] + i2t[10] + t2i[1] + t2i[5] + t2i[10]

    print(f"\n[{name}]")
    print("i2t_R@1 :", i2t[1])
    print("i2t_R@5 :", i2t[5])
    print("i2t_R@10:", i2t[10])
    print("t2i_R@1 :", t2i[1])
    print("t2i_R@5 :", t2i[5])
    print("t2i_R@10:", t2i[10])
    print("rsum    :", rsum)


@torch.no_grad()
def collect_split_features(model, backbone, loader, device):
    model.eval()
    backbone.eval()

    all_text_img_ids = []
    all_uy, all_sy = [], []

    unique_img_bank = {}

    for batch in tqdm(loader, desc="collect"):
        images = batch["images"].to(device)
        texts = batch["texts"]
        image_ids = batch["image_ids"]

        ux = backbone.encode_image(images)
        uy = backbone.encode_text(texts, device)
        out = model.encode_and_decompose(ux, uy)

        uy_cpu = l2norm(out["uy"]).cpu()
        sy_cpu = l2norm(out["sy"]).cpu()
        ux_cpu = l2norm(out["ux"]).cpu()
        sx_cpu = l2norm(out["sx"]).cpu()

        if torch.is_tensor(image_ids):
            image_ids = image_ids.cpu().tolist()
        else:
            image_ids = list(image_ids)

        all_uy.append(uy_cpu)
        all_sy.append(sy_cpu)
        all_text_img_ids.extend(image_ids)

        for i, img_id in enumerate(image_ids):
            if img_id not in unique_img_bank:
                unique_img_bank[img_id] = {
                    "ux": ux_cpu[i],
                    "sx": sx_cpu[i],
                }

    unique_img_ids = sorted(unique_img_bank.keys())

    ux_img = torch.stack([unique_img_bank[i]["ux"] for i in unique_img_ids], dim=0)
    sx_img = torch.stack([unique_img_bank[i]["sx"] for i in unique_img_ids], dim=0)

    uy_txt = torch.cat(all_uy, dim=0)
    sy_txt = torch.cat(all_sy, dim=0)

    image_id_to_index = {img_id: idx for idx, img_id in enumerate(unique_img_ids)}
    img_to_txt = defaultdict(list)
    txt_to_img = {}
    for txt_idx, img_id in enumerate(all_text_img_ids):
        img_idx = image_id_to_index[img_id]
        img_to_txt[img_idx].append(txt_idx)
        txt_to_img[txt_idx] = img_idx

    return {
        "ux_img": ux_img.numpy(),
        "sx_img": sx_img.numpy(),
        "uy_txt": uy_txt.numpy(),
        "sy_txt": sy_txt.numpy(),
        "img_to_txt": img_to_txt,
        "txt_to_img": txt_to_img,
    }


def build_rank_lists_from_scores(scores, indices):
    # NNN returns top-k scores and indices
    return [list(map(int, row)) for row in indices]


def brute_force_topk(query, db, top_k):
    sims = query @ db.T
    idx = np.argsort(-sims, axis=1)[:, :top_k]
    return [list(map(int, row)) for row in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ref_split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--branch", default="clip", choices=["clip", "shared"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alternate_ks", type=int, default=128)
    parser.add_argument("--alternate_weight", type=float, default=0.75)
    parser.add_argument("--use_gpu_nnn", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])

    backbone, train_loader, val_loader, test_loader = build_dataloaders(cfg)
    backbone = backbone.to(device)

    split_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    eval_loader = split_map[args.split]
    ref_loader = split_map[args.ref_split]

    model, fusion, optimizer = build_model_and_optim(cfg, backbone.output_dim, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    eval_data = collect_split_features(model, backbone, eval_loader, device)
    ref_data = collect_split_features(model, backbone, ref_loader, device)

    if args.branch == "clip":
        img_db = eval_data["ux_img"]
        txt_query = eval_data["uy_txt"]
        ref_txt = ref_data["uy_txt"]
        name_prefix = "clip+NNN"
        # vanilla baseline for comparison
        i2t_vanilla = brute_force_topk(img_db, txt_query, args.top_k)
        t2i_vanilla = brute_force_topk(txt_query, img_db, args.top_k)
    else:
        img_db = eval_data["sx_img"]
        txt_query = eval_data["sy_txt"]
        ref_txt = ref_data["sy_txt"]
        name_prefix = "shared+NNN"
        i2t_vanilla = brute_force_topk(img_db, txt_query, args.top_k)
        t2i_vanilla = brute_force_topk(txt_query, img_db, args.top_k)

    print(f"branch = {args.branch}")
    print(f"eval split = {args.split}")
    print(f"reference split = {args.ref_split}")

    report(
        f"{args.branch} vanilla",
        i2t_vanilla,
        t2i_vanilla,
        eval_data["img_to_txt"],
        eval_data["txt_to_img"],
    )

    # text -> image using NNN official API
    retriever = NNNRetriever(img_db.shape[1], use_gpu=args.use_gpu_nnn)
    ranker_t2i = NNNRanker(
        retriever,
        img_db,
        ref_txt,
        alternate_ks=args.alternate_ks,
        alternate_weight=args.alternate_weight,
        batch_size=256,
        use_gpu=args.use_gpu_nnn,
    )
    _, t2i_idx = ranker_t2i.search(txt_query, top_k=args.top_k)
    t2i_nnn = build_rank_lists_from_scores(None, t2i_idx)

    # image -> text: symmetric use, DB=texts, reference=image embeddings from ref split
    if args.branch == "clip":
        txt_db = eval_data["uy_txt"]
        img_query = eval_data["ux_img"]
        ref_img = ref_data["ux_img"]
    else:
        txt_db = eval_data["sy_txt"]
        img_query = eval_data["sx_img"]
        ref_img = ref_data["sx_img"]

    retriever2 = NNNRetriever(txt_db.shape[1], use_gpu=args.use_gpu_nnn)
    ranker_i2t = NNNRanker(
        retriever2,
        txt_db,
        ref_img,
        alternate_ks=args.alternate_ks,
        alternate_weight=args.alternate_weight,
        batch_size=256,
        use_gpu=args.use_gpu_nnn,
    )
    _, i2t_idx = ranker_i2t.search(img_query, top_k=args.top_k)
    i2t_nnn = build_rank_lists_from_scores(None, i2t_idx)

    report(
        name_prefix,
        i2t_nnn,
        t2i_nnn,
        eval_data["img_to_txt"],
        eval_data["txt_to_img"],
    )


if __name__ == "__main__":
    main()