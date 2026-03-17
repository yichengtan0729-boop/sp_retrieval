import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp_retrieval.utils.config import load_config
from sp_retrieval.utils.misc import resolve_device
from sp_retrieval.data.build import build_dataloaders
from sp_retrieval.training.build import build_model_and_optim


def compute_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def build_gt(image_ids, text_ids):
    """
    image_ids: [N_imglike]
    text_ids:  [N_txtlike] (这里只是顺序索引配套文本列表，不直接参与映射)
    实际正样本关系由 image_id 决定：
      - i2t: 同 image_id 的所有文本都是正样本
      - t2i: 每个文本的正图像是与其 image_id 对应的唯一图像索引
    """
    img_to_txt = defaultdict(list)
    txt_to_img = {}

    unique_img_ids = []
    seen = {}
    for i, img_id in enumerate(image_ids):
        if img_id not in seen:
            seen[img_id] = len(unique_img_ids)
            unique_img_ids.append(img_id)

    # 这里 image_ids 应该已经是“每个文本样本对应的 image_id”
    # 我们之后会按 unique image 聚合图像特征，所以要建立 image_id -> image_index
    image_id_to_unique_index = {img_id: idx for idx, img_id in enumerate(unique_img_ids)}

    for txt_idx, img_id in enumerate(image_ids):
        img_idx = image_id_to_unique_index[img_id]
        img_to_txt[img_idx].append(txt_idx)
        txt_to_img[txt_idx] = img_idx

    return img_to_txt, txt_to_img, unique_img_ids


def recall_i2t(sim, img_to_txt, ks=(1, 5, 10)):
    max_k = max(ks)
    topk = sim.topk(max_k, dim=1).indices
    out = {}

    for k in ks:
        correct = 0
        for i in range(sim.size(0)):
            preds = topk[i, :k].tolist()
            pos = set(img_to_txt[i])
            if any(p in pos for p in preds):
                correct += 1
        out[k] = correct / sim.size(0)
    return out


def recall_t2i(sim, txt_to_img, ks=(1, 5, 10)):
    max_k = max(ks)
    topk = sim.topk(max_k, dim=1).indices
    out = {}

    for k in ks:
        correct = 0
        for i in range(sim.size(0)):
            preds = topk[i, :k].tolist()
            pos = txt_to_img[i]
            if pos in preds:
                correct += 1
        out[k] = correct / sim.size(0)
    return out


def report_metrics(name, sim_img_txt, img_to_txt, txt_to_img):
    i2t = recall_i2t(sim_img_txt, img_to_txt, ks=(1, 5, 10))
    t2i = recall_t2i(sim_img_txt.t(), txt_to_img, ks=(1, 5, 10))

    rsum = i2t[1] + i2t[5] + i2t[10] + t2i[1] + t2i[5] + t2i[10]

    print(f"\n[{name}]")
    print("i2t_R@1 :", i2t[1])
    print("i2t_R@5 :", i2t[5])
    print("i2t_R@10:", i2t[10])
    print("t2i_R@1 :", t2i[1])
    print("t2i_R@5 :", t2i[5])
    print("t2i_R@10:", t2i[10])
    print("rsum    :", rsum)

    return {
        "i2t_r1": i2t[1],
        "i2t_r5": i2t[5],
        "i2t_r10": i2t[10],
        "t2i_r1": t2i[1],
        "t2i_r5": t2i[5],
        "t2i_r10": t2i[10],
        "rsum": rsum,
    }


@torch.no_grad()
def collect_outputs(model, backbone, loader, device):
    """
    假设 loader 是 caption-level：
      每条样本有一张图 + 一条文本 + 一个 image_id
    我们收集：
      - 文本侧：全部文本特征
      - 图像侧：按 image_id 去重后保留唯一图像特征
    """
    model.eval()
    backbone.eval()

    all_texts_ux = []
    all_texts_uy = []
    all_texts_sx = []
    all_texts_sy = []
    all_texts_px = []
    all_texts_py = []
    all_text_image_ids = []

    unique_image_bank = {}

    for batch in tqdm(loader, desc="collect"):
        images = batch["images"].to(device)
        texts = batch["texts"]
        image_ids = batch["image_ids"]

        ux = backbone.encode_image(images)
        uy = backbone.encode_text(texts, device)
        out = model.encode_and_decompose(ux, uy)

        # 文本特征全部保留
        all_texts_uy.append(out["uy"].cpu())
        all_texts_sy.append(out["sy"].cpu())
        all_texts_py.append(out["py"].cpu())

        # image_id 按样本顺序保留，用于建立 txt -> img 映射
        if torch.is_tensor(image_ids):
            image_ids_list = image_ids.cpu().tolist()
        else:
            image_ids_list = list(image_ids)

        all_text_image_ids.extend(image_ids_list)

        # 图像特征按 image_id 去重保存
        ux_cpu = out["ux"].cpu()
        sx_cpu = out["sx"].cpu()
        px_cpu = out["px"].cpu()

        for i, img_id in enumerate(image_ids_list):
            if img_id not in unique_image_bank:
                unique_image_bank[img_id] = {
                    "ux": ux_cpu[i],
                    "sx": sx_cpu[i],
                    "px": px_cpu[i],
                }

    # 按 image_id 排序构建唯一图像特征矩阵
    unique_img_ids = sorted(unique_image_bank.keys())

    ux_img = torch.stack([unique_image_bank[i]["ux"] for i in unique_img_ids], dim=0)
    sx_img = torch.stack([unique_image_bank[i]["sx"] for i in unique_img_ids], dim=0)
    px_img = torch.stack([unique_image_bank[i]["px"] for i in unique_img_ids], dim=0)

    uy_txt = torch.cat(all_texts_uy, dim=0)
    sy_txt = torch.cat(all_texts_sy, dim=0)
    py_txt = torch.cat(all_texts_py, dim=0)

    # ground truth
    image_id_to_index = {img_id: idx for idx, img_id in enumerate(unique_img_ids)}

    img_to_txt = defaultdict(list)
    txt_to_img = {}

    for txt_idx, img_id in enumerate(all_text_image_ids):
        img_idx = image_id_to_index[img_id]
        img_to_txt[img_idx].append(txt_idx)
        txt_to_img[txt_idx] = img_idx

    return {
        "ux_img": ux_img,
        "sx_img": sx_img,
        "px_img": px_img,
        "uy_txt": uy_txt,
        "sy_txt": sy_txt,
        "py_txt": py_txt,
        "img_to_txt": img_to_txt,
        "txt_to_img": txt_to_img,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--w_clip", type=float, default=0.8)
    parser.add_argument("--w_shared", type=float, default=0.2)
    parser.add_argument("--w_private", type=float, default=0.1)
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

    data = collect_outputs(model, backbone, loader, device)

    ux_img = data["ux_img"]
    sx_img = data["sx_img"]
    px_img = data["px_img"]

    uy_txt = data["uy_txt"]
    sy_txt = data["sy_txt"]
    py_txt = data["py_txt"]

    img_to_txt = data["img_to_txt"]
    txt_to_img = data["txt_to_img"]

    sim_clip = compute_sim(ux_img, uy_txt)
    sim_shared = compute_sim(sx_img, sy_txt)
    sim_private = compute_sim(px_img, py_txt)

    sim_clip_shared = args.w_clip * sim_clip + args.w_shared * sim_shared
    sim_all = (
        args.w_clip * sim_clip
        + args.w_shared * sim_shared
        + args.w_private * sim_private
    )

    report_metrics("clip", sim_clip, img_to_txt, txt_to_img)
    report_metrics("shared", sim_shared, img_to_txt, txt_to_img)
    report_metrics("private", sim_private, img_to_txt, txt_to_img)
    report_metrics(
        f"clip+shared ({args.w_clip:.2f},{args.w_shared:.2f})",
        sim_clip_shared,
        img_to_txt,
        txt_to_img,
    )
    report_metrics(
        f"clip+shared+private ({args.w_clip:.2f},{args.w_shared:.2f},{args.w_private:.2f})",
        sim_all,
        img_to_txt,
        txt_to_img,
    )


if __name__ == "__main__":
    main()