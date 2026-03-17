from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset


class CandidateRerankDataset(Dataset):
    """
    Uses stage1 candidate cache.
    Each item = one text query + topK candidate images.
    """

    def __init__(
        self,
        annotation_json: str,
        images_root: str,
        candidate_cache: str,
        image_transform,
        split: str,
    ):
        self.images_root = Path(images_root)
        self.image_transform = image_transform

        with open(annotation_json, "r", encoding="utf-8") as f:
            rows = json.load(f)

        with open(candidate_cache, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            raise ValueError(f"No rows found for split={split}")

        self.textid_to_text = {}
        self.textid_to_pos_imgid = {}
        self.imageid_to_path = {}

        for r in split_rows:
            text_id = int(r["text_id"])
            image_id = int(r["image_id"])
            self.textid_to_text[text_id] = r["caption"]
            self.textid_to_pos_imgid[text_id] = image_id
            if image_id not in self.imageid_to_path:
                self.imageid_to_path[image_id] = r["image"]

        valid = []
        for c in candidates:
            tid = int(c["text_id"])
            if tid in self.textid_to_text:
                valid.append(c)

        self.records = valid
        if not self.records:
            raise ValueError("No valid candidate records matched the split annotation set.")

    def __len__(self):
        return len(self.records)

    def _load_image(self, image_id: int):
        rel = self.imageid_to_path[image_id]
        path = self.images_root / rel
        if not path.exists():
            path = Path(rel)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find image for image_id={image_id}: {rel}")
        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img

    def __getitem__(self, idx):
        rec = self.records[idx]
        text_id = int(rec["text_id"])
        query_text = self.textid_to_text[text_id]
        pos_img_id = int(rec["positive_image_id"])

        cand_ids = [int(x) for x in rec["candidate_image_ids"]]
        cand_scores = [float(x) for x in rec["candidate_scores"]]

        if pos_img_id not in cand_ids:
            cand_ids = [pos_img_id] + cand_ids[:-1]
            cand_scores = [cand_scores[0] if len(cand_scores) else 0.0] + cand_scores[:-1]

        cand_imgs = [self._load_image(img_id) for img_id in cand_ids]
        labels = [1 if img_id == pos_img_id else 0 for img_id in cand_ids]

        return {
            "text_id": text_id,
            "query_text": query_text,
            "positive_image_id": pos_img_id,
            "candidate_image_ids": cand_ids,
            "candidate_images": cand_imgs,
            "candidate_scores": cand_scores,
            "labels": labels,
        }


def rerank_collate_fn(batch: List[Dict]):
    bsz = len(batch)
    topk = len(batch[0]["candidate_image_ids"])

    candidate_images = torch.stack(
        [torch.stack(x["candidate_images"], dim=0) for x in batch], dim=0
    )  # [B,K,C,H,W]

    candidate_scores = torch.tensor(
        [x["candidate_scores"] for x in batch], dtype=torch.float32
    )  # [B,K]

    labels = torch.tensor(
        [x["labels"] for x in batch], dtype=torch.long
    )  # [B,K]

    return {
        "text_ids": [x["text_id"] for x in batch],
        "query_texts": [x["query_text"] for x in batch],
        "positive_image_ids": [x["positive_image_id"] for x in batch],
        "candidate_image_ids": [x["candidate_image_ids"] for x in batch],
        "candidate_images": candidate_images,
        "candidate_scores": candidate_scores,
        "labels": labels,
        "topk": topk,
        "batch_size": bsz,
    }