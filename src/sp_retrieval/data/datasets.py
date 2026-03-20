from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class RetrievalJsonDataset(Dataset):
    def __init__(self, annotation_json: str, images_root: str, split: str, image_transform=None):
        self.annotation_json = annotation_json
        self.images_root = Path(images_root)
        self.split = split
        self.image_transform = image_transform

        with open(annotation_json, "r", encoding="utf-8") as f:
            rows = json.load(f)

        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No samples found for split={split}")

    def __len__(self):
        return len(self.rows)

    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        p = self.images_root / rel_or_abs
        if p.exists():
            return p
        p2 = Path(rel_or_abs)
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Cannot find image: {rel_or_abs}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = self._resolve_image_path(row["image"])
        img = Image.open(img_path).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)

        return {
            "image": img,
            "text": row["caption"],
            "image_id": int(row["image_id"]),
            "text_id": int(row["text_id"]),
            "image_path": str(img_path),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    first = images[0]

    if torch.is_tensor(first):
        images_out = torch.stack(images, dim=0)
    else:
        images_out = images  # list[PIL.Image] for processor-based backbones

    return {
        "images": images_out,
        "texts": [x["text"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "text_ids": [x["text_id"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
    }