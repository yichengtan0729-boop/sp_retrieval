from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, List
from PIL import Image
from torch.utils.data import Dataset


class RetrievalJsonDataset(Dataset):
    def __init__(self, annotation_json: str, images_root: str, split: str, image_transform):
        self.annotation_json = Path(annotation_json)
        self.images_root = Path(images_root)
        self.split = split
        self.image_transform = image_transform
        with open(self.annotation_json, 'r', encoding='utf-8') as f:
            items = json.load(f)
        self.samples: List[Dict[str, Any]] = [x for x in items if x['split'] == split]
        if len(self.samples) == 0:
            raise ValueError(f'No samples found for split={split} in {annotation_json}')

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, rel_path: str) -> Path:
        p = self.images_root / rel_path
        if p.exists():
            return p
        p2 = Path(rel_path)
        if p2.exists():
            return p2
        raise FileNotFoundError(f'Cannot find image: {rel_path}')

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        image_path = self._resolve_image_path(row['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        return {
            'image': image,
            'text': row['caption'],
            'image_id': row['image_id'],
            'text_id': row['text_id'],
            'image_path': str(image_path),
        }


def collate_fn(batch):
    import torch
    images = torch.stack([x['image'] for x in batch], 0)
    texts = [x['text'] for x in batch]
    image_ids = [x['image_id'] for x in batch]
    text_ids = [x['text_id'] for x in batch]
    image_paths = [x['image_path'] for x in batch]
    return {
        'images': images,
        'texts': texts,
        'image_ids': image_ids,
        'text_ids': text_ids,
        'image_paths': image_paths,
    }
