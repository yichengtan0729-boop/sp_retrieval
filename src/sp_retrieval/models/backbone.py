from __future__ import annotations
from dataclasses import dataclass
import open_clip
import torch
import torch.nn as nn


@dataclass
class BackboneConfig:
    type: str
    model_name: str
    pretrained: str
    freeze: bool = True


class OpenCLIPBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)
        tokenizer = open_clip.get_tokenizer(cfg.model_name)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        if hasattr(model, 'text_projection') and model.text_projection is not None:
            self.output_dim = model.text_projection.shape[1]
        else:
            self.output_dim = model.transformer.width
        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.freeze = cfg.freeze

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model.encode_image(images).float()
        return self.model.encode_image(images).float()

    def encode_text(self, texts, device) -> torch.Tensor:
        toks = self.tokenizer(texts).to(device)
        if self.freeze:
            with torch.no_grad():
                return self.model.encode_text(toks).float()
        return self.model.encode_text(toks).float()


def build_backbone(cfg_dict):
    cfg = BackboneConfig(**cfg_dict)
    if cfg.type != 'open_clip':
        raise ValueError(f'Unsupported backbone type: {cfg.type}')
    return OpenCLIPBackbone(cfg)
