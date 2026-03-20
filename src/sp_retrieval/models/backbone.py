from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image


@dataclass
class BackboneConfig:
    type: str
    model_name: str
    pretrained: Optional[str] = None
    freeze: bool = True
    return_tokens: bool = False
    max_text_len: Optional[int] = None


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim: Optional[int] = None
        self.token_dim: Optional[int] = None
        self.freeze: bool = True
        self.preprocess = None
        self.uses_processor_images = False

    def maybe_no_grad(self):
        return torch.no_grad() if self.freeze else torch.enable_grad()

    def forward(
        self,
        images,
        texts: List[str],
        device: torch.device,
        return_tokens: Optional[bool] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


class OpenCLIPBackbone(BaseBackbone):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(cfg.model_name)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.freeze = cfg.freeze
        self.uses_processor_images = False

        if hasattr(model, "text_projection") and model.text_projection is not None:
            self.output_dim = model.text_projection.shape[1]
        else:
            self.output_dim = model.transformer.width

        self.token_dim = self.output_dim

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with self.maybe_no_grad():
            x = self.model.encode_image(images).float()
            return F.normalize(x, dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(texts).to(device)
        with self.maybe_no_grad():
            x = self.model.encode_text(toks).float()
            return F.normalize(x, dim=-1)

    def _encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual

        if all(hasattr(visual, x) for x in ["conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer"]):
            x = visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            cls = visual.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)

            pos = visual.positional_embedding.to(x.dtype)
            x = x + pos[: x.shape[1]]

            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)

            if hasattr(visual, "ln_post") and visual.ln_post is not None:
                x = visual.ln_post(x)

            return F.normalize(x.float(), dim=-1)

        g = self.encode_image(images)
        return g.unsqueeze(1)

    def _encode_text_tokens(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(texts).to(device)

        if hasattr(self.model, "token_embedding") and hasattr(self.model, "transformer"):
            x = self.model.token_embedding(toks).to(self.model.dtype)
            x = x + self.model.positional_embedding.to(x.dtype)
            x = x.permute(1, 0, 2)
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            x = x.permute(1, 0, 2)
            x = self.model.ln_final(x)
            return F.normalize(x.float(), dim=-1)

        g = self.encode_text(texts, device)
        return g.unsqueeze(1)

    def forward(
        self,
        images,
        texts: List[str],
        device: torch.device,
        return_tokens: Optional[bool] = None,
    ):
        use_tokens = self.cfg.return_tokens if return_tokens is None else return_tokens

        if not torch.is_tensor(images):
            raise TypeError("OpenCLIPBackbone expects images as a batched tensor.")

        image_global = self.encode_image(images)
        text_global = self.encode_text(texts, device)

        out = {
            "image_global": image_global,
            "text_global": text_global,
            "image_tokens": None,
            "text_tokens": None,
            "text_mask": None,
            "image_mask": None,
        }

        if use_tokens:
            out["image_tokens"] = self._encode_image_tokens(images)
            out["text_tokens"] = self._encode_text_tokens(texts, device)
            toks = self.tokenizer(texts).to(device)
            out["text_mask"] = (toks != 0)

        return out


class SigLIP2Backbone(BaseBackbone):
    """
    Full-processor SigLIP2 path.

    Key points:
    - images should be a list of PIL.Image for SigLIP2
    - processor handles image/text preprocessing end-to-end
    - max_text_len is automatically clipped to model max_position_embeddings
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.freeze = cfg.freeze

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "SigLIP2 requires transformers. Install with: pip install transformers"
            ) from e

        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)

        hidden_size = None
        if hasattr(self.model.config, "projection_dim"):
            hidden_size = self.model.config.projection_dim
        elif hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            hidden_size = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "hidden_size"):
            hidden_size = self.model.config.vision_config.hidden_size

        if hidden_size is None:
            raise ValueError("Could not infer hidden size from SigLIP2 config.")

        self.output_dim = hidden_size
        self.token_dim = hidden_size

        self.preprocess = None
        self.uses_processor_images = True

        # text length cap
        self.model_max_text_len = None
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "max_position_embeddings"):
            self.model_max_text_len = int(self.model.config.text_config.max_position_embeddings)
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.model_max_text_len = int(self.model.config.max_position_embeddings)
        else:
            self.model_max_text_len = 64

        if cfg.max_text_len is None:
            self.max_text_len = self.model_max_text_len
        else:
            self.max_text_len = min(int(cfg.max_text_len), self.model_max_text_len)

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _prepare_inputs(
        self,
        images: Sequence[Image.Image],
        texts: List[str],
        device: torch.device,
    ):
        proc = self.processor(
            images=list(images),
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )
        return {k: v.to(device) for k, v in proc.items()}

    def _prepare_text_only(self, texts: List[str], device: torch.device):
        proc = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )
        return {k: v.to(device) for k, v in proc.items()}

    def encode_image(self, images, device: torch.device) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            raise TypeError("SigLIP2Backbone expects images as a list of PIL images.")
        proc = self.processor(
            images=list(images),
            return_tensors="pt",
        )
        pixel_values = proc["pixel_values"].to(device)

        with self.maybe_no_grad():
            x = self.model.get_image_features(pixel_values=pixel_values)
            return F.normalize(x.float(), dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        proc = self._prepare_text_only(texts, device)
        with self.maybe_no_grad():
            x = self.model.get_text_features(
                input_ids=proc["input_ids"],
                attention_mask=proc.get("attention_mask", None),
            )
            return F.normalize(x.float(), dim=-1)

    def forward(
        self,
        images,
        texts: List[str],
        device: torch.device,
        return_tokens: Optional[bool] = None,
    ):
        use_tokens = self.cfg.return_tokens if return_tokens is None else return_tokens

        if not isinstance(images, (list, tuple)):
            raise TypeError("SigLIP2Backbone forward expects images as a list of PIL images.")

        proc = self._prepare_inputs(images, texts, device)

        with self.maybe_no_grad():
            outputs = self.model(
                input_ids=proc["input_ids"],
                attention_mask=proc.get("attention_mask", None),
                pixel_values=proc["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )

        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        if image_embeds is None or text_embeds is None:
            raise ValueError("SigLIP2 outputs do not contain image_embeds/text_embeds.")

        out = {
            "image_global": F.normalize(image_embeds.float(), dim=-1),
            "text_global": F.normalize(text_embeds.float(), dim=-1),
            "image_tokens": None,
            "text_tokens": None,
            "image_mask": None,
            "text_mask": proc.get("attention_mask", None).bool() if "attention_mask" in proc else None,
        }

        if use_tokens:
            vision_hidden = None
            text_hidden = None

            if hasattr(outputs, "vision_model_output") and outputs.vision_model_output is not None:
                vision_hidden = outputs.vision_model_output.last_hidden_state
            if hasattr(outputs, "text_model_output") and outputs.text_model_output is not None:
                text_hidden = outputs.text_model_output.last_hidden_state

            if vision_hidden is not None:
                out["image_tokens"] = F.normalize(vision_hidden.float(), dim=-1)
            if text_hidden is not None:
                out["text_tokens"] = F.normalize(text_hidden.float(), dim=-1)

        return out


def build_backbone(cfg_dict):
    cfg = BackboneConfig(**cfg_dict)

    if cfg.type == "open_clip":
        return OpenCLIPBackbone(cfg)
    if cfg.type == "siglip2":
        return SigLIP2Backbone(cfg)

    raise ValueError(f"Unsupported backbone type: {cfg.type}")