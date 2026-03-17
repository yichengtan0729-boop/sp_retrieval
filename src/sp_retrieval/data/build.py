from torch.utils.data import DataLoader

from ..models.backbone import build_backbone
from .datasets import RetrievalJsonDataset, collate_fn


def build_dataloaders(cfg):
    dcfg = cfg["dataset"]
    backbone = build_backbone(cfg["backbone"])
    transform = backbone.preprocess

    train_ds = RetrievalJsonDataset(
        dcfg["annotation_json"],
        dcfg["images_root"],
        dcfg["train_split"],
        transform,
    )
    val_ds = RetrievalJsonDataset(
        dcfg["annotation_json"],
        dcfg["images_root"],
        dcfg["val_split"],
        transform,
    )
    test_ds = RetrievalJsonDataset(
        dcfg["annotation_json"],
        dcfg["images_root"],
        dcfg["test_split"],
        transform,
    )

    kwargs = dict(
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=dcfg.get("pin_memory", True),
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs)

    return backbone, train_loader, val_loader, test_loader