from ..models.decomp import ModelConfig, SharedPrivateRetrievalModel
from ..models.fusion import ConservativeFusion
import torch


def build_model_and_optim(cfg, backbone_output_dim, device):
    mcfg = cfg['model']
    ocfg = cfg['optim']
    model_cfg = ModelConfig(
        input_dim=backbone_output_dim,
        hidden_dim=mcfg['hidden_dim'],
        shared_dim=mcfg['shared_dim'],
        private_dim=mcfg['private_dim'],
        temperature=mcfg['temperature'],
        lambda_recon=mcfg['lambda_recon'],
        lambda_private=mcfg['lambda_private'],
        lambda_ortho=mcfg['lambda_ortho'],
        lambda_var=mcfg['lambda_var'],
    )
    model = SharedPrivateRetrievalModel(model_cfg).to(device)
    fusion = ConservativeFusion(tuple(mcfg.get('fusion_init_logits', [2.2, 0.8]))).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(fusion.parameters()),
        lr=ocfg['lr'],
        weight_decay=ocfg['weight_decay'],
    )
    return model, fusion, optimizer
