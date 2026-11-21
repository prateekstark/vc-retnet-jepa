"""Shared utilities for Scaled-up Hybrid RetNet-JEPA Voice Conversion."""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from src.models.retnet_ae_jepa import (
    HybridRetNetAEJEPA,
    RetNetSpectrogramDecoder,
    SplitLatentRetNetEncoder,
)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device(cfg_device: str = "cuda") -> torch.device:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_hybrid_model(
    cfg: Dict[str, Any],
    retnet_mode: str | None = None,
    lambda_jepa_override: float | None = None,
) -> HybridRetNetAEJEPA:
    m_cfg = cfg["model"]
    a_cfg = cfg["audio"]
    p_cfg = cfg["pretrain"]
    mode = retnet_mode or m_cfg["retnet_mode"]
    l_jepa = lambda_jepa_override if lambda_jepa_override is not None else p_cfg["lambda_jepa"]

    encoder = SplitLatentRetNetEncoder(
        n_mels=a_cfg["n_mels"],
        crop_frames=a_cfg["crop_frames"],
        patch_size=tuple(m_cfg["patch_size"]),
        d_model=m_cfg["d_model"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        ffn_mult=m_cfg["ffn_mult"],
        style_dim=m_cfg["style_dim"],
        mode=mode,
        dropout=m_cfg["dropout"],
    )
    decoder = RetNetSpectrogramDecoder(
        n_mels=a_cfg["n_mels"],
        crop_frames=a_cfg["crop_frames"],
        patch_size=tuple(m_cfg["patch_size"]),
        d_model=m_cfg["d_model"],
        style_dim=m_cfg["style_dim"],
        num_layers=m_cfg["decoder_layers"],
        num_heads=m_cfg["num_heads"],
        ffn_mult=m_cfg["ffn_mult"],
        postnet_channels=m_cfg.get("postnet_channels", 128),
        mode=mode,
        dropout=m_cfg["dropout"],
    )
    return HybridRetNetAEJEPA(
        encoder=encoder,
        decoder=decoder,
        predictor_dim=m_cfg["predictor_dim"],
        predictor_layers=m_cfg["predictor_layers"],
        num_heads=m_cfg["num_heads"],
        mask_ratio=p_cfg["mask_ratio"],
        lambda_recon=p_cfg["lambda_recon"],
        lambda_jepa=l_jepa,
        lambda_var=p_cfg["lambda_var"],
    )
