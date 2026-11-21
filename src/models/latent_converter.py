"""Scaled-up B -> A Latent Space Converter with Conv-PostNet Cross-Decode & P_A Verification."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.retnet_ae_jepa import HybridRetNetAEJEPA, RetNetBlock


class RetNetLatentTranslator(nn.Module):
    """3-Layer RetNet translator mapping semantic tokens z_B (B, N, D) -> z_A (B, N, D)."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        mode: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.skip = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult=3, mode=mode, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for blk in self.blocks:
            h = blk(h)
        return self.norm(self.skip(z) + h)


def latent_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = F.smooth_l1_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return l1 + 0.5 * cos


def content_self_similarity_loss(z_src: torch.Tensor, z_translated: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z_src, dim=-1)
    z2 = F.normalize(z_translated, dim=-1)
    sim1 = torch.matmul(z1, z1.transpose(-2, -1))
    sim2 = torch.matmul(z2, z2.transpose(-2, -1))
    return F.mse_loss(sim2, sim1)


class VoiceLatentConverter(nn.Module):
    """Converts Domain/Speaker B latents onto Domain/Speaker A and decodes via D_A."""

    def __init__(
        self,
        model_a: HybridRetNetAEJEPA,
        model_b: HybridRetNetAEJEPA,
        mode: str = "hybrid_ac",
        lambda_latent: float = 1.0,
        lambda_cross_decode: float = 1.5,
        lambda_jepa_verify: float = 0.4,
        lambda_cycle: float = 1.0,
        lambda_content_sim: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in ("unpaired", "paired", "hybrid_ac"):
            raise ValueError(f"Unsupported converter mode: {mode}")
        self.mode = mode
        self.lambda_latent = lambda_latent
        self.lambda_cross_decode = lambda_cross_decode
        self.lambda_jepa_verify = lambda_jepa_verify
        self.lambda_cycle = lambda_cycle
        self.lambda_content_sim = lambda_content_sim

        self.model_a = model_a
        self.model_b = model_b
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
        self.model_a.eval()
        self.model_b.eval()

        d_model = model_a.encoder.d_model
        num_heads = model_a.encoder.blocks[0].ret.num_heads
        ret_mode = model_a.encoder.mode

        self.t_b2a = RetNetLatentTranslator(d_model, num_heads, num_layers=3, mode=ret_mode)
        self.t_a2b = RetNetLatentTranslator(d_model, num_heads, num_layers=3, mode=ret_mode)

    @torch.no_grad()
    def convert_b_to_a(
        self, spec_b: torch.Tensor, ref_spec_a: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_b, _ = self.model_b.encoder(spec_b)
        z_hat_a = self.t_b2a(z_b)
        style_a = None
        if ref_spec_a is not None:
            _, style_a = self.model_a.encoder(ref_spec_a)
            if style_a.shape[0] == 1 and spec_b.shape[0] > 1:
                style_a = style_a.expand(spec_b.shape[0], -1)
        spec_b2a = self.model_a.decoder(z_hat_a, style_a)
        return spec_b2a, z_hat_a

    def forward_step(
        self,
        unpaired_a: torch.Tensor,
        unpaired_b: torch.Tensor,
        pair_a: Optional[torch.Tensor] = None,
        pair_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=unpaired_b.device)
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            z_b, style_b = self.model_b.encoder(unpaired_b)
            z_a, style_a = self.model_a.encoder(unpaired_a)

        z_hat_a = self.t_b2a(z_b)

        # 1. Frozen P_A Manifold Verification + Content Self-Similarity
        if self.lambda_jepa_verify > 0.0:
            jepa_ver_a = self.model_a.jepa_manifold_verify_loss(z_hat_a)
            total = total + self.lambda_jepa_verify * jepa_ver_a
            metrics["jepa_verify"] = jepa_ver_a.item()

        content_loss = content_self_similarity_loss(z_b, z_hat_a)
        total = total + self.lambda_content_sim * content_loss
        metrics["content_sim"] = content_loss.item()

        # 2. Unpaired Cycle Consistency: B -> A -> B and A -> B -> A
        if self.mode in ("unpaired", "hybrid_ac"):
            x_b2a = self.model_a.decoder(z_hat_a, style_a)
            z_a_cyc, _ = self.model_a.encoder(x_b2a)
            z_b_cyc = self.t_a2b(z_a_cyc)
            x_b_rec = self.model_b.decoder(z_b_cyc, style_b)

            z_hat_b = self.t_a2b(z_a)
            x_a2b = self.model_b.decoder(z_hat_b, style_b)
            z_b_cyc2, _ = self.model_b.encoder(x_a2b)
            z_a_cyc2 = self.t_b2a(z_b_cyc2)
            x_a_rec = self.model_a.decoder(z_a_cyc2, style_a)

            cycle_loss = (
                F.l1_loss(x_b_rec, unpaired_b)
                + F.l1_loss(x_a_rec, unpaired_a)
                + 0.5 * latent_match_loss(z_b_cyc, z_b)
                + 0.5 * latent_match_loss(z_a_cyc2, z_a)
            )
            total = total + self.lambda_cycle * cycle_loss
            metrics["cycle"] = cycle_loss.item()

        # 3. Paired Supervision: T_{B->A}(E_B(x_B)) -> E_A(x_A) and D_A(z_hat_A) -> x_A
        if self.mode in ("paired", "hybrid_ac"):
            if pair_a is None or pair_b is None:
                raise ValueError(f"Mode '{self.mode}' requires (pair_a, pair_b).")
            with torch.no_grad():
                za_p, style_a_p = self.model_a.encoder(pair_a)
                zb_p, _ = self.model_b.encoder(pair_b)
            z_hat_a_p = self.t_b2a(zb_p)
            lat_loss = latent_match_loss(z_hat_a_p, za_p)
            y_pre, y_post = self.model_a.decoder.forward_both(z_hat_a_p, style_a_p)
            cross_dec_loss = (
                0.5 * (F.l1_loss(y_pre, pair_a) + 0.5 * F.mse_loss(y_pre, pair_a))
                + 1.0 * (F.l1_loss(y_post, pair_a) + 0.5 * F.mse_loss(y_post, pair_a))
            )

            total = total + self.lambda_latent * lat_loss + self.lambda_cross_decode * cross_dec_loss
            metrics["latent_pair"] = lat_loss.item()
            metrics["cross_decode"] = cross_dec_loss.item()

        metrics["total"] = total.item()
        return total, metrics
