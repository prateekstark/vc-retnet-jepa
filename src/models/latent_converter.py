"""Scaled-up B -> A Latent Space Converter with DTW Paired Cross-Decode, Deep CORAL, & P_A Verification."""

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


def coral_and_band_moment_loss(
    z_pred: torch.Tensor,
    z_real: torch.Tensor,
    n_freq_patches: int = 10,
) -> torch.Tensor:
    """Match per-frequency-band (mean, std) + Deep CORAL covariance between z_pred and z_real."""
    b, n, d = z_pred.shape
    t_p = n // n_freq_patches

    # 1. Per-frequency-patch mean & std alignment across (B, T_p)
    zp_grid = z_pred.view(b, t_p, n_freq_patches, d)
    zr_grid = z_real.view(b, t_p, n_freq_patches, d)

    mu_p = zp_grid.mean(dim=(0, 1))
    mu_r = zr_grid.mean(dim=(0, 1))
    std_p = torch.sqrt(zp_grid.var(dim=(0, 1), unbiased=False) + 1e-4)
    std_r = torch.sqrt(zr_grid.var(dim=(0, 1), unbiased=False) + 1e-4)
    moment_loss = F.mse_loss(mu_p, mu_r) + F.mse_loss(std_p, std_r)

    # 2. Deep CORAL second-order feature covariance alignment
    zp_flat = z_pred.reshape(-1, d)
    zr_flat = z_real.reshape(-1, d)
    zp_c = zp_flat - zp_flat.mean(dim=0, keepdim=True)
    zr_c = zr_flat - zr_flat.mean(dim=0, keepdim=True)
    cov_p = (zp_c.T @ zp_c) / max(1, zp_flat.shape[0] - 1)
    cov_r = (zr_c.T @ zr_c) / max(1, zr_flat.shape[0] - 1)
    coral_loss = F.mse_loss(cov_p, cov_r)

    return moment_loss + 2.0 * coral_loss


def spectral_contrast_loss(spec_pred: torch.Tensor, spec_real: torch.Tensor) -> torch.Tensor:
    """Prevent harmonic over-smoothing by matching per-mel-bin mean/std and frequency/time sharpness."""
    # spec: (B, 1, F, T)
    mu_p = spec_pred.mean(dim=(0, 3))
    mu_r = spec_real.mean(dim=(0, 3))
    std_p = torch.sqrt(spec_pred.var(dim=(0, 3), unbiased=False) + 1e-4)
    std_r = torch.sqrt(spec_real.var(dim=(0, 3), unbiased=False) + 1e-4)

    # Frequency-axis local contrast (harmonic peak-to-valley sharpness)
    df_p = spec_pred[:, :, 1:, :] - spec_pred[:, :, :-1, :]
    df_r = spec_real[:, :, 1:, :] - spec_real[:, :, :-1, :]
    df_std_p = torch.sqrt(df_p.var(dim=(0, 3), unbiased=False) + 1e-4)
    df_std_r = torch.sqrt(df_r.var(dim=(0, 3), unbiased=False) + 1e-4)

    return F.l1_loss(mu_p, mu_r) + F.l1_loss(std_p, std_r) + F.l1_loss(df_std_p, df_std_r)


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
        lambda_coral: float = 0.5,
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
        self.lambda_coral = lambda_coral

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

        # 1. Frozen P_A Manifold Verification + Deep CORAL Distribution Matching + Content Self-Similarity
        if self.lambda_jepa_verify > 0.0:
            jepa_ver_a = self.model_a.jepa_manifold_verify_loss(z_hat_a)
            total = total + self.lambda_jepa_verify * jepa_ver_a
            metrics["jepa_verify"] = jepa_ver_a.item()

        coral_loss = coral_and_band_moment_loss(
            z_hat_a, z_a, n_freq_patches=self.model_a.encoder.n_freq_patches
        )
        total = total + self.lambda_coral * coral_loss
        metrics["coral"] = coral_loss.item()

        content_loss = content_self_similarity_loss(z_b, z_hat_a)
        total = total + self.lambda_content_sim * content_loss
        metrics["content_sim"] = content_loss.item()

        # 2. Unpaired Cycle Consistency: B -> A -> B and A -> B -> A + Spectral Contrast Matching
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

            contrast_loss = spectral_contrast_loss(x_b2a, unpaired_a)
            cycle_loss = (
                F.l1_loss(x_b_rec, unpaired_b)
                + F.l1_loss(x_a_rec, unpaired_a)
                + 0.5 * latent_match_loss(z_b_cyc, z_b)
                + 0.5 * latent_match_loss(z_a_cyc2, z_a)
                + 0.5 * contrast_loss
            )
            total = total + self.lambda_cycle * cycle_loss
            metrics["cycle"] = cycle_loss.item()

        # 3. DTW-Aligned Paired Supervision: T_{B->A}(E_B(x_B)) -> E_A(x_A_dtw) and D_A(z_hat_A) -> x_A_dtw
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
                + 0.4 * spectral_contrast_loss(y_post, pair_a)
            )

            total = total + self.lambda_latent * lat_loss + self.lambda_cross_decode * cross_dec_loss
            metrics["latent_pair"] = lat_loss.item()
            metrics["cross_decode"] = cross_dec_loss.item()

        metrics["total"] = total.item()
        return total, metrics
