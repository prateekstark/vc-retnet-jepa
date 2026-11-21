"""Scaled-up Hybrid Split-Latent RetNet Encoder (with 32-D InstanceNorm Information Bottleneck) + Conv-PostNet Decoder + JEPA."""

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_swiglu(w1_x: torch.Tensor, w2_x: torch.Tensor) -> torch.Tensor:
    return F.silu(w1_x) * w2_x


def compute_retention_decay(
    positions: torch.Tensor,
    gammas: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    pos0 = positions[0:1]
    diff = pos0.unsqueeze(2) - pos0.unsqueeze(1)
    if causal:
        mask = (diff >= 0).to(gammas.dtype)
        dist = torch.clamp(diff, min=0).to(gammas.dtype)
    else:
        mask = torch.ones_like(diff, dtype=gammas.dtype)
        dist = diff.abs().to(gammas.dtype)
    log_g = torch.log(gammas).view(1, -1, 1, 1)
    decay = torch.exp(log_g * dist.unsqueeze(1)) * mask.unsqueeze(1)
    return decay / decay.sum(dim=-1, keepdim=True).clamp(min=1.0)


def build_2d_sincos_pos_embed(
    n_freq_patches: int, n_time_patches: int, d_model: int
) -> torch.Tensor:
    assert d_model % 4 == 0
    d_half = d_model // 2

    def _1d(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        omega = 1.0 / (10000.0 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)))
        out = pos * omega.unsqueeze(0)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    f_emb = _1d(n_freq_patches, d_half)
    t_emb = _1d(n_time_patches, d_half)
    grid_t = t_emb.unsqueeze(1).expand(n_time_patches, n_freq_patches, d_half)
    grid_f = f_emb.unsqueeze(0).expand(n_time_patches, n_freq_patches, d_half)
    pos = torch.cat([grid_f, grid_t], dim=-1).reshape(n_time_patches * n_freq_patches, d_model)
    return pos.unsqueeze(0)


class MultiScaleRetention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mode: str = "bidirectional", dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.is_causal = mode == "causal"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        gammas = 1.0 - torch.exp2(-5.0 - torch.arange(num_heads, dtype=torch.float32))
        self.register_buffer("gammas", gammas)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, seq_len, d = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, seq_len)
        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        decay = compute_retention_decay(positions, self.gammas.to(dtype=x.dtype), self.is_causal)
        out = torch.matmul(self.dropout(qk * decay), v)
        out = out.transpose(1, 2).contiguous().view(b * seq_len, d)
        out = self.group_norm(out).view(b, seq_len, d)
        return self.out_proj(F.silu(self.g_proj(x)) * out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(fused_swiglu(self.w1(x), self.w2(x))))


class RetNetBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int = 3, mode: str = "bidirectional", dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.ret = MultiScaleRetention(d_model, num_heads, mode=mode, dropout=dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.ret(self.norm1(x), positions=positions)
        x = x + self.ffn(self.norm2(x))
        return x


class SplitLatentRetNetEncoder(nn.Module):
    """Encoder with 32-D InstanceNorm Information Bottleneck on z_sem + intermediate style z_style."""

    def __init__(
        self,
        n_mels: int = 80,
        crop_frames: int = 128,
        patch_size: Tuple[int, int] = (8, 8),
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_mult: int = 3,
        style_dim: int = 128,
        bottleneck_dim: int = 32,
        mode: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pf, pt = patch_size
        assert n_mels % pf == 0 and crop_frames % pt == 0
        self.patch_size = patch_size
        self.n_freq_patches = n_mels // pf
        self.n_time_patches = crop_frames // pt
        self.num_patches = self.n_freq_patches * self.n_time_patches
        self.d_model = d_model
        self.style_dim = style_dim
        self.mode = mode
        self.split_layer = max(1, num_layers // 2)

        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=(pf, pt), stride=(pf, pt))
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(self.n_freq_patches, self.n_time_patches, d_model)
        )
        self.blocks = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult, mode=mode, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_pre_bn = nn.RMSNorm(d_model)
        # Upgrade 1: Tight 32-D Information Bottleneck + InstanceNorm1d strips global speaker timbre from z_sem
        self.sem_bottleneck_down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.sem_inorm = nn.InstanceNorm1d(bottleneck_dim, affine=False, eps=1e-4)
        self.sem_bottleneck_up = nn.Linear(bottleneck_dim, d_model, bias=False)
        self.norm_sem = nn.RMSNorm(d_model)

        self.style_head = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )
        self.orth_proj = nn.Linear(d_model, style_dim, bias=False)

    def _apply_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_pre_bn(x)
        z_bn = self.sem_bottleneck_down(h).transpose(1, 2)  # (B, 32, L)
        z_bn = self.sem_inorm(z_bn).transpose(1, 2)         # (B, L, 32)
        return self.norm_sem(self.sem_bottleneck_up(z_bn))

    def patchify(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(spec)
        x = x.permute(0, 3, 2, 1).contiguous().view(spec.shape[0], self.num_patches, self.d_model)
        return x + self.pos_embed

    def forward(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patchify(spec)
        z_style = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 == self.split_layer:
                z_style = self.style_head(x.mean(dim=1))
        z_sem = self._apply_bottleneck(x)
        return z_sem, z_style

    def encode_context(self, spec: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.patchify(spec)
        idx_exp = keep_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        x = torch.gather(tokens, dim=1, index=idx_exp)
        for blk in self.blocks:
            x = blk(x, positions=keep_indices)
        return self._apply_bottleneck(x)


class Conv2DPostNet(nn.Module):
    """5-layer Residual Conv2D Post-Net with SiLU to eliminate patch grid artifacts and sharpen harmonics."""

    def __init__(self, channels: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        for i in range(4):
            in_c = 1 if i == 0 else channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, channels, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
        layers.append(nn.Conv2d(channels, 1, kernel_size=5, padding=2))
        self.net = nn.Sequential(*layers)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.net(spec)


class RetNetSpectrogramDecoder(nn.Module):
    """Reconstructs 2D spectrogram y_hat from semantic tokens z_sem + style vector z_style + ConvPostNet."""

    def __init__(
        self,
        n_mels: int = 80,
        crop_frames: int = 128,
        patch_size: Tuple[int, int] = (8, 8),
        d_model: int = 256,
        style_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 3,
        postnet_channels: int = 128,
        mode: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pf, pt = patch_size
        self.pf, self.pt = pf, pt
        self.n_freq_patches = n_mels // pf
        self.n_time_patches = crop_frames // pt
        self.d_model = d_model

        self.default_style = nn.Parameter(torch.zeros(1, style_dim))
        self.style_proj = nn.Linear(style_dim, d_model * 2)
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(self.n_freq_patches, self.n_time_patches, d_model)
        )
        self.blocks = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult, mode=mode, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model)
        self.patch_to_pixel = nn.Linear(d_model, pf * pt)
        self.postnet = Conv2DPostNet(channels=postnet_channels, dropout=dropout)

    def forward_both(
        self, z_sem: torch.Tensor, z_style: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, _ = z_sem.shape
        if z_style is None:
            z_style = self.default_style.expand(b, -1)
        gamma, beta = self.style_proj(z_style).unsqueeze(1).chunk(2, dim=-1)

        x = z_sem * (1.0 + gamma) + beta + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        patches = self.patch_to_pixel(x)
        patches = patches.view(b, self.n_time_patches, self.n_freq_patches, self.pf, self.pt)
        y_pre = patches.permute(0, 2, 3, 1, 4).contiguous().view(
            b, 1, self.n_freq_patches * self.pf, self.n_time_patches * self.pt
        )
        y_post = y_pre + self.postnet(y_pre)
        return y_pre, y_post

    def forward(
        self, z_sem: torch.Tensor, z_style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, y_post = self.forward_both(z_sem, z_style)
        return y_post


class RetNetJEPAPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        predictor_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 8,
        mode: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_proj = nn.Linear(d_model, predictor_dim)
        self.blocks = nn.ModuleList(
            [RetNetBlock(predictor_dim, num_heads, 2, mode=mode, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, d_model)

    def forward(
        self,
        ctx_latents: torch.Tensor,
        ctx_indices: torch.Tensor,
        tgt_indices: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        b, l_ctx, _ = ctx_latents.shape
        l_tgt = tgt_indices.shape[1]
        pos_exp = pos_embed.expand(b, -1, -1)
        ctx_pos = torch.gather(pos_exp, 1, ctx_indices.unsqueeze(-1).expand(-1, -1, pos_exp.shape[-1]))
        tgt_pos = torch.gather(pos_exp, 1, tgt_indices.unsqueeze(-1).expand(-1, -1, pos_exp.shape[-1]))

        ctx_tok = self.in_proj(ctx_latents) + self.pos_proj(ctx_pos)
        tgt_tok = self.mask_token.expand(b, l_tgt, -1) + self.pos_proj(tgt_pos)
        all_tok = torch.cat([ctx_tok, tgt_tok], dim=1)
        all_pos = torch.cat([ctx_indices, tgt_indices], dim=1)

        sort_idx = torch.argsort(all_pos, dim=1)
        all_pos_sorted = torch.gather(all_pos, 1, sort_idx)
        all_tok_sorted = torch.gather(
            all_tok, 1, sort_idx.unsqueeze(-1).expand(-1, -1, all_tok.shape[-1])
        )
        x = all_tok_sorted
        for blk in self.blocks:
            x = blk(x, positions=all_pos_sorted)
        x = self.norm(x)

        inv_idx = torch.argsort(sort_idx, dim=1)
        x_unsorted = torch.gather(x, 1, inv_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return self.out_proj(x_unsorted[:, l_ctx:, :])


def sample_jepa_masks(
    batch_size: int,
    n_freq_patches: int,
    n_time_patches: int,
    mode: str,
    mask_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_patches = n_freq_patches * n_time_patches
    if mode == "causal":
        t_split = max(1, int(round(n_time_patches * (1.0 - mask_ratio))))
        split_idx = t_split * n_freq_patches
        ctx = torch.arange(0, split_idx, device=device)
        tgt = torch.arange(split_idx, num_patches, device=device)
    else:
        mask = torch.zeros(n_time_patches, n_freq_patches, dtype=torch.bool, device=device)
        n_tgt = max(1, int(round(num_patches * mask_ratio)))
        while mask.sum().item() < n_tgt:
            t_len = int(torch.randint(1, max(2, n_time_patches // 2), (1,)).item())
            f_len = int(torch.randint(1, max(2, n_freq_patches), (1,)).item())
            t0 = int(torch.randint(0, max(1, n_time_patches - t_len + 1), (1,)).item())
            f0 = int(torch.randint(0, max(1, n_freq_patches - f_len + 1), (1,)).item())
            mask[t0 : t0 + t_len, f0 : f0 + f_len] = True
        flat = mask.view(-1)
        tgt = torch.nonzero(flat, as_tuple=False).squeeze(-1)
        ctx = torch.nonzero(~flat, as_tuple=False).squeeze(-1)
        if ctx.numel() == 0:
            ctx = torch.tensor([0], device=device)
            tgt = torch.arange(1, num_patches, device=device)
    return ctx.unsqueeze(0).expand(batch_size, -1), tgt.unsqueeze(0).expand(batch_size, -1)


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    z_flat = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z_flat.var(dim=0, unbiased=False) + 1e-4)
    return torch.mean(F.relu(1.0 - std))


class HybridRetNetAEJEPA(nn.Module):
    """Joint Bottleneck Encoder + Conv-PostNet Decoder + JEPA with Orthogonal Style Disentanglement."""

    def __init__(
        self,
        encoder: SplitLatentRetNetEncoder,
        decoder: RetNetSpectrogramDecoder,
        predictor_dim: int = 192,
        predictor_layers: int = 3,
        num_heads: int = 8,
        mask_ratio: float = 0.45,
        lambda_recon: float = 1.0,
        lambda_jepa: float = 0.5,
        lambda_var: float = 0.05,
        lambda_orth: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.decoder = decoder
        self.predictor = RetNetJEPAPredictor(
            d_model=encoder.d_model,
            predictor_dim=predictor_dim,
            num_layers=predictor_layers,
            num_heads=num_heads,
            mode=encoder.mode,
        )
        self.mask_ratio = mask_ratio
        self.lambda_recon = lambda_recon
        self.lambda_jepa = lambda_jepa
        self.lambda_var = lambda_var
        self.lambda_orth = lambda_orth

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996) -> None:
        for p_c, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1.0 - decay)

    def jepa_manifold_verify_loss(self, z_sem: torch.Tensor) -> torch.Tensor:
        b = z_sem.shape[0]
        ctx_idx, tgt_idx = sample_jepa_masks(
            b,
            self.encoder.n_freq_patches,
            self.encoder.n_time_patches,
            self.encoder.mode,
            self.mask_ratio,
            z_sem.device,
        )
        d = z_sem.shape[-1]
        z_ctx = torch.gather(z_sem, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, d))
        z_tgt = torch.gather(z_sem, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, d))
        z_tgt_norm = F.layer_norm(z_tgt, (d,))
        pred = self.predictor(z_ctx, ctx_idx, tgt_idx, self.encoder.pos_embed)
        return F.smooth_l1_loss(pred, z_tgt_norm)

    def forward(self, spec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        b = spec.shape[0]
        z_sem, z_style = self.encoder(spec)
        with torch.no_grad():
            self.decoder.default_style.data.mul_(0.99).add_(z_style.mean(dim=0, keepdim=True), alpha=0.01)

        y_pre, y_post = self.decoder.forward_both(z_sem, z_style)
        recon_loss = (
            0.5 * (F.l1_loss(y_pre, spec) + 0.5 * F.mse_loss(y_pre, spec))
            + 1.0 * (F.l1_loss(y_post, spec) + 0.5 * F.mse_loss(y_post, spec))
        )

        if self.lambda_jepa > 0.0:
            ctx_idx, tgt_idx = sample_jepa_masks(
                b,
                self.encoder.n_freq_patches,
                self.encoder.n_time_patches,
                self.encoder.mode,
                self.mask_ratio,
                spec.device,
            )
            with torch.no_grad():
                full_tgt, _ = self.target_encoder(spec)
                tgt_latents = torch.gather(
                    full_tgt, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_tgt.shape[-1])
                )
                tgt_latents = F.layer_norm(tgt_latents, (tgt_latents.shape[-1],))

            ctx_latents = self.encoder.encode_context(spec, keep_indices=ctx_idx)
            pred_latents = self.predictor(ctx_latents, ctx_idx, tgt_idx, self.encoder.pos_embed)
            jepa_loss = F.smooth_l1_loss(pred_latents, tgt_latents)
        else:
            jepa_loss = torch.tensor(0.0, device=spec.device)

        var_loss = variance_loss(z_sem)
        # Upgrade 1b: Cosine Orthogonality Disentanglement between semantic content and speaker style
        sem_proj = self.encoder.orth_proj(z_sem.mean(dim=1))
        orth_loss = F.cosine_similarity(sem_proj, z_style, dim=-1).abs().mean()

        total = (
            self.lambda_recon * recon_loss
            + self.lambda_jepa * jepa_loss
            + self.lambda_var * var_loss
            + self.lambda_orth * orth_loss
        )
        return total, {
            "recon": recon_loss.item(),
            "jepa": jepa_loss.item(),
            "orth": orth_loss.item(),
            "var": var_loss.item(),
            "total": total.item(),
        }
