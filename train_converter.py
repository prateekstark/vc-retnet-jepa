"""Train Scaled-up B -> A Latent Space Converter with Pre-Cached DTW & Frozen Encoder Latents."""

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.vc_dataset import (
    PairedSpeakerDataset,
    SpeakerMelDataset,
    SyntheticPairedSpeakerDataset,
    SyntheticSpeakerDataset,
)
from src.models.latent_converter import VoiceLatentConverter
from src.utils import build_hybrid_model, get_device, load_config, set_seed


@torch.no_grad()
def get_or_build_latent_cache(
    converter: VoiceLatentConverter,
    ds_pair,
    cache_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if cache_path.exists():
        print(f"Loading precomputed DTW + frozen encoder latents from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"Precomputing DTW alignment & frozen (E_A, E_B) latents for {len(ds_pair)} pairs -> {cache_path} ...")
    loader = DataLoader(ds_pair, batch_size=32, shuffle=False)
    spec_a_list, spec_b_list = [], []
    za_list, style_a_list, zb_list = [], [], []

    for sa, sb in tqdm(loader, desc="DTW + Frozen Encoder Cache", leave=False):
        sa_d = sa.to(device).float()
        sb_d = sb.to(device).float()
        za, style_a = converter.model_a.encoder(sa_d)
        zb, _ = converter.model_b.encoder(sb_d)
        spec_a_list.append(sa_d.cpu())
        spec_b_list.append(sb_d.cpu())
        za_list.append(za.cpu())
        style_a_list.append(style_a.cpu())
        zb_list.append(zb.cpu())

    data = {
        "spec_a": torch.cat(spec_a_list, dim=0),
        "spec_b": torch.cat(spec_b_list, dim=0),
        "za": torch.cat(za_list, dim=0),
        "style_a": torch.cat(style_a_list, dim=0),
        "zb": torch.cat(zb_list, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B -> A Voice Latent Converter.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt-a", type=str, required=True)
    parser.add_argument("--ckpt-b", type=str, required=True)
    parser.add_argument("--wav-dir-a", type=str, default=None)
    parser.add_argument("--wav-dir-b", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["unpaired", "paired", "hybrid_ac"], default=None)
    parser.add_argument("--paired-fraction", type=float, default=None)
    parser.add_argument("--lambda-jepa-verify", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    c_cfg = cfg["convert"]
    a_cfg = cfg["audio"]

    mode = args.mode or c_cfg["mode"]
    epochs = args.epochs or c_cfg["epochs"]
    l_jepa_ver = args.lambda_jepa_verify if args.lambda_jepa_verify is not None else c_cfg["lambda_jepa_verify"]

    ckpt_a = torch.load(args.ckpt_a, map_location="cpu", weights_only=True)
    ckpt_b = torch.load(args.ckpt_b, map_location="cpu", weights_only=True)
    retnet_mode = ckpt_a.get("retnet_mode", cfg["model"]["retnet_mode"])

    model_a = build_hybrid_model(cfg, retnet_mode=retnet_mode)
    model_a.load_state_dict(ckpt_a["model"])

    model_b = build_hybrid_model(cfg, retnet_mode=retnet_mode)
    model_b.load_state_dict(ckpt_b["model"])

    converter = VoiceLatentConverter(
        model_a=model_a,
        model_b=model_b,
        mode=mode,
        lambda_latent=c_cfg["lambda_latent"],
        lambda_cross_decode=c_cfg["lambda_cross_decode"],
        lambda_jepa_verify=l_jepa_ver,
        lambda_cycle=c_cfg["lambda_cycle"],
        lambda_content_sim=c_cfg["lambda_content_sim"],
    ).to(device)

    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_amp = cc_major >= 7
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bs = args.batch_size or (24 if cc_major >= 7 else 8)

    if args.wav_dir_a and args.wav_dir_b:
        ds_a = SpeakerMelDataset(
            args.wav_dir_a, split="train", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=False,
        )
        ds_b = SpeakerMelDataset(
            args.wav_dir_b, split="train", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=False,
        )
        ds_pair = PairedSpeakerDataset(ds_a, ds_b, random_crop=False, seed=cfg["seed"])
        cache_key = f"{Path(args.ckpt_a).stem}__{Path(args.ckpt_b).stem}"
    else:
        ds_a = SyntheticSpeakerDataset(200, speaker="a", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"])
        ds_b = SyntheticSpeakerDataset(200, speaker="b", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"])
        ds_pair = SyntheticPairedSpeakerDataset(ds_a, ds_b)
        cache_key = "synthetic"

    cache_path = Path(args.out).parent / f".latent_cache_{cache_key}.pt"
    cached = get_or_build_latent_cache(converter, ds_pair, cache_path, device)

    tensor_ds = TensorDataset(cached["spec_a"], cached["za"], cached["style_a"], cached["zb"])
    loader = DataLoader(tensor_ds, batch_size=bs, shuffle=True, drop_last=True)

    trainable = [p for p in converter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=c_cfg["lr"], weight_decay=0.02)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[Converter mode={mode}] retnet_mode={retnet_mode} | lambda_jepa_verify={l_jepa_ver} | "
        f"bs={bs} | trainable_params={sum(p.numel() for p in trainable):,} | device={device}"
    )

    for epoch in range(1, epochs + 1):
        converter.train()
        converter.model_a.eval()
        converter.model_b.eval()
        running = {}

        for sa_p, za_p, style_a_p, zb_p in tqdm(loader, desc=f"Convert {epoch}/{epochs}", leave=False):
            sa_p = sa_p.to(device).float()
            za_p = za_p.to(device).float()
            style_a_p = style_a_p.to(device).float()
            zb_p = zb_p.to(device).float()

            # For unpaired terms, roll batch index by 1 so (A, B) are unpaired utterances
            sa_u = sa_p.roll(shifts=1, dims=0)
            za_u = za_p.roll(shifts=1, dims=0)
            style_a_u = style_a_p.roll(shifts=1, dims=0)
            zb_u = zb_p

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, stats = converter.forward_cached_step(
                    unpaired_a=sa_u,
                    za_u=za_u,
                    style_a_u=style_a_u,
                    zb_u=zb_u,
                    pair_a=sa_p if mode in ("paired", "hybrid_ac") else None,
                    za_p=za_p if mode in ("paired", "hybrid_ac") else None,
                    style_a_p=style_a_p if mode in ("paired", "hybrid_ac") else None,
                    zb_p=zb_p if mode in ("paired", "hybrid_ac") else None,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        n_steps = max(1, len(loader))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            print(f"Epoch {epoch:03d}/{epochs} | {summary}")
            torch.save(
                {"converter": converter.state_dict(), "mode": mode, "retnet_mode": retnet_mode, "epoch": epoch},
                out_path,
            )

    print(f"Saved converter checkpoint to {out_path}")


if __name__ == "__main__":
    main()
