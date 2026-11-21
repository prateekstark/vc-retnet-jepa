"""Train Scaled-up B -> A Latent Space Converter (with Auto-AMP & compile on T4+)."""

import argparse
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vc_dataset import (
    PairedSpeakerDataset,
    SpeakerMelDataset,
    SyntheticPairedSpeakerDataset,
    SyntheticSpeakerDataset,
)
from src.models.latent_converter import VoiceLatentConverter
from src.utils import build_hybrid_model, get_device, load_config, set_seed


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
    paired_frac = args.paired_fraction if args.paired_fraction is not None else c_cfg["paired_fraction"]
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
    default_bs = 24 if cc_major >= 7 else c_cfg["batch_size"]
    bs = args.batch_size or default_bs

    if args.wav_dir_a and args.wav_dir_b:
        ds_a = SpeakerMelDataset(
            args.wav_dir_a, split="train", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=True,
        )
        ds_b = SpeakerMelDataset(
            args.wav_dir_b, split="train", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=True,
        )
        ds_pair = (
            PairedSpeakerDataset(ds_a, ds_b, paired_fraction=paired_frac, seed=cfg["seed"])
            if mode in ("paired", "hybrid_ac")
            else None
        )
    else:
        ds_a = SyntheticSpeakerDataset(200, speaker="a", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"])
        ds_b = SyntheticSpeakerDataset(200, speaker="b", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"])
        ds_pair = SyntheticPairedSpeakerDataset(ds_a, ds_b) if mode in ("paired", "hybrid_ac") else None

    loader_a = DataLoader(ds_a, batch_size=bs, shuffle=True, drop_last=True)
    loader_b = DataLoader(ds_b, batch_size=bs, shuffle=True, drop_last=True)
    loader_pair = (
        DataLoader(ds_pair, batch_size=min(bs, len(ds_pair)), shuffle=True, drop_last=False)
        if ds_pair is not None
        else None
    )

    trainable = [p for p in converter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=c_cfg["lr"], weight_decay=0.02)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location=device, weights_only=True)
            converter.load_state_dict(prev["converter"], strict=True)
            start_epoch = prev.get("epoch", epochs) + 1
            print(f"Resuming {out_path} from epoch {start_epoch}/{epochs}")
        except Exception:
            print(f"Architecture changed for {out_path}; starting fresh from epoch 1.")

    step_fn = converter.forward_step
    if cc_major >= 7 and not args.no_compile and hasattr(torch, "compile"):
        print("Enabling torch.compile (Inductor/Triton) + FP16 AMP for T4/modern GPU...")
        step_fn = torch.compile(converter.forward_step)

    print(
        f"[Converter mode={mode}] retnet_mode={retnet_mode} | lambda_jepa_verify={l_jepa_ver} | "
        f"amp={use_amp} | bs={bs} | trainable_params={sum(p.numel() for p in trainable):,} | device={device}"
    )

    for epoch in range(start_epoch, epochs + 1):
        converter.train()
        converter.model_a.eval()
        converter.model_b.eval()
        iter_a = cycle(loader_a)
        iter_pair = cycle(loader_pair) if loader_pair is not None else None
        running = {}

        for spec_b in tqdm(loader_b, desc=f"Convert {epoch}/{epochs}", leave=False):
            spec_b = spec_b.to(device).float()
            spec_a = next(iter_a).to(device).float()
            pa, pb = None, None
            if iter_pair is not None:
                pa_raw, pb_raw = next(iter_pair)
                pa, pb = pa_raw.to(device).float(), pb_raw.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, stats = step_fn(spec_a, spec_b, pair_a=pa, pair_b=pb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        n_steps = max(1, len(loader_b))
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
