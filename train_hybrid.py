"""Train Scaled-up Hybrid Split-Latent RetNet Encoder + Conv-PostNet Decoder + JEPA (with Auto-AMP & compile on T4+)."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vc_dataset import SpeakerMelDataset, SyntheticSpeakerDataset
from src.utils import build_hybrid_model, get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Hybrid AE+JEPA on a speaker domain.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--speaker", type=str, default="a")
    parser.add_argument("--wav-dir", type=str, default=None)
    parser.add_argument("--retnet-mode", type=str, choices=["bidirectional", "causal"], default=None)
    parser.add_argument("--lambda-jepa", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    p_cfg = cfg["pretrain"]
    a_cfg = cfg["audio"]
    epochs = args.epochs or p_cfg["epochs"]
    bs = args.batch_size or p_cfg["batch_size"]

    if args.wav_dir:
        dataset = SpeakerMelDataset(
            wav_dir=args.wav_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"],
            hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=True,
        )
    else:
        dataset = SyntheticSpeakerDataset(
            num_samples=200,
            speaker=args.speaker,
            n_mels=a_cfg["n_mels"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"],
        )

    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    raw_model = build_hybrid_model(cfg, retnet_mode=args.retnet_mode, lambda_jepa_override=args.lambda_jepa).to(device)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]

    # Auto-enable FP16 AMP & torch.compile on Volta/Turing/Ampere+ (e.g. Colab T4, capability >= 7.0)
    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_amp = cc_major >= 7
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(trainable, lr=p_cfg["lr"], weight_decay=p_cfg["weight_decay"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location=device, weights_only=True)
            raw_model.load_state_dict(prev["model"], strict=True)
            start_epoch = prev.get("epoch", epochs) + 1
            print(f"Resuming {out_path} from epoch {start_epoch}/{epochs}")
        except Exception:
            print(f"Architecture changed for {out_path}; starting fresh from epoch 1.")

    model = raw_model
    if cc_major >= 7 and not args.no_compile and hasattr(torch, "compile"):
        print("Enabling torch.compile (Inductor/Triton) + FP16 AMP for T4/modern GPU...")
        model = torch.compile(raw_model)

    print(
        f"[Hybrid AE+JEPA Speaker-{args.speaker.upper()}] mode={raw_model.encoder.mode} | "
        f"lambda_jepa={raw_model.lambda_jepa} | amp={use_amp} | bs={bs} | "
        f"trainable_params={sum(p.numel() for p in trainable):,} | device={device} | samples={len(dataset)}"
    )

    for epoch in range(start_epoch, epochs + 1):
        raw_model.train()
        running = {}
        for specs in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            specs = specs.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, stats = model(specs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            raw_model.update_ema(decay=p_cfg["ema_decay"])
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        n_steps = max(1, len(loader))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            print(f"Epoch {epoch:03d}/{epochs} | {summary}")
            torch.save(
                {"model": raw_model.state_dict(), "retnet_mode": raw_model.encoder.mode, "epoch": epoch},
                out_path,
            )

    print(f"Saved hybrid checkpoint to {out_path}")


if __name__ == "__main__":
    main()
