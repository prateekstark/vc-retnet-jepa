"""Evaluate Voice Conversion metrics and synthesize WAV samples via 16kHz HiFi-GAN."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.audio.mel_vocoder import LogMelFrontend, load_wav, save_wav
from src.data.vc_dataset import (
    PairedSpeakerDataset,
    SpeakerMelDataset,
    SyntheticPairedSpeakerDataset,
    SyntheticSpeakerDataset,
)
from src.models.latent_converter import VoiceLatentConverter
from src.utils import build_hybrid_model, get_device, load_config, set_seed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B -> A Voice Conversion & export HiFi-GAN WAV samples.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--converter-ckpt", type=str, required=True)
    parser.add_argument("--wav-dir-a", type=str, default=None)
    parser.add_argument("--wav-dir-b", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hifigan-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/samples")
    parser.add_argument("--num-wavs", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    a_cfg = cfg["audio"]

    ckpt = torch.load(args.converter_ckpt, map_location="cpu", weights_only=True)
    retnet_mode = ckpt.get("retnet_mode", cfg["model"]["retnet_mode"])
    mode = ckpt.get("mode", "hybrid_ac")

    model_a = build_hybrid_model(cfg, retnet_mode=retnet_mode)
    model_b = build_hybrid_model(cfg, retnet_mode=retnet_mode)
    converter = VoiceLatentConverter(model_a=model_a, model_b=model_b, mode=mode)
    converter.load_state_dict(ckpt["converter"])
    converter.to(device).eval()

    frontend = LogMelFrontend(
        sample_rate=a_cfg["sample_rate"],
        n_fft=a_cfg["n_fft"],
        hop_length=a_cfg["hop_length"],
        n_mels=a_cfg["n_mels"],
        f_min=a_cfg["f_min"],
        f_max=a_cfg["f_max"],
        hifigan_ckpt=args.hifigan_ckpt or a_cfg.get("hifigan_ckpt", "data/hifigan_16k_generator.ckpt"),
    ).to(device)

    if args.wav_dir_a and args.wav_dir_b:
        ds_a = SpeakerMelDataset(
            args.wav_dir_a, split="eval", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=False,
        )
        ds_b = SpeakerMelDataset(
            args.wav_dir_b, split="eval", sample_rate=a_cfg["sample_rate"],
            n_fft=a_cfg["n_fft"], hop_length=a_cfg["hop_length"],
            n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], random_crop=False,
        )
        ds_eval = PairedSpeakerDataset(ds_a, ds_b, random_crop=False)
    else:
        ds_a = SyntheticSpeakerDataset(50, speaker="a", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"] + 1)
        ds_b = SyntheticSpeakerDataset(50, speaker="b", n_mels=a_cfg["n_mels"], crop_frames=a_cfg["crop_frames"], seed=cfg["seed"] + 1)
        ds_eval = SyntheticPairedSpeakerDataset(ds_a, ds_b)

    loader = DataLoader(ds_eval, batch_size=16, shuffle=False)
    raw_mel_l1, conv_mel_l1 = 0.0, 0.0
    raw_cos, conv_cos = 0.0, 0.0
    jepa_err = 0.0
    n_batches = 0

    for spec_a, spec_b in loader:
        spec_a, spec_b = spec_a.to(device).float(), spec_b.to(device).float()
        spec_b2a, z_hat_a = converter.convert_b_to_a(spec_b, ref_spec_a=spec_a)
        z_a_gt, _ = converter.model_a.encoder(spec_a)
        z_a_on_b, _ = converter.model_a.encoder(spec_b)

        raw_mel_l1 += F.l1_loss(spec_b, spec_a).item()
        conv_mel_l1 += F.l1_loss(spec_b2a, spec_a).item()
        raw_cos += F.cosine_similarity(z_a_on_b, z_a_gt, dim=-1).mean().item()
        conv_cos += F.cosine_similarity(z_hat_a, z_a_gt, dim=-1).mean().item()
        jepa_err += converter.model_a.jepa_manifold_verify_loss(z_hat_a).item()
        n_batches += 1

    n_batches = max(1, n_batches)
    print(f"=== Voice Conversion Eval (mode={mode}, retnet_mode={retnet_mode}) ===")
    print(f"Mel L1 Distance to Target A   | Unconverted: {raw_mel_l1/n_batches:.4f} -> Converted: {conv_mel_l1/n_batches:.4f}")
    print(f"Latent Cosine Sim to Target A | Unconverted: {raw_cos/n_batches:.4f} -> Converted: {conv_cos/n_batches:.4f}")
    print(f"Target P_A JEPA Manifold Loss | {jepa_err/n_batches:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.num_wavs, len(ds_eval))):
        sa, sb = ds_eval[i]
        sa, sb = sa.unsqueeze(0).to(device).float(), sb.unsqueeze(0).to(device).float()
        sb2a, _ = converter.convert_b_to_a(sb, ref_spec_a=sa)

        wav_b = frontend.mel_to_wav(sb)
        wav_a_voc = frontend.mel_to_wav(sa)
        wav_b2a = frontend.mel_to_wav(sb2a)

        save_wav(out_dir / f"sample_{i}_source_b_hifigan.wav", wav_b, a_cfg["sample_rate"])
        save_wav(out_dir / f"sample_{i}_target_a_hifigan_gt.wav", wav_a_voc, a_cfg["sample_rate"])
        save_wav(out_dir / f"sample_{i}_converted_b_to_a.wav", wav_b2a, a_cfg["sample_rate"])

        # Also copy the raw original ground-truth WAV recording if evaluating on real data
        if isinstance(ds_eval, PairedSpeakerDataset):
            ia, ib, _ = ds_eval.pairs[i]
            raw_a = load_wav(ds_a.wavs[ia], target_sr=a_cfg["sample_rate"])
            raw_b = load_wav(ds_b.wavs[ib], target_sr=a_cfg["sample_rate"])
            save_wav(out_dir / f"sample_{i}_target_a_raw.wav", raw_a, a_cfg["sample_rate"])
            save_wav(out_dir / f"sample_{i}_source_b_raw.wav", raw_b, a_cfg["sample_rate"])

    print(f"Saved {min(args.num_wavs, len(ds_eval))} HiFi-GAN vocoded WAV sets to {out_dir}")


if __name__ == "__main__":
    main()
