"""Multi-speaker dataset loader for Voice Conversion (CMU Arctic / VCTK / custom WAVs)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.audio.mel_vocoder import LogMelFrontend, crop_or_pad, load_wav


class SpeakerMelDataset(Dataset):
    """Loads log-mel spectrograms for a single speaker (train or eval split)."""

    def __init__(
        self,
        wav_dir: str | Path,
        split: str = "train",
        eval_count: int = 100,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        crop_frames: int = 192,
        random_crop: bool = True,
    ) -> None:
        self.wav_dir = Path(wav_dir)
        self.sample_rate = sample_rate
        self.crop_frames = crop_frames
        self.random_crop = random_crop
        self.frontend = LogMelFrontend(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, hifigan_ckpt=None
        )

        all_wavs = sorted(self.wav_dir.rglob("*.wav"))
        if not all_wavs:
            raise FileNotFoundError(f"No WAV files found in {self.wav_dir}")

        if len(all_wavs) > eval_count * 2:
            if split == "train":
                self.wavs = all_wavs[:-eval_count]
            else:
                self.wavs = all_wavs[-eval_count:]
        else:
            self.wavs = all_wavs

        self.utt_to_idx: Dict[str, int] = {p.stem: i for i, p in enumerate(self.wavs)}
        self._cache: Dict[int, Tuple[torch.Tensor, float, float]] = {}

    def __len__(self) -> int:
        return len(self.wavs)

    def get_full_mel(self, idx: int) -> Tuple[torch.Tensor, float, float, str]:
        if idx not in self._cache:
            wav = load_wav(self.wavs[idx], target_sr=self.sample_rate)
            mel, mean, std = self.frontend.extract(wav)
            self._cache[idx] = (mel, mean, std)
        mel, mean, std = self._cache[idx]
        return mel, mean, std, self.wavs[idx].stem

    def __getitem__(self, idx: int) -> torch.Tensor:
        mel, _, _, _ = self.get_full_mel(idx)
        return crop_or_pad(mel, self.crop_frames, random_crop=self.random_crop)


class PairedSpeakerDataset(Dataset):
    """Pairs parallel utterances (x_A, x_B) by utterance stem (e.g. arctic_a0001)."""

    def __init__(
        self,
        ds_a: SpeakerMelDataset,
        ds_b: SpeakerMelDataset,
        paired_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.ds_a = ds_a
        self.ds_b = ds_b

        common = sorted(set(ds_a.utt_to_idx.keys()) & set(ds_b.utt_to_idx.keys()))
        if not common:
            raise ValueError("No matching utterance IDs found between Speaker A and Speaker B.")

        pairs = [(ds_a.utt_to_idx[u], ds_b.utt_to_idx[u], u) for u in common]
        if paired_fraction < 1.0:
            g = torch.Generator().manual_seed(seed)
            n_keep = max(1, int(round(len(pairs) * paired_fraction)))
            perm = torch.randperm(len(pairs), generator=g)[:n_keep].tolist()
            pairs = [pairs[i] for i in perm]

        self.pairs: List[Tuple[int, int, str]] = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ia, ib, _ = self.pairs[idx]
        # Use synchronized center crops for paired training so frames align
        mel_a, _, _, _ = self.ds_a.get_full_mel(ia)
        mel_b, _, _, _ = self.ds_b.get_full_mel(ib)
        spec_a = crop_or_pad(mel_a, self.ds_a.crop_frames, random_crop=False)
        spec_b = crop_or_pad(mel_b, self.ds_b.crop_frames, random_crop=False)
        return spec_a, spec_b


class SyntheticSpeakerDataset(Dataset):
    """Synthetic speaker dataset for fast unit testing."""

    def __init__(
        self,
        num_samples: int = 200,
        speaker: str = "a",
        n_mels: int = 80,
        crop_frames: int = 192,
        seed: int = 42,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.content = torch.randn(num_samples, 1, n_mels, crop_frames, generator=g).cumsum(dim=-1)
        self.content = self.content / self.content.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        self.speaker = speaker
        self.n_mels = n_mels
        self.crop_frames = crop_frames

    def __len__(self) -> int:
        return self.content.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        c = self.content[idx]
        if self.speaker == "a":
            formant = torch.sin(torch.linspace(0, 4 * 3.14159, self.n_mels)).view(1, self.n_mels, 1)
            x = c + 0.6 * formant
        else:
            formant = torch.cos(torch.linspace(0, 6 * 3.14159, self.n_mels)).view(1, self.n_mels, 1)
            x = 0.8 * c.roll(shifts=3, dims=1) + 0.6 * formant
        return (x - x.mean()) / x.std().clamp(min=1e-5)


class SyntheticPairedSpeakerDataset(Dataset):
    def __init__(self, ds_a: SyntheticSpeakerDataset, ds_b: SyntheticSpeakerDataset) -> None:
        self.ds_a = ds_a
        self.ds_b = ds_b

    def __len__(self) -> int:
        return min(len(self.ds_a), len(self.ds_b))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ds_a[idx], self.ds_b[idx]
