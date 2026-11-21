"""Multi-speaker dataset loader with MFCC Dynamic Time Warping (DTW) alignment for parallel pairs."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct
from torch.utils.data import Dataset

from src.audio.mel_vocoder import LogMelFrontend, crop_or_pad, load_wav


def dtw_align_mel_a_to_b(mel_a: torch.Tensor, mel_b: torch.Tensor) -> torch.Tensor:
    """Warp mel_a (1, n_mels, T_a) onto mel_b's timeline (1, n_mels, T_b) using CMVN-MFCC DTW.

    By using cepstral mean-variance normalized MFCCs (c1..c13, excluding c0 energy and high pitch harmonics),
    the DTW cost is invariant to speaker pitch/timbre and aligns pure phonetic timing.
    """
    a_np = mel_a.squeeze(0).cpu().numpy()  # (n_mels, T_a)
    b_np = mel_b.squeeze(0).cpu().numpy()  # (n_mels, T_b)

    # Extract MFCCs c1..c13 via Type-II DCT along frequency axis
    mfcc_a = dct(a_np, type=2, axis=0, norm="ortho")[1:14, :]  # (13, T_a)
    mfcc_b = dct(b_np, type=2, axis=0, norm="ortho")[1:14, :]  # (13, T_b)

    # Cepstral Mean and Variance Normalization (CMVN) per utterance removes speaker timbre bias
    mfcc_a = (mfcc_a - mfcc_a.mean(axis=1, keepdims=True)) / (mfcc_a.std(axis=1, keepdims=True) + 1e-5)
    mfcc_b = (mfcc_b - mfcc_b.mean(axis=1, keepdims=True)) / (mfcc_b.std(axis=1, keepdims=True) + 1e-5)

    # L2 normalize frames for cosine distance cost matrix (T_b, T_a)
    mfcc_a /= np.linalg.norm(mfcc_a, axis=0, keepdims=True) + 1e-6
    mfcc_b /= np.linalg.norm(mfcc_b, axis=0, keepdims=True) + 1e-6
    cost = 1.0 - np.dot(mfcc_b.T, mfcc_a)  # (T_b, T_a)

    tb, ta = cost.shape
    D = np.full((tb + 1, ta + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    # Sakoe-Chiba band constraint (35% of sequence length) for speed and monotonic speech alignment
    band = max(25, int(max(tb, ta) * 0.35))

    for i in range(1, tb + 1):
        j_center = int(round(i * ta / tb))
        j_start = max(1, j_center - band)
        j_end = min(ta + 1, j_center + band)
        c_row = cost[i - 1]
        d_prev = D[i - 1]
        d_curr = D[i]
        for j in range(j_start, j_end):
            d_curr[j] = c_row[j - 1] + min(d_prev[j - 1], d_prev[j], d_curr[j - 1])

    # Backtrack optimal warping path; for each frame i in B, collect aligned frame(s) j in A
    i, j = tb, ta
    # If band missed (tb, ta), start from minimum cost in last row
    if not np.isfinite(D[tb, ta]):
        j = int(np.argmin(D[tb, 1:])) + 1

    b_to_a_sum = np.zeros((a_np.shape[0], tb), dtype=np.float32)
    b_to_a_cnt = np.zeros((1, tb), dtype=np.float32)

    while i > 0 and j > 0:
        b_to_a_sum[:, i - 1] += a_np[:, j - 1]
        b_to_a_cnt[0, i - 1] += 1.0
        if i == 1 and j == 1:
            break
        diag = D[i - 1, j - 1]
        up = D[i - 1, j]
        left = D[i, j - 1]
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1

    b_to_a_cnt = np.maximum(b_to_a_cnt, 1.0)
    warped_a = b_to_a_sum / b_to_a_cnt
    return torch.from_numpy(warped_a).unsqueeze(0).to(dtype=mel_a.dtype)


class SpeakerMelDataset(Dataset):
    """Loads log-mel spectrograms for a single speaker (train or eval split)."""

    def __init__(
        self,
        wav_dir: str | Path,
        split: str = "train",
        eval_count: int = 100,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        crop_frames: int = 128,
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
    """Pairs parallel utterances (x_A, x_B) with DTW time-alignment and synchronized crops."""

    def __init__(
        self,
        ds_a: SpeakerMelDataset,
        ds_b: SpeakerMelDataset,
        paired_fraction: float = 1.0,
        random_crop: bool = True,
        seed: int = 42,
    ) -> None:
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.random_crop = random_crop

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
        self._dtw_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_aligned_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx not in self._dtw_cache:
            ia, ib, _ = self.pairs[idx]
            mel_a, _, _, _ = self.ds_a.get_full_mel(ia)
            mel_b, _, _, _ = self.ds_b.get_full_mel(ib)
            mel_a_aligned = dtw_align_mel_a_to_b(mel_a, mel_b)
            self._dtw_cache[idx] = (mel_a_aligned, mel_b)
        return self._dtw_cache[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_a_aligned, mel_b = self._get_aligned_pair(idx)
        target_frames = self.ds_b.crop_frames
        _, n_mels, t = mel_b.shape
        if t < target_frames:
            pad_a = torch.full((1, n_mels, target_frames - t), -1.5, dtype=mel_a_aligned.dtype)
            pad_b = torch.full((1, n_mels, target_frames - t), -1.5, dtype=mel_b.dtype)
            return torch.cat([mel_a_aligned, pad_a], dim=-1), torch.cat([mel_b, pad_b], dim=-1)
        if t == target_frames:
            return mel_a_aligned, mel_b
        # Synchronized crop window [start : start + target_frames] on both DTW-aligned spectrograms
        if self.random_crop:
            start = int(torch.randint(0, t - target_frames + 1, (1,)).item())
        else:
            start = (t - target_frames) // 2
        return (
            mel_a_aligned[:, :, start : start + target_frames],
            mel_b[:, :, start : start + target_frames],
        )


class SyntheticSpeakerDataset(Dataset):
    """Synthetic speaker dataset for fast unit testing."""

    def __init__(
        self,
        num_samples: int = 200,
        speaker: str = "a",
        n_mels: int = 80,
        crop_frames: int = 128,
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
