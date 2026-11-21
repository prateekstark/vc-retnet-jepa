"""80-bin Slaney Log-Mel Frontend + Pretrained 16kHz HiFi-GAN Neural Vocoder."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from torch.nn.utils import remove_weight_norm, weight_norm


# Global normalization constants so all speakers & converted mels share HiFi-GAN's exact log-mel scale
MEL_MEAN = -5.5
MEL_STD = 2.2


def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    mel = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = freq >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mel


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = mels >= min_log_mel
    freqs[log_mask] = min_log_hz * np.exp(logstep * (mels[log_mask] - min_log_mel))
    return freqs


def build_slaney_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Build (n_mels, n_fft // 2 + 1) Slaney-scale, Slaney area-normalized mel filterbank."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_min, mel_max = _hz_to_mel_slaney(np.array([f_min, f_max]))
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz_slaney(mel_pts)

    fdiff = np.diff(hz_pts)
    ramps = np.subtract.outer(hz_pts, fft_freqs)
    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    weights *= enorm[:, np.newaxis]
    return torch.from_numpy(weights)


def load_wav(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio)
    if sr != target_sr:
        new_len = int(round(wav.shape[0] * target_sr / sr))
        wav = F.interpolate(
            wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
        ).view(-1)
    # Peak-normalize to 1.0 (matches SpeechBrain min_max_energy_norm=True)
    peak = wav.abs().max().clamp(min=1e-6)
    return wav / peak


def save_wav(path: str | Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().cpu().view(-1).numpy()
    peak = max(float(np.abs(audio).max()), 1e-5)
    audio = np.clip(audio / peak * 0.95, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)


class _WNConv1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(in_c, out_c, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _WNConvTranspose1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = weight_norm(
            nn.ConvTranspose1d(in_c, out_c, kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                _WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                _WNConv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HifiGANGenerator(nn.Module):
    """Exact architecture matching speechbrain/tts-hifigan-libritts-16kHz generator.ckpt."""

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        upsample_factors: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        inference_padding: int = 5,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.inference_padding = inference_padding

        self.conv_pre = _WNConv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                _WNConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k in resblock_kernel_sizes:
                self.resblocks.append(_ResBlock1(ch, k, (1, 3, 5)))

        self.conv_post = _WNConv1d(ch, out_channels, 7, 1, padding=3, bias=True)

    @torch.no_grad()
    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: (B, 80, T) unnormalized log-mel -> waveform (B, 1, T * 256)."""
        x = F.pad(log_mel, (self.inference_padding, self.inference_padding), mode="replicate")
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0.0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))
        # Trim inference padding (5 * 256 = 1280 samples on each side)
        trim = self.inference_padding * 256
        if trim > 0 and x.shape[-1] > 2 * trim:
            x = x[:, :, trim:-trim]
        return x


class LogMelFrontend(nn.Module):
    """Computes 80-bin Slaney log-mels and synthesizes audio via pretrained 16kHz HiFi-GAN."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        hifigan_ckpt: Optional[str | Path] = "data/hifigan_16k_generator.ckpt",
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        fb = build_slaney_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", fb)

        self.vocoder: Optional[HifiGANGenerator] = None
        if hifigan_ckpt and Path(hifigan_ckpt).exists():
            voc = HifiGANGenerator()
            sd = torch.load(str(hifigan_ckpt), map_location="cpu", weights_only=True)
            voc.load_state_dict(sd, strict=True)
            voc.eval()
            self.vocoder = voc

    @torch.no_grad()
    def extract(self, wav: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Return globally normalized log-mel (1, 80, T) plus (MEL_MEAN, MEL_STD)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        # SpeechBrain HiFi-GAN uses power=1 (magnitude spectrogram)
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-9)
        mel = torch.matmul(self.mel_fb, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        norm_mel = (log_mel - MEL_MEAN) / MEL_STD
        return norm_mel, MEL_MEAN, MEL_STD

    @torch.no_grad()
    def mel_to_wav(
        self,
        norm_log_mel: torch.Tensor,
        mean: float = MEL_MEAN,
        std: float = MEL_STD,
    ) -> torch.Tensor:
        """Synthesize 1D waveform from normalized log-mel using pretrained 16kHz HiFi-GAN."""
        if norm_log_mel.dim() == 4:
            norm_log_mel = norm_log_mel.squeeze(1)
        elif norm_log_mel.dim() == 2:
            norm_log_mel = norm_log_mel.unsqueeze(0)

        log_mel = norm_log_mel * std + mean  # (B, 80, T)
        if self.vocoder is not None:
            return self.vocoder(log_mel).view(-1)
        raise RuntimeError("Pretrained HiFi-GAN checkpoint not loaded.")


def crop_or_pad(spec: torch.Tensor, target_frames: int, random_crop: bool = True) -> torch.Tensor:
    _, n_mels, t = spec.shape
    if t == target_frames:
        return spec
    if t < target_frames:
        pad = torch.full((1, n_mels, target_frames - t), -1.5, dtype=spec.dtype, device=spec.device)
        return torch.cat([spec, pad], dim=-1)
    start = int(torch.randint(0, t - target_frames + 1, (1,)).item()) if random_crop else (t - target_frames) // 2
    return spec[:, :, start : start + target_frames]
