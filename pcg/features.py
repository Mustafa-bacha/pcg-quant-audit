"""Signal preprocessing and the log-mel front-end.

Everything here is deterministic and shared by all models so that the only
difference between the raw-waveform and log-mel models is the representation
(that contrast is what tests hypothesis H1).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt

FS = 4000  # CirCor sample rate, verified in p0_verify_data.py

# Band limits shared with Kutlu & Coban (JCM 2026) and Ludera & Kersten-Oertel (MICCAI 2026 MIRASOL),
# whose ablation found 25-400 Hz > no filter ~ 25-1000 Hz. See docs/lit_update_2026-10-20.md.
BAND_LO, BAND_HI = 25.0, 400.0


def bandpass(x: np.ndarray, fs: int = FS, lo: float = BAND_LO, hi: float = BAND_HI, order: int = 4) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass (25-400 Hz by default)."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-recording zero mean, unit variance. Loud S1/S2 set the scale: that is the
    point of H1 (a quiet murmur occupies few INT8 levels after this scaling)."""
    return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)


def preprocess(x: np.ndarray, fs: int = FS) -> np.ndarray:
    return standardize(bandpass(x, fs))


# --------------------------------------------------------------------------- log-mel

def _hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, fs: int, f_min: float = BAND_LO, f_max: float = BAND_HI) -> np.ndarray:
    """Slaney-style triangular filterbank, shape (n_mels, n_fft//2 + 1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, fs / 2, n_bins)
    mel_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_pts = np.array([_mel_to_hz(m) for m in mel_pts])
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    # area normalisation
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, None]
    return fb


class LogMel(nn.Module):
    """Waveform (B, T) -> log-mel (B, 1, n_mels, frames). Pure torch so it exports to ONNX
    and can sit in front of a quantized backbone (the front-end itself stays float).

    Defaults at 4 kHz: 50 ms window (n_fft 200 -> 256 padded), 25 ms hop (100 samples), 64 mels over
    25-400 Hz. These match the JCM 2026 benchmark (50 ms / 25 ms / 64 mels / 25-400 Hz at 2 kHz)
    so that a 5 s window gives 64 x 201 frames in both pipelines.
    """

    def __init__(self, fs: int = FS, n_fft: int = 256, win: int = 200, hop: int = 100, n_mels: int = 64,
                 f_min: float = BAND_LO, f_max: float = BAND_HI):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        w = torch.zeros(n_fft)
        w[(n_fft - win) // 2 : (n_fft - win) // 2 + win] = torch.hann_window(win)  # centred, zero-padded window
        self.register_buffer("window", w, persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(n_mels, n_fft, fs, f_min, f_max)), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True, center=True)
        power = spec.real.pow(2) + spec.imag.pow(2)  # (B, n_bins, frames)
        mel = torch.matmul(self.fb, power)  # (B, n_mels, frames)
        logmel = torch.log(mel + 1e-6)
        # per-example standardisation so the backbone sees a fixed scale
        m = logmel.mean(dim=(1, 2), keepdim=True)
        s = logmel.std(dim=(1, 2), keepdim=True) + 1e-5
        return ((logmel - m) / s).unsqueeze(1)


class SpecAugment(nn.Module):
    """Time and frequency masking on (B, 1, F, T). Train-time only."""

    def __init__(self, freq_mask: int = 8, time_mask: int = 20, n_masks: int = 2):
        super().__init__()
        self.freq_mask, self.time_mask, self.n_masks = freq_mask, time_mask, n_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, _, F, T = x.shape
        x = x.clone()
        for b in range(B):
            for _ in range(self.n_masks):
                f = int(torch.randint(0, self.freq_mask + 1, (1,)))
                f0 = int(torch.randint(0, max(F - f, 1), (1,)))
                x[b, :, f0 : f0 + f, :] = 0
                t = int(torch.randint(0, self.time_mask + 1, (1,)))
                t0 = int(torch.randint(0, max(T - t, 1), (1,)))
                x[b, :, :, t0 : t0 + t] = 0
        return x
