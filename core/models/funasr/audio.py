# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Audio preprocessing for Fun-ASR model.

Implements mel-filterbank feature extraction with Low Frame Rate (LFR) processing.
"""

import math
from typing import Union

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_audio

# Audio hyperparameters for Fun-ASR
SAMPLE_RATE = 16000
N_FFT = 400  # 25ms window at 16kHz
HOP_LENGTH = 160  # 10ms hop
N_MELS = 80

# LFR (Low Frame Rate) parameters
LFR_M = 7  # Stack every 7 frames
LFR_N = 6  # Subsample by factor of 6


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> mx.array:
    """
    Compute the log-Mel spectrogram with hamming window.

    Parameters
    ----------
    audio : Union[str, np.ndarray, mx.array]
        The path to audio or audio waveform array
    n_mels : int
        The number of Mel-frequency filters
    n_fft : int
        FFT size
    hop_length : int
        Hop length for STFT
    sample_rate : int
        Sample rate of audio

    Returns
    -------
    mx.array, shape = (n_frames, n_mels)
        Log mel spectrogram features
    """
    if isinstance(audio, str):
        audio = load_audio(audio, sr=sample_rate)
    elif isinstance(audio, np.ndarray):
        audio = mx.array(audio)
    audio_np = np.array(audio, dtype=np.float32)
    return mx.array(_kaldi_fbank(audio_np, sample_rate=sample_rate, n_mels=n_mels))


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(freq / 700.0)


def _kaldi_mel_banks(
    n_mels: int,
    padded_window_size: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    num_fft_bins = padded_window_size // 2
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq += nyquist

    fft_bin_width = sample_rate / padded_window_size
    mel_low = _mel_scale(np.array(low_freq, dtype=np.float32))
    mel_high = _mel_scale(np.array(high_freq, dtype=np.float32))
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    bins = np.arange(n_mels, dtype=np.float32)[:, None]
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta
    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float32))[None, :]

    up_slope = (mel - left) / (center - left)
    down_slope = (right - mel) / (right - center)
    return np.maximum(0.0, np.minimum(up_slope, down_slope)).astype(np.float32)


def _kaldi_fbank(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    preemphasis: float = 0.97,
) -> np.ndarray:
    """Kaldi-compatible fbank matching FunASR WavFrontend defaults."""
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    waveform = audio.astype(np.float32) * (1 << 15)

    window_size = int(sample_rate * frame_length_ms * 0.001)
    window_shift = int(sample_rate * frame_shift_ms * 0.001)
    padded_window_size = _next_power_of_2(window_size)
    if waveform.shape[0] < window_size:
        return np.empty((0, n_mels), dtype=np.float32)

    num_frames = 1 + (waveform.shape[0] - window_size) // window_shift
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, window_size),
        strides=(window_shift * waveform.itemsize, waveform.itemsize),
    ).copy()

    frames -= frames.mean(axis=1, keepdims=True)
    previous = np.pad(frames, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    frames = frames - preemphasis * previous
    frames *= np.hamming(window_size).astype(np.float32)[None, :]

    if padded_window_size != window_size:
        frames = np.pad(
            frames,
            ((0, 0), (0, padded_window_size - window_size)),
            mode="constant",
        )

    spectrum = np.abs(np.fft.rfft(frames, n=padded_window_size, axis=1)).astype(np.float32)
    power = spectrum * spectrum
    mel_filters = _kaldi_mel_banks(n_mels, padded_window_size, sample_rate)
    mel_filters = np.pad(mel_filters, ((0, 0), (0, 1)), mode="constant")
    mel_energies = power @ mel_filters.T
    return np.log(np.maximum(mel_energies, np.finfo(np.float32).eps)).astype(np.float32)


def apply_lfr(
    features: mx.array,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Apply Low Frame Rate (LFR) processing to features.

    This stacks consecutive frames and subsamples to reduce the frame rate.
    Uses vectorized gather operations for efficiency.

    Parameters
    ----------
    features : mx.array, shape = (n_frames, n_mels)
        Input mel spectrogram features
    lfr_m : int
        Number of frames to stack (default: 7)
    lfr_n : int
        Subsampling factor (default: 6)

    Returns
    -------
    mx.array, shape = (ceil(n_frames / lfr_n), n_mels * lfr_m)
        LFR-processed features with stacked frames
    """
    T, n_mels = features.shape

    # Output length uses ceiling division
    T_lfr = int(math.ceil(T / lfr_n))

    # Left padding
    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        left_padding = mx.broadcast_to(features[0:1], (left_pad, n_mels))
        features = mx.concatenate([left_padding, features], axis=0)

    # Right padding to ensure we have enough frames
    T_padded = features.shape[0]
    total_needed = (T_lfr - 1) * lfr_n + lfr_m
    if total_needed > T_padded:
        right_pad = total_needed - T_padded
        right_padding = mx.broadcast_to(features[-1:], (right_pad, n_mels))
        features = mx.concatenate([features, right_padding], axis=0)

    # Create indices for all output frames
    # Shape: (T_lfr, lfr_m)
    start_indices = mx.arange(T_lfr) * lfr_n
    offsets = mx.arange(lfr_m)
    # Broadcasting: (T_lfr, 1) + (lfr_m,) -> (T_lfr, lfr_m)
    indices = start_indices[:, None] + offsets[None, :]

    # Gather frames: features[indices] -> (T_lfr, lfr_m, n_mels)
    gathered = features[indices]

    # Reshape to (T_lfr, lfr_m * n_mels)
    return gathered.reshape(T_lfr, -1)


def apply_cmvn(
    features: mx.array,
    cmvn_mean: mx.array = None,
    cmvn_istd: mx.array = None,
) -> mx.array:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN).

    Uses the formula: (features + mean) * istd
    where mean and istd come from precomputed statistics.

    If cmvn_mean and cmvn_istd are not provided, applies per-utterance
    normalization.

    Parameters
    ----------
    features : mx.array
        Input features
    cmvn_mean : mx.array, optional
        Additive shift (negative of mean)
    cmvn_istd : mx.array, optional
        Multiplicative scale (inverse std)

    Returns
    -------
    mx.array
        Normalized features
    """
    if cmvn_mean is None or cmvn_istd is None:
        # Per-utterance normalization
        mean = mx.mean(features, axis=0, keepdims=True)
        std = mx.std(features, axis=0, keepdims=True) + 1e-6
        return (features - mean) / std

    # Apply precomputed CMVN: (x + mean) * istd
    # Note: cmvn_mean is actually the negative mean (shift)
    return (features + cmvn_mean) * cmvn_istd


def preprocess_audio(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
    cmvn_mean: mx.array = None,
    cmvn_istd: mx.array = None,
    apply_normalization: bool = False,
) -> mx.array:
    """
    Full audio preprocessing pipeline for Fun-ASR.

    1. Compute log mel spectrogram
    2. Apply LFR (frame stacking and subsampling)
    3. Optionally apply CMVN

    Parameters
    ----------
    audio : Union[str, np.ndarray, mx.array]
        Input audio (path or waveform)
    n_mels : int
        Number of mel bins
    lfr_m : int
        LFR frame stacking count
    lfr_n : int
        LFR subsampling factor
    cmvn_mean : mx.array, optional
        Precomputed CMVN mean shift
    cmvn_istd : mx.array, optional
        Precomputed CMVN inverse std
    apply_normalization : bool
        Whether to apply CMVN normalization

    Returns
    -------
    mx.array, shape = (ceil(time / lfr_n), n_mels * lfr_m)
        Preprocessed audio features ready for the encoder
    """
    # Compute log mel spectrogram
    mel_features = log_mel_spectrogram(audio, n_mels=n_mels)

    # Apply LFR processing
    lfr_features = apply_lfr(mel_features, lfr_m=lfr_m, lfr_n=lfr_n)

    # Apply normalization
    if apply_normalization:
        lfr_features = apply_cmvn(lfr_features, cmvn_mean, cmvn_istd)

    return lfr_features


def compute_feature_lengths(
    audio_lengths: mx.array,
    hop_length: int = HOP_LENGTH,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Compute output feature lengths after preprocessing.

    Parameters
    ----------
    audio_lengths : mx.array
        Lengths of input audio in samples
    hop_length : int
        Hop length for STFT
    lfr_n : int
        LFR subsampling factor

    Returns
    -------
    mx.array
        Output feature lengths
    """
    # Frames after STFT
    n_frames = audio_lengths // hop_length

    # Frames after LFR (ceiling division)
    out_len = (n_frames + lfr_n - 1) // lfr_n

    return out_len.astype(mx.int32)
