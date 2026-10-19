#!/usr/bin/env python3
"""Measure spectral similarity of matched original and generated singing."""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import welch


def bands(wave, sr):
    frequencies, power = welch(wave, sr, nperseg=4096)
    total = power[(frequencies >= 80) & (frequencies < 8000)].sum()
    return {f'{lo}-{hi}Hz': float(power[(frequencies >= lo) &
                                         (frequencies < hi)].sum() / total)
            for lo, hi in ((80, 300), (300, 1000), (1000, 3000), (3000, 8000))}


def log_mel(wave, sr):
    spectrum = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=2048, hop_length=512, n_mels=80,
        fmin=40, fmax=16000, power=1)
    return np.log(np.maximum(spectrum, 1e-5))


def compare(original, rendered):
    reference, sr = sf.read(original)
    signal, render_sr = sf.read(rendered)
    assert sr == render_sr and reference.ndim == signal.ndim == 1
    length = min(len(reference), len(signal))
    reference, signal = reference[:length], signal[:length]
    gain = float(np.sqrt(np.mean(reference ** 2) /
                         max(np.mean(signal ** 2), 1e-12)))
    a, b = log_mel(reference, sr), log_mel(signal * gain, sr)
    active = librosa.feature.rms(
        y=reference, frame_length=2048, hop_length=512)[0] > .008
    active = active[:min(a.shape[1], b.shape[1])]
    mae = float(np.abs(a[:, :len(active)][:, active] -
                       b[:, :len(active)][:, active]).mean())
    return {'gain_aligned_log_mel_mae': mae, 'gain': gain,
            'original_rms': float(np.sqrt(np.mean(reference ** 2))),
            'rendered_rms': float(np.sqrt(np.mean(signal ** 2))),
            'original_bands': bands(reference, sr),
            'rendered_bands': bands(signal, sr),
            'original_seconds': len(reference) / sr,
            'rendered_seconds': len(signal) / sr,
            'active_frames': int(active.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    parser.add_argument('rendered', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.original, args.rendered), indent=2))


if __name__ == '__main__':
    main()
