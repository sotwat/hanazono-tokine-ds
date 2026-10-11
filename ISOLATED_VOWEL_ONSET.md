# Isolated vowel onset smoothing

Rejected on 2026-10-12: the user still heard a raspy noise at the instant of i onset. This smoothing was removed from the installed bank. Numerical spectral smoothing did not establish a perceptual repair. See ONSET_PERIODICITY.md for the replacement trial.

The local acoustic wrapper smooths the first three frames of a vowel following SP. It handles a/e/i/o/u equally, without lyric or song matching. Only immediate neighbors inside the same phone and within 30 cents participate. Half of the normalized log-mel shape is mixed with the compatible neighbors; the original per-frame mean linear-mel energy is restored. Frames without a compatible neighbor remain exactly unchanged. F0, durations, training weights, WORLD processing and consonant timing are unchanged.

Reproduce with the DiffSinger Python environment:

```sh
python tools/smooth_isolated_vowel_onset.py before.onnx candidate.onnx
python tools/check_isolated_vowel_onset.py before.onnx candidate.onnx checks.json
python tools/render_isolated_onset_comparison.py before.onnx candidate.onnx vocoder.onnx native.npz output
```

`native.npz` contains OpenUtau debug tokens, durations, F0 and both raw acoustic spectra. Keep private score/audio fixtures outside Git. The comparison renderer reuses identical raw tensors; WORLD excitation can differ independently outside the baseline-only interval.

2026-10-12: Leia measure 6 onset i, tone C5, duration 240 ticks at 190 BPM, has no vibrato and a 368-cent pitch scoop. Its initial F0 lies below the WORLD blend threshold. Existing sustain/connection processing did not change the initial frames. This patch is a conservative mitigation, not proof that the underlying model caused the reported perceptual distortion. Fixed-raw comparison changes frames 8 and 9 (about 23 ms). Energy-relative error is 1.2e-7; normalized shape difference drops 0.334 to 0.167. All 840 locality/pitch/energy tests pass, and native OpenUtau output matches the candidate wrapper exactly. Waveforms are finite and unclipped. Perceptual resolution still requires listening to the comparison.
