# Portable high-register WORLD renderer

The acoustic ONNX emits two 128-bin mel batches: original F0 and a donor clamped to A3–C5. The paired vocoder consumes both, performs WORLD analysis/resynthesis using standard ONNX operators, and blends D5–F5. Below D5 its output is exactly the baseline branch from the same run. Do not mix these models with an ordinary acoustic model or vocoder.

This is a DiffSinger/WORLD hybrid, not a VOCALOID implementation. It requires a host that passes the acoustic mel tensor unchanged to the bundled vocoder, including batch size 2. It is not a universal DiffSinger format extension. No Python, custom ONNX operators, or external executables are needed at inference.

Build dependencies used: Python 3.10, torch 2.14.0, diffsptk 4.0.0, onnx 1.22.0, onnxscript 0.7.2, onnx-ir 1.0.0, numpy 1.26.4. Validation uses onnxruntime 1.23.2 and pyworld 0.3.4.

```sh
python tools/portable/probe_world_export.py BUILD/world-probe.onnx
python tools/portable/fix_inverse_dft.py BUILD/world-probe.onnx BUILD/world-fixed.onnx
python tools/portable/export_blend.py BUILD/blend.onnx
python tools/portable/build_pair.py BANK BUILD/world-fixed.onnx BUILD
python tools/portable/validate_pair.py BUILD
```

BANK is the local v1.1.2 sustained-vowel bank with `he_raw` as the vocoder's original waveform. Training weights are unchanged. The old high EQ/gain/saturation is bypassed, matching the approved WORLD audition. Long-vowel acoustic processing remains.

The export adaptation replaces data-dependent padding with a bounded 512-bin pad, searchsorted with binary search, and fftshift with slicing. Complex phase construction uses explicit sine/cosine. Export optimization must be disabled: its near-zero constant folding removed the D4C denominator epsilon. The inverse DFT fix reconstructs the Hermitian spectrum and disables invalid inverse+onesided flags. diffsptk/WORLD attribution and licenses accompany the bank.

The PyTorch WORLD implementation and the earlier pyworld prototype differ numerically and use random noise. Waveform identity to that prototype is not promised. This first portable build is a release candidate; subjective equivalence remains to be checked by the singer owner. Windows/Linux and other editors have not been tested.
