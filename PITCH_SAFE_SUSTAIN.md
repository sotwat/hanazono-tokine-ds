# Pitch-safe sustain and finite WORLD aperiodicity

The Leia i→o example used a five-frame donor spanning about 50 cents. Holding that harmonic pattern while the requested F0 moved was an avoidable source of spectral mismatch. This is a plausible source of the reported rough sound; subjective resolution requires audition.

`tools/pitch_safe_sustain.py` adds two common acoustic safeguards to both original and donor branches:

- Search 32 donor centers at 20–85% of a 40–171-frame sustained phone. Prefer candidate windows whose pitch span is <=30 cents, falling back to the strongest window if none is stable.
- Average the five reference frames only when their pitch span is <=30 cents; otherwise use the center spectrum. Blend sustain back toward frame-local raw acoustic output when target pitch differs from the reference: full hold within 20 cents, smooth fade to no hold at 70 cents. The original score, F0, phoneme timing, and training weights remain intact.

At constant pitch, silence, stops, phones shorter than 40 frames, and the previous >=172-frame donor behavior remain exact. With pitch movement, longer sustained phones also receive the compatibility guard. The Leia donor span became 0 cents, and sustained-o RMS remained strong (0.076054→0.081420 in a fixed acoustic draw).

## WORLD zero-energy regression

Repeated F5/C6 synthesis captured an independent failure: D4C divided a zero band-energy numerator by a zero denominator. One invalid coarse aperiodicity value propagated to 279 analysis bins and 2,995 waveform samples. Baseline and donor neural waveforms were finite.

`tools/portable/guard_aperiodicity_ratio.py` uses ratio 1 (fully aperiodic) for zero-energy bands and bounds the log input to [1e-12,1]. Positive valid ratios in that range are bit-identical. This repairs the calculation rather than replacing invalid waveform samples with silence or adding a saturator/limiter.

```sh
python tools/pitch_safe_sustain.py adaptive-acoustic.onnx acoustic.onnx
python tools/check_pitch_safe_sustain.py adaptive-acoustic.onnx acoustic.onnx checks.json
python tools/portable/guard_aperiodicity_ratio.py paired-vocoder.onnx vocoder.onnx
python tools/portable/check_aperiodicity_ratio.py paired-vocoder.onnx vocoder.onnx failed-draw.npz world-checks.json
```

3,908 acoustic checks passed. Three replays of the captured failing donor produced 2,995→0 nonfinite waveform samples and exact valid D4C ratios. Seven registers C3/G3/C4/G4/C5/F5/C6 with five vowels are available for audition; WORLD excitation noise is independent between comparisons. Local evidence is under `DiffSinger/artifacts/tokine-pitch-safe-sustain/`. No claim that every timbral distortion is resolved follows from these numerical checks. Release packages are not updated by a local patch.
