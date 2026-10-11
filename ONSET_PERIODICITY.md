# Isolated vowel attack trial

On 2026-10-12 the user rejected log-mel onset smoothing: a raspy noise remained at the instant of i onset. That smoothing was removed. The reported cause is still not conclusively identified.

The replacement trial has two parts, limited to vowels after SP:

- SP before a vowel keeps up to 12 ms of release, then fades out over up to 12 ms. Both intervals shrink for short rests. This removes the neural waveform leaking into an intended rest, while preserving a short release from preceding speech.
- The following vowel rises with a smoothstep over at most 8 ms (15% of short-phone duration). Pitch-synchronous averaging mixes the current sample with up to two cycles on either side, respecting the phone bounds. Sample F0 is integrated so the cycle lookups follow pitch bends. Wet amount 80%, fading back to dry between 50 and 75 ms. F0 below 60 Hz bypasses cycle averaging. The gate still follows the phone boundary.

Consonants, uninterrupted vowel pairs, sustained interiors and unrelated rests bypass this processing exactly. No fixed lyric/song matching, gain boost, score editing, retraining or new WORLD low-range synthesis is used. The previous attack smoothing is absent.

The acoustic ONNX appends a third 128-channel plane carrying eligibility and phone start/end frames. The first two mel planes are preserved. The paired vocoder reads this plane and applies the short attack filter after synthesis. These two files must be installed together; do not mix with the old two-plane vocoder. Native OpenUtau AU was verified with a [3,91,128] mel tensor; its control plane exactly matched the independently calculated phone boundaries. Other hosts remain unverified.

## Reproduce

```sh
python tools/portable/onset_periodicity.py acoustic-before.onnx vocoder-before.onnx output
python tools/portable/check_onset_periodicity.py output native-before.npz vocoder-before.onnx
```

The fixture is a private native OpenUtau debug tensor containing tokens/durations/F0/mel. It is not committed. Five synthetic pitches C3/C4/C5/F5/D#6 retain the test sinusoid to within 0.00036 amplitude and attenuate independently added white noise by 6.7–7.3 dB in the wet interval. This is a synthetic filter test, not a measurement of the reported noise. The real onset's harmonic residual test changed only about 0.29 dB with cycle averaging alone; it did not demonstrate a solved distortion. The pre-vowel SP residual RMS was 0.00270 and is zero in the last two SP frames after the gate. Those observations motivated the additional boundary gate, without proving that SP leakage was the only cause.

175 boundary cases check finite output and exact bypass for consonants/unrelated phones, including short durations and pitch bends. The integrated onset output matches the standalone filter. Output is finite and unclipped, peak 0.444 in the fixed-raw comparison; a native independent draw peaked 0.470. The finalized short-rest fade uses the same values as that native fixture.

Local install is a listening trial, not an approved public release. Audio comparison is under DiffSinger/artifacts/tokine-onset-periodicity. The user must hear the result before claiming the raspy onset was resolved.
