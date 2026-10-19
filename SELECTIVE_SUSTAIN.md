# Selective sustain

`tools/selective_sustain.py` patches the existing original/donor sustain mix in a portable acoustic graph. It requires the existing `su_raw`, `su_loop`, `su_weight3`, and `su_wet` tensors. Apply once to an unpatched model:

```sh
python tools/selective_sustain.py before.onnx after.onnx
python tools/check_selective_sustain.py before.onnx after.onnx checks.json
```

The gate compares smoothed broadband mel amplitude proxies from raw and held spectra. The held reference falls back to raw outside the existing sustain mask, preventing artificial edges from inactive loop frames. Both envelopes use a centered nine-frame mean (about 104 ms at hop 512 / 44.1 kHz). Above a raw/reference ratio of 0.5 the hold is bypassed; below 0.25 the existing hold weight is retained. Smoothstep interpolates between these limits. Existing phone eligibility, duration limits, pitch compatibility and releases remain in force.

This reduces unnecessary spectral freezing on healthy medium and long vowels. It does not repair weak raw spectra without a usable donor. Existing vowel-connection processing remains separate. Learned initializers, input F0, vocoder and score timing are unchanged by this graph patch.

Verification on 2026-10-19:

- 192 synthetic branch / phone / duration / decay conditions: healthy bypass, fading-vowel support, protected silence and stop phones, finite outputs.
- All existing initializers retained byte for byte.
- Actual installed OpenUtau AU DiffSinger renderer: 27 phrases, 98.760703 seconds before/after, finite output and no clipping (full mix peaks 0.5757 / 0.5980). Native synthesis used depth 0.1 and 20 steps. Independent acoustic draws are not a controlled waveform comparison.
- Three additional fixed-input comparisons reuse identical native raw mel, durations and F0. Mean absolute postprocessing distance from raw for the reported section fell from 0.168879 to 0.017850; this measures processing magnitude, not perceptual quality.
- The listener reported less distortion with complete mel-postprocessing bypass. Listening approval of this selective candidate remains pending.

A separate score override squeezed a reported /t/ to 6.94 ms. Removing that one override restored 115.97 ms while preserving the following vowel onset. That score repair is not part of the model patch. Private scores, audio and native diagnostic inputs are kept outside this repository.
