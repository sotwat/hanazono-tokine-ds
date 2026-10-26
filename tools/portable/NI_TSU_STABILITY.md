# Ni / tsu residual stability (2026-10-26)

User accepted the ni AP trial and the tsu spectral smoothing trial. AP attenuation on tsu and removing the ts articulation pass were rejected. Keep the existing ts articulation.

Build from the installed sustain-envelope pair (acoustic 6368865e, vocoder aa331486):

```
python tools/portable/ni_tsu_stability.py BANK OUTPUT
python tools/portable/check_ni_tsu_stability.py
```

The acoustic model appends two context masks to control columns 9–10; acoustic mel generation and duration remain unchanged. Both require at least 20 frames. ni (n followed by i) fades in across MIDI 73–74. tsu (ts followed by u) fades in across MIDI 68–69. Mean phone pitch determines eligibility. Independent vowels and other consonant contexts are excluded.

ni attenuates WORLD aperiodicity by up to 70% below 7 kHz, tapering to zero attenuation at 12 kHz. tsu smooths WORLD log spectral power over seven frames at 85% strength. Both have four-frame smooth edges and bypass zero-weight frames exactly. These are context-limited synthesis corrections, not retraining or a guarantee for every song/register.

Evidence: artifacts/tokine-ni-tsu-residual-20261026 in the DiffSinger workspace. 162 context cases plus correction boundary checks passed. Final fixed-seed waveforms match accepted trials within 2e-7. Native OpenUtau render errors are empty for both fixtures and phrase metadata is unchanged. Early ts-removal comparisons had a random-generator invocation mismatch; subsequent comparisons start fresh fixed-seed sessions for each variant. The user selected spectral smoothing in the controlled comparison. Approval is limited to these auditioned sounds.
