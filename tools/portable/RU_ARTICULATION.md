# R before U articulation trial

Status: listening trial, not installed or released (2026-10-21).

`ru_articulation.py` adds bounded spectral contrast within r immediately before
u, using the following vowel as the reference. It targets every eligible r/u
pair, regardless of lyric. R needs at least two frames and U at least three.
A sine-squared window and per-bin ±0.65 log limit bound the correction.
Per-frame log-sum-exp is restored, rather than adding output gain.

Following vowel mel, other phones, timing, F0 and controls remain identical.
Both original and donor mel branches receive the same rule. This does not add
new recorded flap articulation or retrain the model; perceptual improvement
requires listening and the consonant's timbre may change.

```sh
python tools/portable/ru_articulation.py ACOUSTIC.onnx NEW_DIRECTORY
python tools/portable/check_ru_articulation.py NEW_DIRECTORY/ru.onnx
```

Evidence: `artifacts/tokine-ru-articulation-20261021` in the DiffSinger workspace.
1024 scope/energy conditions passed. All 627 existing learned initializers are
preserved. Seeded comparisons changed only r frames 67–69 in おどる and 95–98
in するたび; control and all non-r mel are exact. Native OpenUtau DIFFS JA
rendering completed with no errors, identical phone metadata and 47104/157696
samples. Native peaks .45256/.46588 are finite and unclipped. These checks prove
scope and operability, not better perceived pronunciation. The candidate retains
the newly approved high-vowel AP vocoder; installed acoustic remains unchanged.
