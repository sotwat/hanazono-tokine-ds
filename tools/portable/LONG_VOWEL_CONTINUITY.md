# Long distinct-vowel continuity — 2026-10-26

The user approved the previous /o/ and 願った revisions, but heard an abrupt
き→あ handoff in 解き明かそう. Intermediate outputs show a release valley
before the separate learned /a/ onset. Equalizing whole-phone levels cannot
remove that boundary valley. The rearticulated /a/ and handoff layers then
produce a steep rise. This does not establish that all perceived differences
are amplitude alone.

Bridge only directly adjacent **different vowels** when the preceding phone
is at least 40 frames and the incoming phone at least 12 frames. The mel energy
at boundary−4 must be 0.4 log units below both boundary−8 and boundary+4.
Use a linear-amplitude smoothstep spectral bridge between those anchors.
The 12-frame window is about 139 ms; its midpoint precedes the boundary by
23 ms. No lyric/score matching or change to F0/phone timestamps is used.
This repairs the postprocessor handoff, not the trained model's weights.

```
python tools/portable/long_vowel_continuity.py SOURCE_ACOUSTIC NEW_OUTPUT
python tools/portable/check_long_vowel_continuity.py NEW_OUTPUT/link.onnx
```

Artifacts: `DiffSinger/artifacts/tokine-ia-continuity-20261026`.
Final build is `build4`; earlier builds are diagnostic, not installed.

- 1960 Torch/ONNX cases pass: short/identical/non-vowel/no-valley cases bypass,
  zero durations remain finite, changes stay in the boundary window.
- Fixed-seed full-context comparison: omoi and negatta audio are byte-identical.
- 解き明かそう changes only acoustic frames 351–362; control/F0 are identical.
  The 3-frame RMS valley around 355 changes from −37.3 to −23.2 dB, between
  neighbors around −24 to −22 dB. This is a local waveform measurement.
- Native OpenUtau render: no errors, same full phone metadata and 305664 samples,
  peak 0.374461, finite. Vocoder is unchanged.
- Installed into both local banks, manifests 36/36 verified. Prior acoustic,
  metadata and 220 DS cache files preserved in `installed/`.
- Acoustic SHA256: f509ea9f148a4615173d38e15cc331db5ce2d6788c322d279ce8a9e03a081b1b.
- Display version remains v1.3.0-rc.1; no public Release update.
- New transition's listening approval remains pending. Do not infer approval
  of this revision from the user's approval of the other two phrases.

The skill/tool investigation from ENVELOPE_REVISION.md remains applicable to
this immediate continuation; no dependency or additional skill was installed.

## Follow-up: softer attack

The user accepted smoother continuity but still found /a/'s attack strong.
Linear spectral mixing also interpolated total amplitude linearly, producing
up to 0.95 dB extra mel level on the original branch (2.18 dB donor branch)
relative to a log-level transition. Preserve the normalized mixed spectrum
and separately interpolate its total level in log amplitude. This changes
neither the bridge window nor the vowel core after it.

Artifacts: `tokine-ia-attack-20261026/build`. Fixed-seed /o/ and 願った remain
byte-identical. The boundary's measured three-frame RMS is about 1 dB lower;
the frame361 window differs by only 0.03 dB. Maximum within-frame spectral
shape deviation is below 1e-6 log units. The 1960 scope/parity checks and
additional energy-curve/spectral-shape assertions pass. Native OpenUtau:
no errors, identical phone metadata, 305664 samples, peak 0.319362.

Installed acoustic SHA256:
`d1d831db4687bf2af31e1ee45c7c9e9b7879fc598b557efc33f27cb32af4a494`.
Both 36-entry manifests verified; previous acoustic/metadata and 220 cache
files backed up. OpenUtau restarted and original score restored at 180 BPM,
DIFFS JA. This follow-up's listening approval is still pending.

### 2026-10-26 mo-to-i valley detection

Valley at boundary -2 was missed by the single -4 detector. Use minimum over -6..-1 while retaining gates, threshold, anchors and log-level blend. 6860 Torch/ONNX cases and spectral/level tests pass. Fixed-seed omoi changes only mel208-218; negatta and tokiakasou waves identical. Native OpenUtau omoi errors empty and phoneme metadata identical. Both local banks installed acoustic e709466e1a3bfe472f2c5545aee4403282b6af51dfce28f3d79ebcad8ed19e88, vocoder unchanged, manifest36/36 each. Backups include old models/metadata and220 cache files. AU restarted into original Backup Track9 DIFFS JA180BPM. Release unchanged. Previous i-to-a attack revision user-approved; new mo-to-i listening approval pending. Artifact /Users/aaa/Documents/DiffSinger/artifacts/tokine-oi-continuity-20261026; omoi-comparison.wav before/silence/after. Existing skill research reused, no dependencies added.
