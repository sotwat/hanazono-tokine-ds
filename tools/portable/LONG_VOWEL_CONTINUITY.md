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
