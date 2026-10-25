# Vowel envelope revision — 2026-10-26

## Reproduction

The user's Backup/Track9 score uses DIFFS JA. The saved score changed from
120 to 180 BPM during diagnosis; final comparisons use the current 180 BPM.
Reproduce complete uninterrupted phrases, not just the reported notes:
11040–18480, 29760–37680, and 38400–48000 ticks. Cropping changes acoustic
context and can conceal the onset problem.

OpenUtau preferences report depth 1, but the bank's `max_depth: 0.1` caps the
actual inference at 0.1. Native cache filenames confirm this. This setting
was investigated and ruled out; it was not changed.

Artifacts: `DiffSinger/artifacts/tokine-root-envelope-20261026`.
The `whole/` fixtures preserve the phrase's lyrics, timing, pitches and curves.
`*-ui-cache.npz` contains matched real UI acoustic output. `whole/*-comparison.wav`
and the three Japanese-named short comparisons play before, 0.5 s silence, after.
No separate normalization is applied to those comparison files.

## Causes and changes

1. Repeated identical vowels are already merged in the neural input, yet an
   older postprocessor additionally applied a fixed-spectrum vowel morph.
   The /o/ onset held a weak anchor for about 13 frames. Weak repeated /a/
   could instead be replaced by a stronger single spectral frame from another
   note, suppressing natural spectral movement and creating a level jump.
   Bypass that duplicate same-vowel morph and the full-phone weak-anchor
   replacement. Different-vowel joins still use their local anchors.
2. Neural short-vowel restoration inherited the late onset window originally
   intended for fixed-spectrum sustain repair: zero through frame 12, ramping
   to full restoration at frame 20. Combined with the weak anchor this creates
   an artificial rise. Use a six-frame neural crossfade for the first 20 frames,
   retaining the existing failure gate, eligibility, and later release behavior.
   Healthy/bypassed phones are not forced through restoration.
3. Pitch/phoneme-dependent level differences are already present in the neural
   output. The existing balance's 1.5 dB deadband, 65% correction and 4 dB limit
   left substantial differences. Revise that existing layer to 0.75 dB,
   90%, maximum +6 dB / −12 dB. Gain stays constant within each vowel core;
   it does not follow the decaying tail. Voicing/noise exclusion, boundary fades,
   phrase peak protection and explicit host expressions remain in place.
   This is output level calibration, not retraining the neural model or proof
   that its learned level bias has disappeared.

No score-specific lyric matching, vibrato, EQ, saturation, new waveform loop,
phoneme-boundary shift, or change to learned weights is introduced.

## Build and verification

```
python tools/portable/envelope_revision.py BANK NEW_OUTPUT_DIRECTORY
python tools/portable/check_envelope_revision.py ARTIFACT_DIRECTORY BANK_BEFORE_UPDATE
```

Both `acoustic.onnx` and `vocoder.onnx` must be installed together. This tool
expects the preceding local bank's graph; applying it twice is rejected.
The check uses the retained `whole/` fixtures and `final-build/` outputs.

- 48 restoration scope cases: noneligible/healthy paths and frames after the
  onset window are unchanged; finite outputs.
- All 627 existing acoustic and 522 vocoder initializers remain byte-identical.
- The complete three-phrase comparison retains control data/F0 and tensor sizes.
- Native OpenUtau renders have no phoneme errors and identical phone metadata,
  timestamps and sample counts: 239616 / 255488 / 305664 at 44100 Hz.
- Native revised peaks: 0.255 / 0.442 / 0.337, finite and below clipping.
- Fixed-seed phrase comparison: the five target vowels in 解き明かそう have a
  central-RMS span of 6.59 dB before and 2.89 dB after. The repeated /a/ core
  is about 2.3 dB lower; its fixed spectral replacement is removed.
- This is measured verification. Auditory disappearance of every distortion
  and quality across every song have not been established.

## Local installation

Applied to both local v1.3.0-rc.1 banks; display name/version unchanged.
Acoustic SHA256: `2e256aed95eba1e4c4032e02bd4b88deabefcdd37af954513405c15df4756fda`.
Vocoder SHA256: `894c2dd6c6e1eb08b468bf09b4efb8ad653d69b50a836bea5f8f253e74642c81`.
All 36 manifest entries verified in each bank. Earlier models/metadata and
1604 DS cache files are retained under the artifact's `installed/` directory.
Other singers and source recordings are untouched. No GitHub release updated.

OpenUtau AU was restarted with the updated bank after preserving the saved
score. Track9's measure 18 /ga a ta/ waveform was visibly regenerated in the
piano roll. This verifies display/render integration, not auditory approval.

## Tool review

A web search for a dedicated DiffSinger debugging agent skill found no suitable
new dependency. The audio-debugging candidate in majiayu000/claude-skill-registry
addresses VSSK/Web Audio playback/cache diagnostics, not acoustic ONNX inference;
it was not installed. Existing Torch/ONNX/ORT and OpenUtau rendering were used.
