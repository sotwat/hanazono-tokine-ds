# Connected vowels and note level variation

Local revision, 2026-10-20. Listening acceptance is pending; this is not a release.

The previous vowel balance only raised quiet vowels by at most 1.5 dB. It never attenuated loud vowels. The replacement uses the central 60% of each voiced vowel to choose a constant per-phone correction: 1.5 dB deadband, 65% correction, capped at ±4 dB. It does not follow the decaying tail. A pitch-period correlation threshold of 0.25 excludes noisy/unvoiced material from gain estimation and correction. Silence and consonants bypass this stage. The existing final phrase gain and peak guard remain downstream, so they can still change overall phrase level.

The acoustic revision anticipates the *spectral* transition to a connected vowel by up to 19.97 ms. Eligible incoming phones are a/e/i/o/u after another vowel lasting at least 20 frames, with incoming duration at least 6 frames. A cubic smooth window starts 4 frames before the boundary and restores the original timeline within the first half of the incoming phone, at most 6 frames. This replaces some outgoing spectrum earlier rather than raising the attack. It does not change phoneme timing, pitch, score notes, or the general sustained-vowel generator. It does not prove the user's overheld-tail hypothesis, nor guarantee natural attacks on every combination.

Build against the previously approved core24 acoustic + vowel-balanced vocoder:

```
python tools/portable/level_handoff_revision.py BANK OUTPUT
python tools/portable/check_level_handoff.py
python tools/portable/check_vowel_balance.py OUTPUT/balance.onnx
```

Use DIFFS JA to compare. The optional TOKINE JA plugin independently advances boundaries and is not required for this revision; using both can anticipate transitions twice.

Evidence: local artifact `DiffSinger/artifacts/tokine-level-handoff-20261020`. 1,048 layer checks, 504 balance checks, 7 noise/dynamic ONNX checks passed. Two native OpenUtau excerpts have identical phoneme metadata and sample counts before/after (165,888 and 297,984 samples), finite output and no clipping. All old ONNX initializers are byte-identical. Same-input/seed vocoder comparison reduces the selected /a,o/ central-RMS spread from 11.28 to 8.25 dB. This is one measured excerpt, not a claim that all perceptually undesirable volume variation is resolved. The spectral handoff also needs user listening judgment.

Installed acoustic SHA256: `3f9c77fe72582286ab2eeadc903b504ca08c52a0a50017185beb5073603153c5`.
Installed vocoder SHA256: `25d101bfbf563009a6c68e24f4cc865e10d06aad76a41e55dfa6a8a60dec3c9b`.
Both registered and local banks retain display `花園トキネDS v1.3.0-rc.1`. The bank default is DIFFS JA. Old models, metadata and caches are recoverable in the artifact directory. Other singers and user note/pitch data are preserved. GitHub release assets are unchanged.
