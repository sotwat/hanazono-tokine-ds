# Sustained vowel envelope revision (2026-10-26)

The previous per-phone balance multiplied each vowel by one constant; it could not remove a swell within that vowel. Backup measure 11 /te/ retained a 3.44 dB slow RMS range in its sustain. The user confirmed that leveling reduced the swell but left pre-existing distortion unchanged. Extending the existing WORLD route in this register reduced that distortion; the user accepted the timbre.

`sustain_envelope.py BANK OUTPUT` builds a replacement vocoder only. It preserves the acoustic model, token/duration/F0 inputs, existing i/u routing, and phrase peak guard.

- Envelope: a/e/i/o/u, at least 36 frames (~418 ms), reference RMS above .008. Central 60% power sets the reference. An 11-frame phone-bounded power window followed by a 9-frame gain window estimates slow variation. Gain is interpolated per sample, capped at -6/+3 dB. First/last 60 ms are untouched, with an 80 ms smooth transition to full correction. It precedes the existing per-phone balance and final peak guard. User DYN downstream remains available.
- Source: same sustained vowels, full existing WORLD route at mean phone MIDI 57–72, smooth one-semitone margins 56–57 and 72–73. Mean log F0 gives a constant weight within each phone, preventing vibrato from repeatedly crossing routing thresholds. Existing greater route weights are retained. Other ranges keep their current source; envelope leveling still applies.
- This is a bounded inference correction, not retraining. It does not promise mathematically constant loudness or the elimination of every distortion. Nasal N and consonants are excluded from envelope leveling. Onset, release, short vowels and intentional user dynamics are not flattened.

Validation: `python check_sustain_envelope.py OUTPUT/sustain-layer.onnx`: 64 envelope cases and 264 source conditions; Torch/ORT agreement, shape/finite, edge preservation, bypasses, synthetic swell reduction, route continuity and no reduction of prior weight. Actual fixed-seed /te/ sustain range 3.44→0.50 dB; /i/ 2.77→0.63 dB. Accepted /te/ trial and final output identical. Native OpenUtau render errors empty and phone metadata identical. Other vowel contexts still require listening; numeric checks do not prove universal perceptual quality.

Local evidence: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-sustain-envelope-20261026` (fixtures, comparisons, final verification, native-approved). Previous vocoder and metadata are retained in `installed/` when deployed. Restore those files together and restart OpenUtau to roll back. No public Release update.
