# High vowel WORLD excitation trial

Status: approved by the user and installed in both local banks (2026-10-21).
GitHub release assets have not been updated. Installed vocoder SHA-256:
`19b6b27991631b5e3b5426a719cdec1357895f0b95f62c7d8846fbf5a1954201`.
Both manifests were verified. The normal AU UI rendered 158208 finite samples
after restart; installation evidence is in the artifact `installed/` directory.

The current G5–F#5 passage in 「ぶたい」 uses the WORLD branch fully.
Captured WORLD analysis shows substantial aperiodicity during its vowels.
This is a plausible source of breathy noise, not proof of the entire perceptual cause.

`high_vowel_aperiodicity.py` inserts a correction after the existing nasal AP
correction and before WORLD synthesis. Only vowel control channel 6 is eligible.
The register fades in from D5 to F5; below D5, consonants, nasal N, silence,
and unvoiced frames retain identical AP. Five-frame smoothing is multiplied by
the original eligibility mask, so it cannot extend the correction into a consonant.

At full strength, AP amplitude is multiplied by .55 below 6 kHz, tapering to
unchanged at 10 kHz. WORLD spectral envelope, target F0, acoustic model, timing,
gain and existing volume balance remain unchanged. A change in noise excitation
can still change perceived timbre; acceptance requires listening.

```sh
python tools/portable/high_vowel_aperiodicity.py VOCODER.onnx NEW_DIRECTORY
python tools/portable/check_high_vowel_aperiodicity.py NEW_DIRECTORY/high-vowel-ap.onnx
```

Local evidence: `artifacts/tokine-high-hoarseness-20261021` in the DiffSinger
workspace. Final trial is `build2`, not the superseded build with an incorrect
control rank. Captured mel and seeded vocoder comparison preserve 158208 samples,
all 522 original initializers, and AP outside the eligible mask exactly.
Peaks: .35150 before, .33015 after; both finite and unclipped. These are processing
and scope checks, not a claim that the perceived hoarseness has been eliminated.
