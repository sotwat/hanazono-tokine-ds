# Independent vowel envelope trial — 2026-10-26

The user clarified that the unwanted /a/ is a `<>` envelope, perceived as fortepiano, and wants a steadier `==` sustain. Lowering the complete vowel or balancing adjacent pitches does not address this requirement.

Inspection found that `a_vowel_core_revision.Contour` deliberately copies the 12-frame independent onset's level trajectory onto the longer 24-frame regenerated spectrum. The old onset is crossfaded back into production between 50% and 75% of the vowel. Intermediate outputs show the copied rise and fall. This is a verified source of envelope motion, not proof that it explains every perceptual artifact.

`a_vowel_plateau.py` replaces that contour transfer. For the existing independent /a/ eligibility rule only, it takes the mean log-mel amplitude at 75–85% of the vowel, after the artificial onset handoff, as the sustain level. Correction is bounded to 3 dB and eased over four frames at each end. The regenerated spectrum, trained weights, F0, duration and following consonant remain intact. This is a correction to the imposed level trajectory, not retraining or a guarantee of natural articulation. A late reference could itself be weak in other examples; the bound limits that risk.

Reproduce:

```sh
python tools/portable/a_vowel_plateau.py CURRENT_ACOUSTIC.onnx NEW_OUTPUT_DIRECTORY
python tools/portable/check_a_vowel_plateau.py NEW_OUTPUT_DIRECTORY
```

Artifact `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-plateau-20261026` includes seeded before/after full phrases and intermediate outputs. 245 dynamic token/duration cases pass ONNX/Torch comparison, target scope, spectral-shape preservation at this layer, finite output and the gain bound. `omoi` and `negatta` acoustic outputs are unchanged. `tokiakasou` changes frames 351–408 including the existing transition bridge; control channels are exact. Both output peaks are below 0.31. Its 35-ms waveform windows at frames 377/389 move from -19.99/-23.92 to -21.59/-21.75 dBFS. Other within-vowel fluctuations remain; this is not a claim of perfectly flat waveform loudness.

Trial only. Neither installed bank nor published release is replaced. Listening acceptance is pending. Compare `tokiakasou-comparison.wav`: existing version, 0.5 seconds silence, trial. No export normalization.

Native OpenUtau trial renders with no errors and exact phone metadata versus the prior version. All 627 existing initializers are byte-identical. Both unaffected full-phrase waveforms are byte-identical under fixed seeds.

## Provisional local adoption

2026-10-26: user requested installing this trial for now. Both Mac banks now contain the exact trial acoustic ca2c97657d4f90f6acaa08f8c60d7bd26fac117a41a024446c508c99fa44d332, with vocoder d61a1bd0 unchanged. All 36 manifest entries per bank verified; previous acoustic/metadata and 87 cache files preserved in artifact installed/. AU restarted into the original Backup score, 180 BPM and Track9 DIFFS JA. Version label and public release unchanged. This is provisional adoption, not confirmation of all quality issues resolved.

2026-10-26 導入済bankをnative OpenUtauで再合成：errors空、試作との全phone metadata一致、44100Hz・305664samples・finite/nonclip確認。
