# Vowel sustain with preserved edges — rejected experiment

**2026-10-20: rejected by the user.** The user heard more distortion and preferred the previous version. Both local banks were restored byte-for-byte to acoustic SHA256 `807a109b5588497dae6b7644b041b009655fd73a2938ce761066f731496171cf`. Do not install this experiment as an improvement. Reduced envelope modulation and passing timing checks did not predict perceptual quality. The original sustain distortion and weak vowel attacks remain unresolved.

For reproduction only, build on the acoustic model that already contains `tc_*` short-vowel passes:

```sh
python tools/portable/stable_vowel_resynthesis.py INPUT.onnx OUTPUT_DIRECTORY
python tools/portable/check_stable_vowel_resynthesis.py OUTPUT_DIRECTORY
```

The previous restore mixed the short neural vowel according to an energy-deficit gate. In the reported D4 long /a/, its maximum weight was 0.0927, leaving most of the unstable original output. Its uniform time map also stretched the onset and release.

This patch replaces that restore map and blend window. It retains four real-time frames at each end of the neural time map and stretches the middle. A smooth window blends into the restored vowel over the first four frames and returns to the original boundary over the final six frames. The interior reaches full replacement independently of the energy-deficit gate. No waveform loop, cycle averaging, added vibrato, or amplitude boost is introduced.

Eligible voiced a/e/i/o/u phones have at least 20 frames, F0 above 60 Hz and a central pitch span below approximately 50 cents. The short pass uses half the duration, bounded to 12–24 frames. Unstable pitch cores bypass this path; this is not universal coverage of every long note. Existing articulation, donor mixing, vocoder, duration model and bounded vowel loudness processing remain installed. All pre-existing initializer bytes are retained.

## 2026-10-20 verification

Artifact: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-sustain-residual-20261020`.

- 768 map cases: token IDs, length threshold, pitch glides and edge pitch motion; ONNX/PyTorch agreement, monotonic maps and unchanged edges. Zero-duration phones, exact bypass and a full interior blend checked separately.
- Native OpenUtau rendering of four unchanged excerpts: long /ta/, long /te/, the second /a/ in「あの日あのとき」, and long「き→あ→か」. No phonemizer errors; before/after phone times and sample counts identical, outputs finite and below clipping.
- All 627 original initializers byte-identical.
- A diagnostic 12–80 Hz envelope modulation measure decreased by approximately 6 dB for the reported long /ta/. This is not a perceptual distortion score. Long /te/ did not improve on this metric; listening approval remains pending for all excerpts. Do not claim every distortion has disappeared or that timbre is perceptually unchanged.

Comparison WAVs preserve the phrase and use no additional gain or normalization. Source score is untouched.
