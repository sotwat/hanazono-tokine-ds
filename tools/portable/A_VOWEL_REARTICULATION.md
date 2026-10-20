# Independent learned /a/ onset trial

2026-10-20: the user rejected the moving-spectrum trial as sounding unchanged. That trial removed the fixed anchor locally but retained the underlying weak/coarticulated vowel. The +4 dB gain trial was also rejected. This trial changes neural context for the incoming onset; it does not add gain.

## Implementation

For /a/ following e/i/o/u lasting at least 40 frames, with /a/ at least 12 frames long:

1. Run an additional acoustic branch using the same trained weights. In this branch only, replace the preceding vowel token with SP and zero its F0. Compress the target /a/ to 12 frames in this branch and resample its conditioning F0. This requests a fresh short learned /a/. Original branch inputs and the score are unchanged.
2. Use the resulting original/donor raw spectra only at the selected /a/ joins. Read its moving voiced core (frames 5–9) with linear interpolation, avoiding its weak breath startup and release. Preserve the requested F0 for the actual waveform synthesis.
3. Crossfade over two frames before the target boundary, reaching the new onset spectrum at the boundary. Hold the moving new spectrum through half the vowel, then smoothly return to the production spectrum by 75%. The latter quarter, following consonants and unrelated notes retain production processing.

This bypasses both the fixed GV anchor and the sustain blend **within the incoming connection window only**. The core resampling is confined to the onset/connection window. No changes to the duration predictor, vocoder, trained tensors, EQ, saturation, or sustained portions outside that window. Repeated /a/, consonant and rest onsets stay on the production path. It is a common conditional rule; no song positions or lyric phrases are embedded.

```sh
python tools/portable/a_vowel_rearticulation.py ACOUSTIC.onnx OUTPUT_DIRECTORY
python tools/portable/check_a_vowel_rearticulation.py ACOUSTIC.onnx OUTPUT_DIRECTORY
```

## Evidence

Artifact: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-rearticulation-20261020`.

- 6,146 token/duration and multiple-target conditions validate independent context changes, exact spectral bypass outside target windows, frame mapping, finite output and preservation of every production initializer.
- Actual native「き→あ→か」inputs with shared diffusion/vocoder seeds: only mel frames 128–167 change, control channels remain exact. Both outputs have 187,904 samples. Spectral shape RMS after subtracting per-frame means is 0.475, and waveform residual RMS over the first 26 /a/ frames is 108.5% of baseline RMS. These demonstrate a non-scalar output change; they do not prove audible improvement.
- Separate native OpenUtau candidate rendering (`native4`) validates integration, original phone metadata and duration.
- Comparison files are baseline → 0.5 seconds silence → new trial, PCM24/44.1 kHz mono. No extra normalization or gain. Final files are `き→あ_再制作_元→新試作.wav` (short connection) and `き→あ→か_再制作_元→新試作.wav` (full context). The crop is made from the same comparison generation.

An earlier internal full-length independent onset retained a weak middle and was discarded before delivery. Final build is `build4`; those intermediate artifacts are not the submitted trial.

This remains uninstalled and unapproved. Both production banks remain acoustic `807a109b5588497dae6b7644b041b009655fd73a2938ce761066f731496171cf` and vocoder `b5d848b24172b8cd1e10a024f2f0002e6b760750767bb030a68f7a21c032a3b4`. A regenerated onset can still differ in articulation or timbre; listening approval is needed. Remaining sustained distortion is not declared resolved.

## Follow-up: residual distortion, longer-core trial

The user confirmed improved volume in build4 but reported slight distortion. This is partial approval, not acceptance of the whole onset. Production is unchanged.

`a_vowel_core_revision.py` keeps build4 as a reference and generates an additional 24-frame independent /a/. Its core frames 6–21 replace the former 5–9 core. This reduces spectral time expansion (for the 52-frame target: about 9.75× to 2.6×) without expanding the edited connection window. Both branches share all trained tensors. At each affected frame, the new spectrum receives a scalar log offset so that its log-sum-exp matches the prior accepted-level spectrum. This preserves the spectral level contour; it does not guarantee identical waveform RMS or timbre. No added waveform gain, saturation, EQ, score edit or changes to global sustain processing.

The short-core stretching is a **suspected contributor**, not a proven cause of the audible distortion. The longer core may change articulation or timbre; perceptual approval is still required. Two extra acoustic branches increase inference work; this remains a trial outside the registered singers.

```sh
python tools/portable/a_vowel_core_revision.py ACOUSTIC.onnx OUTPUT_DIRECTORY
python tools/portable/check_a_vowel_core_revision.py ACOUSTIC.onnx OUTPUT_DIRECTORY
```

Artifact `tokine-a-rearticulation-20261020/core24`: 6,146 prior checks plus 480 longer-core/level/scope cases pass. Fixed native inputs and matched random seeds change only frames 128–167; controls and unrelated mel frames remain exact. Maximum contour log error is 3.1e-6. The 50ms target waveform RMS windows differ by −0.96 to −0.004dB from the prior trial, preserving its improved level approximately. Both have 187,904 samples, finite output and no clipping (candidate peak 0.3402). This measures scope and level, not distortion removal.

Listen to `き→あ_歪み再調整_前回→今回.wav` or full-context `き→あ→か_歪み再調整_前回→今回.wav`. Each is the volume-approved build4 → 0.5 seconds silence → longer-core trial, with no export normalization. The previous original-production comparison is a different file. Native integration results are recorded separately in `core24/native-verification.json`.
