# High WORLD energy retention

Leia's D#6 N is weak in the high WORLD branch. Native inputs retain the original score, including pitch bends and vibrato. In the same draw, donor N RMS is 0.06743 while WORLD N is 0.01069; preceding B5 u is 0.19639 versus 0.10777. This identifies a loss during resynthesis, separately from any weak learned donor output.

`tools/portable/high_world_energy.py` adds bounded donor/WORLD power compensation before the existing high-pitch blend. A centered 60 ms power window determines sqrt((donor_power+1e-8)/(world_power+1e-8)), bounded to 1–2 for vowels/m/n and 1–8 for standalone N. The N cap ramps over 20 ms. Active tokens are N, a/e/i/o/u and m/n. A 20 ms ramp at each active region boundary avoids gain steps. Inactive samples bypass exactly. This compensates WORLD loss; it cannot restore a weak donor or guarantee every high phoneme is audible. It is a gain correction, without EQ or saturation. It is unrelated to filling a decaying long-note tail.

The existing pitch-dependent blend determines where corrected WORLD reaches the final voice. Lower pitches with zero WORLD blend remain unchanged. SP gating remains after the blend. Notes, vocal F0, phoneme durations and learned weights are retained. Acoustic control channel 4 carries the frame mask, and channel 5 marks standalone N; install acoustic/vocoder together. Channels 0–3 retain the earlier silence and consonant-F0 controls.

Build and check:

```sh
python tools/portable/high_world_energy.py acoustic.onnx vocoder.onnx output
python tools/portable/check_high_world_energy.py output/energy.onnx LOCAL_NATIVE_FIXTURE
```

40 boundary/loss conditions verify finite output, gain bound and exact inactive bypass. Fixed native WORLD replay changes N RMS 0.010689 to 0.064444, +15.61 dB, without modifying inactive samples. New native OpenUtau synthesis gives N 0.011082 before correction to 0.062848 after, final peak 0.771176 and 135168 samples at 44.1 kHz. Control mask matches all tokens; trailing SP is zero. Other before/after differences include independent diffusion/WORLD random draws. Evidence and audio stay in local `DiffSinger/artifacts/tokine-high-nasal`. The user approved this comparison on 2026-10-12 with “OK”. The correction is adopted locally; the public release is unchanged. This approval does not establish every high phoneme as defect-free.

## Balance revision after listening

The user reported overly loud high ki and mild remaining rasp on N after approving the initial energy correction. Native E6 i had donor RMS0.2493, raw WORLD0.0592 and corrected0.2583, a roughly4.4x increase. The revised non-N cap is2x while standalone N retains8x. Fixed replay lowers i0.2583 to0.1183. New native synthesis lowers i0.2583 to0.1101, while N stays0.06285→0.06136. Peaks0.43174/0.69570, finite, same frame counts. Independent random draws contribute to before/after changes.

For standalone N only, WORLD aperiodicity coefficients are multiplied by0.75 in the interior, with a five-frame averaged boundary weight. This is a restrained synthesis change intended to reduce rasp; it is not proof that the rasp was a software defect. Non-N AP coefficients bypass exactly, spectral envelope/F0 remain intact. WORLD pulse responses can affect neighboring waveform samples, so exact non-N waveform bypass is not claimed for this AP change. Low pitches remain governed by the unchanged baseline blend. This trial may alter nasal timbre and needs listening approval.

80 compensation cases check distinct caps, short regions and inactive bypass;8 AP cases check finite range, maximum25% coefficient reduction and exact non-N AP bypass. Native acoustic channel5 equals the N frame mask for both phrases. Evidence in `DiffSinger/artifacts/tokine-high-balance`. The original score curves and notes are retained. This revision supersedes the initial8x vowel cap and is a local listening trial.
