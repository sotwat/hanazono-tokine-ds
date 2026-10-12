# High WORLD energy retention

Leia's D#6 N is weak in the high WORLD branch. Native inputs retain the original score, including pitch bends and vibrato. In the same draw, donor N RMS is 0.06743 while WORLD N is 0.01069; preceding B5 u is 0.19639 versus 0.10777. This identifies a loss during resynthesis, separately from any weak learned donor output.

`tools/portable/high_world_energy.py` adds bounded donor/WORLD power compensation before the existing high-pitch blend. A centered 60 ms power window determines sqrt((donor_power+1e-8)/(world_power+1e-8)), bounded to 1–8. Active tokens are N, a/e/i/o/u and m/n. A 20 ms ramp at each active region boundary avoids gain steps. Inactive samples bypass exactly. This compensates WORLD loss; it cannot restore a weak donor or guarantee every high phoneme is audible. It is a gain correction, without EQ or saturation. It is unrelated to filling a decaying long-note tail.

The existing pitch-dependent blend determines where corrected WORLD reaches the final voice. Lower pitches with zero WORLD blend remain unchanged. SP gating remains after the blend. Notes, vocal F0, phoneme durations and learned weights are retained. Acoustic control channel 4 carries the frame mask; install acoustic/vocoder together. Channels 0–3 retain the earlier silence and consonant-F0 controls.

Build and check:

```sh
python tools/portable/high_world_energy.py acoustic.onnx vocoder.onnx output
python tools/portable/check_high_world_energy.py output/energy.onnx LOCAL_NATIVE_FIXTURE
```

40 boundary/loss conditions verify finite output, gain bound and exact inactive bypass. Fixed native WORLD replay changes N RMS 0.010689 to 0.064444, +15.61 dB, without modifying inactive samples. New native OpenUtau synthesis gives N 0.011082 before correction to 0.062848 after, final peak 0.771176 and 135168 samples at 44.1 kHz. Control mask matches all tokens; trailing SP is zero. Other before/after differences include independent diffusion/WORLD random draws. Evidence and audio stay in local `DiffSinger/artifacts/tokine-high-nasal`. Listening approval is pending; the public release is unchanged.
