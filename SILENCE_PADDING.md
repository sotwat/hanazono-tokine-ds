# Silence segment waveform gating

The user rejected both isolated i-onset trials on 2026-10-12: the reported rasp occurs earlier than their exported comparisons. Those fixtures omitted the preceding phrase. The cycle averaging and vowel attack filter were withdrawn.

The original Leia autosave was exported as a full track through OpenUtau AU. Matching the native phrase WAVs against the track located the previous phrase at 4.736712 seconds and the following phrase at 6.854467 seconds. Before the i note at 6.947368 seconds, most of the measured residual comes from the previous phrase's SP release/padding, which the former pre-vowel filter did not cover. This identifies the source of a signal; it does not by itself establish that it is exactly the user-reported rasp.

`tools/portable/silence_padding.py` appends a control plane to the existing two mel planes. Each SP segment records frame bounds and whether a real phone borders it. The paired vocoder applies a waveform gate only inside SP:

- After a phone, smoothstep release over at most 25 ms, capped at half the SP length.
- Before a phone, permit a smooth anticipation over at most 5 ms, capped at 15% of the SP length.
- Leading padding has no outgoing release. Trailing padding has no upcoming attack. Consecutive SP segments do not restart a release.
- Non-SP samples bypass exactly. AP/breath phones are not SP. Score, pitch, phoneme duration, learned weights and the previous sustain/WORLD processing are retained.

The acoustic and vocoder files must be installed as a pair. Other hosts are unverified. The user confirmed on 2026-10-12 that the reported i-onset rasp was fixed. This processing is retained in the local bank; no new public release has been published.

```sh
python tools/portable/silence_padding.py baseline-acoustic.onnx baseline-vocoder.onnx output
python tools/portable/check_silence_padding.py output/silence-filter.onnx
```

120 boundary/length/flag cases check finite output, no peak increase, exact non-SP bypass and fully silent unconnected padding. Private source audio, native tensors and models remain in the local `DiffSinger/artifacts/tokine-onset-periodicity/original-context` directory.

Native full-track export verification: 207.409138 seconds, 44.1 kHz mono. In the 6.84–6.93 second pre-i interval, waveform RMS fell from 0.005719 to zero (all samples zero). The 4.5–9.5 second comparison includes the previous phrase and retains the original score. Native 98/192-frame control tensors also show silent SP interiors. The user confirmed resolution of the reported rasp; the earlier high phrase already contains PCM-limit samples, so this does not establish that all distortion is fixed.
