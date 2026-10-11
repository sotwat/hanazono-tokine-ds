# Adaptive sustain donor selection

The Leia i→o example exposed a weak donor in a 54-frame vowel. The raw acoustic output became stronger near its later interior, but the sustain wrapper searched fixed centers 10–30 and held an early weak spectrum.

`tools/adaptive_sustain_donor.py` replaces that fixed search with six centers at 25–75% of the phone duration, each scored over three frames (±2). The existing five-point constant spectrum hold remains; this change does not cycle donor frames or add gain. It applies to the existing sustained phone classes at 40–171 frames in both acoustic branches. Phones below 40 frames, stops, silence, and the existing >=172-frame long-note reference are unchanged. All candidate windows used by this range remain inside their phone.

Run against a bank already containing the stable sustain, general vowel connection, and short phone connection patches:

```sh
python tools/adaptive_sustain_donor.py source.onnx candidate.onnx
python tools/check_adaptive_sustain_donor.py source.onnx candidate.onnx checks.json
```

744 fixed-raw checks cover all 31 token IDs, durations 1–275, both branches, protected silence, and a late strong interior. In the reported Leia phrase, identical raw acoustic data and F0 produced sustained-o RMS 0.047604→0.074058 (1.56×, about 3.83 dB). This is waveform evidence of increased level, not a guarantee of perceptual balance across all songs. The score and pitch curve were preserved; no training weights or vocoder changes were made.

Local evidence: `DiffSinger/artifacts/tokine-leia-weak-o/`, including fixed-raw comparison, native AU output, and verification. Public release packages require a separate update.
