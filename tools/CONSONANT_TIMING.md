# Consonant anticipation ceilings

The local duration revision extends `ts_timing.py` to all 23 onset consonant
IDs in this bank. Use `consonant_timing.py` on the prior ts-corrected duration model;
it retains the earlier ts patch and the coordinated linguistic model.

| Phonemes | Maximum frames | Approx. ms |
| --- | ---: | ---: |
| b/by/d/g/k/ky/m/n/p/t | 6 | 70 |
| ch/f/h/hy/j/ts | 7 | 81 |
| r/ry | 4 | 46 |
| w/y | 5 | 58 |
| s/sh/z | 9 | 104 |

These are corrective ceilings, not fixed natural durations or a claim that
all contexts should have the same timing. Shorter predictions are retained.
Consonant clusters share a nine-frame budget. Initial groups use absolute
frame counts; subsequent groups convert the ceiling into their normalized
prediction scale. Released time is returned proportionally to real vowels
in the same group. Vowel anchors and group sums are preserved. SP/AP, N,
explicit closure `cl` (促音), user note positions and editor overrides retain
their respective behavior. Groups without a real vowel bypass except the
initial padding group. No acoustic weights or vocoder changes are made.

## Reproduce

```sh
python tools/consonant_timing.py PRIOR_BANK/dsdur/tokine_song_variance_v1.dur.onnx candidate.onnx
python tools/check_consonant_timing.py PRIOR_BANK candidate.onnx checks.json
```

9,504 real duration-model conditions cover 23 onset IDs plus closure control,
six vowels, eleven lengths (including zero), three pitches and two group
contexts. Fifty additional cluster/empty-cluster conditions test the total
budget. Outputs are finite, closure is unchanged, initial SP is unchanged,
and noninitial group sums are preserved.

A separate native OpenUtau AU phonemizer fixture covers 552 combinations:
23 consonants, tempos 90/193, rests 0/15/120/960 ticks and note lengths
120/480/2880 ticks. Every vowel onset is exactly equal before/after;
456 consonant spans change. Phrase partition counts change from 750 to
801 because shorter anticipation no longer bridges some rests; comparisons
therefore join phones by part before checking anchors. Tick quantization
makes the measured ceiling a few milliseconds larger than its frame value
(maximum observed: 107.51 ms for s).

For the reported ANIMA「ほら」phrase, h changes from 185.88 to 80.96 ms.
All following phones retain exactly the same positions and lengths.
Independent diffusion renders are unsuitable for attributing every waveform
difference to this duration change. Native waveform checks establish finite,
unclipped output, not listening acceptance of every possible pronunciation.

Native rendering also covers all 23 onset consonants at 193 BPM with a
120-tick rest and 480-tick note. All outputs are finite and unclipped
(maximum peak 0.365815). The installed bank reproduces the candidate phones;
the normal UI creates the corresponding h-phrase cache with 25,600 samples
at 44.1 kHz, peak 0.291351. Listening acceptance remains pending.
