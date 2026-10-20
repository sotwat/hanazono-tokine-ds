# ch attack revision

The local revision extends the existing ts affricate inference pass to ch.
A ch immediately before a/e/i/o/u with at least two frames is resynthesized
as a short learned t followed by sh. The stop uses one third of the span,
clamped to one to three frames. Both portions remain inside the original
ch span. One-frame consonants and other contexts bypass this expansion.

The resulting raw mel replaces only the original ch frames, in both acoustic
branches. Vowel timing, external F0, all phone boundaries, learned tensors,
existing ts behavior, duration ceilings and vocoder are preserved. The score
is not edited. Aliases that resolve to ch share the same treatment.

This is an approximation using existing phonemes, not additional training of
a new ch recording. Naturalness requires listening comparison, particularly
at the consonant-to-vowel join. It does not guarantee every ch will improve.

```sh
python ch_affricate.py PRIOR_TS_ACOUSTIC OUTPUT_DIRECTORY
python check_ts_affricate.py PRIOR_TS_ACOUSTIC OUTPUT_DIRECTORY/acoustic.onnx checks.json --include-ch
```

The map check covers 1,488 phoneme/duration/context cases and checks total
length, target locality, unchanged learned tensors and bit-identical
non-target selection with fixed spectra. Independent diffusion draws must
not be mistaken for an isolated causal waveform comparison.

Native OpenUtau AU rendering covers 24 isolated/control phrases at
F3/C4/C5/G5 and four connected phrases. Before/after phone positions,
durations and sample counts are identical. All audio is finite and unclipped
(maximum peak across both revisions 0.550549). A further 1,440 non-ch map
cases are bit-identical to the prior ts-only map. The first ch-containing
phrase of the user's ANIMA score is also rendered for comparison.
