# Duration-adaptive vowel connections

`general_vowel_connections.py source.onnx output.onnx` updates the v1.2.2
acoustic model. It replaces the earlier fixed-frame vowel onset/morph stages,
retaining sustained-vowel synthesis and fricative connections. NumPy, ONNX and
ONNX Runtime are used by the exporter and checks.

Consecutive identical a/e/i/o/u tokens are grouped for the acoustic text
encoder and their durations summed. Groups split at note boundaries when the
note start crosses another 128-frame interval of the vowel run. This bounds
repeated-vowel context; an individual long note is not split. Unlimited
merging was rejected because some late vowels lost voicing. Silence and
consonants remain separate. Original per-note F0 and sustain boundaries remain,
so the sustain donor is not blindly held across every pitch change.

For a vowel preceded by another vowel, an onset reference is selected from
five positions at 25–65% of its own duration, using mean linear mel amplitude.
Indices stay inside the vowel and within the frame count. The earlier fixed
+12-frame reference could select a weak onset or cross a brief vowel.

The preceding vowel needs four frames and the current vowel six frames.
A smoothstep transition starts at most four frames before the boundary and
reaches the reference two frames after it. The reference blends back to the
existing output by half the vowel duration, capped at 24 frames. Zero-duration
inputs have guarded divisions. The former 40-frame requirement no longer
applies. All 25 vowel pairs are covered in both acoustic branches.

If a repeated vowel has a weak reference, the same uninterrupted vowel run
may provide a stronger reference within two semitones of the current pitch.
Replacement requires reference energy greater than 1.5 times local energy.
That reference is used with onset/release fades through the weak vowel.
Silence or a consonant ends eligibility for borrowing. F0 is not replaced and
no output gain/EQ is added. If no suitable voiced material exists anywhere in
the run, this mechanism cannot manufacture it.

Lyrics, note timing, phoneme overrides, model training weights and vocoder are
unchanged. The change is common acoustic inference processing, not a
song-specific score edit or retraining.

## Checks

Run `check_general_vowels.py before.onnx after.onnx checks.json`.
370 wrapper cases cover all vowel pairs, brief/long/zero-duration inputs,
consonants and silence. They check finite output and bit-identical spectra
outside applicable windows. Four grouping cases cover repeated vowels,
bounded context and preservation of silence/consonants. Four donor cases
check weak-vowel recovery and the prohibition on borrowing across silence.

## Native verification, 2026-10-10

Four complete phrases from the user's unchanged 180 BPM score were rendered
in OpenUtau AU v0.1.572.1. Short excerpts failed to reproduce all reported
weaknesses, so conclusions use full-phrase renders. Private scores and audio
are not in this repository. Output lengths match, samples are finite and no
waveform clips.

Approximate first-100-ms RMS at each reported vowel boundary:

| Connection | v1.2.2 | Local candidate |
| --- | ---: | ---: |
| a → a | 0.08936 | 0.07637 |
| u → a | 0.02053 | 0.05079 |
| i → a, preceded by hy | 0.01446 | 0.07607 |
| i → a, preceded by k | 0.00898 | 0.04535 |

For a → a the following 100 ms rose from 0.05739 to 0.07703, reducing the
within-vowel drop. The other three connections now start near their following
100-ms levels (0.04821, 0.06959, 0.04657). Boundary positions were estimated
from the original note timing and waveform pitch plateaus, not a manually
asserted sample-exact phoneme boundary.

Pitch error p90 over scored note interiors was 4.86 → 3.91, 9.98 → 6.00,
8.80 → 8.05 and 6.34 → 6.92 cents respectively, using WORLD estimation.
Unvoiced frames are counted separately. In the u → a phrase, all 982 evaluated
interior frames remained voiced; this rejected an earlier merge-only trial
that lost voicing in later notes. A final zero-duration guard was verified
bit-identical for nonzero-duration wrapper inputs used by these renders.

Independent diffusion draws are included. These measurements establish
reproduction and improvements in the tested phrases, not perceptual quality
for every song. Listening evaluation remains necessary.

This repair was published as v1.2.3. The subsequent long-note duration and
join work is documented in LONG_NOTE_CONNECTIONS.md.
