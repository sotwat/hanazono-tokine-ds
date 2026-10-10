# Long-note duration and sustain repair

The local v1.2.4-rc.1 candidate changes the duration-model pair and acoustic
postprocessing from v1.2.3. It uses standard ONNX operators and the existing
vocoder. Training weights, note positions, lyrics and user phone overrides
are preserved.

## Duration budgets

OpenUtau aligns a vowel and the next syllable's onset to consecutive note
anchors, stretching the predicted group to fit the note. In the old bank,
3-second groups could assign 0.5–0.9 seconds to the following consonant.
See OpenUtau's `DiffSingerBasePhonemizer.ProcessPart`:
https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Core/DiffSinger/DiffSingerBasePhonemizer.cs

`long_note_timing.py` exports a coordinated linguistic/duration pair. The
linguistic output carries the original 128-channel encoding, another
128-channel encoding with word lengths capped at 43 frames, and three
channels containing phone ID, alignment-group ID and the actual frame
budget. The duration model uses original predictions for short groups and
uses the reference encoding for groups of at least 40 frames (~464 ms).
Groups containing SP/AP, including the initial padding group, are excluded.

Consonant predictions are normalized to the 43-frame reference and bounded
by phoneme class (2–16 frames, roughly 23–186 ms). Their combined allocation
cannot exceed 40% of a group. The remaining frames are assigned to its
vowel(s), preserving the group's total duration and alignment anchors.
Stops, nasals, liquids, glides and fricatives have separate bounds. This is
a duration prior, not a claim that each linguistic context has one perfect
consonant length. Existing manual overrides still take precedence.

The matching `dsdur/dsconfig.yaml` must use `hidden_size: 259`. Both new
dsdur models must be installed together. The independent dspitch models
are unchanged.

## Sustained signal and joins

`long_note_sustain.py` starts the existing stable-spectrum sustain earlier
for phones of at least 172 frames (16-frame onset and 10-frame blend). It
retains the existing donor selection and fixed averaged donor spectrum.
Connected phones retain sustain up to the actual phone boundary rather
than fading back into a decayed raw signal. Phrase endings retain their
12-frame release. The same-vowel repair's release follows this context too.

The previous wide, fricative-only morph is replaced by short joins after
sustained phones: normally one frame on each side, two for fricatives and
affricates, shortened for brief consonants. This avoids extending a morph
far into a previous vowel. F0 is unchanged, and stops retain their closure.
A zero-duration guard prevents invalid interpolation widths.

## Reproduction

Run with the DiffSinger Python environment, NumPy, ONNX and ONNX Runtime:

```sh
python tools/long_note_timing.py BASE_BANK DURATION_OUTPUT
python tools/long_note_sustain.py BASE_ACOUSTIC CANDIDATE_ACOUSTIC
python tools/check_long_notes.py BASE_BANK CANDIDATE_BANK checks.json
```

Copy the two duration outputs into `dsdur`, set its hidden size to 259,
and replace the acoustic model. Preserve the paired vocoder and remaining
models. The checks cover short/zero and 40–860-frame duration groups,
all six vowel/sonorant IDs with 24 consonant IDs, and both acoustic branches.
Actual phrase rendering and listening remain separate from synthetic
wrapper checks. Windows, Linux and other editors have not been verified.

## Verification of the local candidate

- 2,016 duration cases: all six vowel/sonorant IDs × 24 consonant IDs ×
  14 lengths from 0 to 860 frames. Short-group predictions remain equal to
  the original within 1e-5; long groups preserve the complete frame budget.
- 960 acoustic wrapper cases across both branches, plus 16 zero/one-frame
  consonant edge cases: finite output and consonant interiors preserved.
- Five complete phrases rendered in OpenUtau AU v0.1.572.1 on Apple Silicon:
  equal sample counts before/after, finite waveform, no clipping. Evaluated
  note interiors were all voiced; candidate pitch error p90 was 3.17–5.16
  cents. Pitch-based alignment is an estimate, not a sample-exact boundary.
- Eight 3-second CV-to-CV examples and four 6-second notes (including N and
  G5): candidate evaluated vowel interiors were all voiced, pitch p90 under
  12 cents. Duration prediction and acoustic output were exercised together
  through ONNX Runtime. These stress examples are separate from native
  OpenUtau renders.

Diffusion renders use independent draws. These checks establish the
measured behavior, not perfect perceived pronunciation in every possible
score. Natural stop closures and voiceless consonants remain intentional.
