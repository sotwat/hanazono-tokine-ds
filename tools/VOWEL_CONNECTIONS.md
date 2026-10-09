# Adjacent sustained vowels (local v1.2.1-rc.1)

`connect_sustained_vowels.py` extends the v1.2.0 / rc.4 acoustic ONNX wrapper.
It is a local audition candidate; the public release remains v1.2.0.

- For a sustained vowel followed directly by another vowel, return to the raw
  spectrum over the last 2 frames, instead of 6.
- If both adjacent vowels have at least 40 frames, use the incoming vowel's
  raw spectrum at local frame 12 at its onset, then fade to its regular spectrum
  over 10 frames (about 116 ms at hop512/44100 Hz).
- Apply this to all a/e/i/o/u pairs in both acoustic branches. No lyric matching,
  song positions, gain floor, note edits, or duration/pitch predictor changes.
- Consonant transitions and phrase-final release retain their existing behavior.
  A short incoming vowel receives no onset replacement; an eligible preceding
  sustained vowel still uses the shorter outgoing release.

```sh
python tools/connect_sustained_vowels.py baseline.onnx connected.onnx
python tools/check_vowel_connections.py baseline.onnx connected.onnx checks.json
```

The checker isolates both wrappers and feeds identical raw spectra: 25 vowel
pairs plus short-vowel, stop-consonant, continuous-consonant, and silence cases
per branch (62 total). Values outside eligible boundary windows remain exactly
unchanged. The existing sustain coverage check also passed.

OpenUtau AU rendered identical musical excerpts for the baseline and candidate.
At the affected connection, one 50 ms RMS window changed from 0.02596 to 0.15475,
and the following window from 0.06702 to 0.13308. Candidate peak was 0.51746,
with finite samples and unchanged output length. Independent diffusion draws
are included in these measurements; they are not a controlled effect estimate
or a subjective naturalness assessment.

Cache mapping was verified using the candidate's unique equality of onset and
local-frame-12 mel, combined with serialized renderer writes. File creation
order alone is insufficient to assign baseline/candidate labels.

Candidate acoustic SHA256:
`bd1191630769135cb26d15598df2f7197af14175d208d24eed849af4ed2e722f`

The learned weights are unchanged. The rule assumes the spectrum 12 frames
inside an eligible vowel is usable; broader listening remains necessary before
promoting this local candidate to a public release.
