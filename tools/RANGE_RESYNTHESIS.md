# Experimental range resynthesis

This is an **offline prototype**, not the VOCALOID engine and not yet an
OpenUtau renderer. No change is made to the installed singer by this command.

It generates a donor with the existing Tokine DS acoustic model in A3–C5,
then extracts WORLD spectral envelope and aperiodicity. WORLD synthesis uses
the requested F0 with those donor characteristics. The high mix rises smoothly from D5 (0%) to F5 (100%):
D#5 is about 26% and E5 about 74%. F#3- remains the low target.
Equal-power mixing and 20 ms inward edge fades are used. Unprocessed samples
are exactly preserved within each run. Selection currently follows continuous
F0, not the editor's discrete note names; vibrato across a threshold can switch
branches and needs listening review before production integration.

Existing vocoder EQ, gain and saturation are bypassed for this controlled
comparison. The bank's acoustic sustained-vowel processing is retained.
This is neural donor synthesis plus classical resynthesis, not a diphone
recording database. Large shifts can alter naturalness; measured F0 accuracy
is not proof of voice identity or intelligibility.

## Run

Tested with Python 3.10, numpy 1.26.4, onnx 1.22.0, onnxruntime 1.23.2,
pyworld 0.3.4, soundfile 0.14.0 and PyYAML. WORLD uses a modified BSD license;
the PyWorld wrapper uses MIT. Dependencies are not bundled here.

```sh
python tools/range_resynthesis.py BANK INPUTS.npz NEW_OUTPUT_DIRECTORY
python tools/check_range_render.py INPUTS.npz NEW_OUTPUT_DIRECTORY
```

NPZ arrays: `tokens` and `durations` int64 `[1,N]`, `f0` float32 `[1,T]` Hz;
`durations.sum() == T`. Names prefixed `dbg_` are also accepted from the local
OpenUtau diagnostic export. Input must use this bank's phoneme IDs and the
44.1 kHz / 512-sample hop. The pitch checker uses Tokine's vowel token IDs.

Outputs include raw-float original, donor, WORLD, hybrid WAVs, parameter NPZ
and report JSON. Audition normalization must be kept separate from synthesis.
Source song inputs and generated song audio are deliberately not in this repo.

## Transition revision, 2026-10-10

The user liked the high-register sound but found the F5 switch too abrupt.
The revised smoothstep pitch weight starts at D5 and reaches one at F5.
Equal-power waveform mixing avoids the expected power dip of a linear mix
between independent phases; perceptual continuity still needs listening review.
The exact same original/WORLD waveforms were reused for the high A/B comparison.
Zero-weight samples remain bit-identical to the baseline; fully wet samples
remain bit-identical to WORLD. The weight is clipped to [0,1] to avoid
floating-point overshoot in square roots.

Low comparisons now use a phrase at E3–G#3 and a five-note a scale:
E3, F3, F#3, G3, A3. Audition copies of low A/B pairs are RMS matched;
raw synthesis files retain their original levels. This revision is offline
only and has not changed the production OpenUtau singer.

## Initial verification, 2026-10-10

On interior vowel frames, WORLD Harvest estimated:

| Test | Original median / p90 error (cents) | Hybrid median / p90 | Frames over 50 cents, original / hybrid |
| --- | --- | --- | --- |
| Native OpenUtau high-register phrase | 1.18 / 2.20 | 1.59 / 5.00 | 0 / 0 |
| Same phrase, two octaves lower | 4.00 / 21.24 | 3.22 / 8.62 | 4 / 0 |
| Eight isolated a notes, C3 to C6 | 2.78 / 12.19 | 2.12 / 7.57 | 5 / 0 |

The scale has 400 selected vowel frames. Original: 346 detected voiced frames;
hybrid: 400. Errors exclude undetected frames, so detection counts matter.
The scale notes are C3, F#3, G3, A4, E5, F5, A5 and C6; this does not establish
quality at every intervening note. All tested outputs are finite, below full
scale, equal length and bit-identical to their baseline outside the target mask.

Subjective pronunciation/identity and production integration remain unverified.
Do not label the production voicebank as using this prototype.

## Primary references

- [Kenmochi, 2010, VOCALOID and Hatsune Miku phenomenon in Japan](https://www.isca-archive.org/intersinging_2010/kenmochi10_intersinging.pdf): diphone concatenation and frequency-domain pitch/timbre processing. Our algorithm is different.
- [WORLD official implementation](https://github.com/mmorise/World): F0, spectral envelope, aperiodicity analysis and synthesis.
