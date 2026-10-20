# ts articulation

The local 1.3.0-rc.1 revision resynthesizes `ts` through existing learned `t`
and `s` phonemes. It changes acoustic inference, not the training weights or
the kana dictionary. This is a listening candidate, not a claim that every
pronunciation has been perceptually validated.

## Scope

- `ts` immediately followed by a vowel, with at least two acoustic frames.
- Inside a separate neural pass, split its existing span into `t` (one third
  of the span, rounded down, minimum one and maximum three frames) and `s`
  (the remainder).
- Use the resulting raw spectra only within the original `ts` span, in both
  the original and high-register donor branches. Existing vowel processing,
  including the approved short-vowel resynthesis, remains separate.
- Keep the score, external phonemes, vowel onset, F0, duration model, vocoder,
  and learned initializer tensors unchanged. Do not add a gain boost or
  synthetic noise burst.

The additional acoustic pass increases rendering cost. One-frame `ts`, a
terminal `ts`, and `ts` without an immediately following vowel are bypassed.
This also covers dictionary aliases using `ts`, such as `つぁ`.

## Build and checks

```sh
python ts_affricate.py CURRENT_ACOUSTIC OUTPUT_DIRECTORY
python check_ts_affricate.py CURRENT_ACOUSTIC OUTPUT_DIRECTORY/acoustic.onnx checks.json
```

The input must contain the current short-vowel-resynthesis graph. The helper
checks 1,488 token/duration/context cases, preservation of phone boundaries,
unchanged learned tensors, and bit-identical non-target mel selection with
fixed tensors. Independent diffusion/WORLD draws in separate native renders
must not be mistaken for a controlled before/after test of every waveform
sample.

Local native OpenUtau tests use isolated and connected `ts/u` at F3, C4, C5
and G5, with `s/u` controls. Inspect finite samples, peak headroom, identical
external phone timing, and the actual attack waveform. Listening acceptance
is a separate check.
