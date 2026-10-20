# Long vowel join timing

The user approved the exclusive short sustain revision: the reported distortion
was gone. They then heard the incoming /i/ in バックアップ「期待」 about 30 ms
late. The native phoneme boundary itself was correct (6333.327 ms).

The general vowel onset morph completed two 512-sample frames after that
boundary. This is a plausible source of the perceived delay, not a measured
human vowel recognition threshold. This revision moves that morph window two
frames earlier after vowels lasting at least 40 frames. The width and interpolation
curve are retained. It reaches the incoming vowel anchor at the boundary rather
than 23.22 ms later. Both original and donor branches receive the same change.

Scope: existing vowel-to-vowel eligibility, a/e/i/o/u, incoming vowel at least six
frames. Shorter outgoing vowels, consonants and rests keep their old path.
This is not a universal 23 ms shift of phonemes. Durations, external F0, trained
initializers, vocoder and the accepted sustain algorithm remain unchanged.

## Reproduce

Use the approved exclusive-sustain acoustic model as BEFORE, an unused AFTER
path, and Python with onnx/onnxruntime/numpy installed:

```sh
python tools/portable/long_vowel_join_timing.py BEFORE AFTER
python tools/portable/check_long_vowel_join_timing.py BEFORE AFTER REPORT.json
```

The identical-input graph test checks 1,790 cases across both branches: all 25
vowel pairs, outgoing durations 4/39/40/87/250 and incoming 1/5/6/8/90 frames,
plus all other token IDs on either side. It checks completion at the boundary,
bit-exact bypass in 1,340 controls, unchanged output outside the local window,
finite output and byte-identical existing initializers.

Native AU renderer validation and comparison audio are saved locally under
`DiffSinger/artifacts/tokine-sustain-joins-20261020`. The actual target preserves
all phone positions/durations and 184832 output samples. Thirty connection
controls cover all vowel pairs and following ta/chi/tsu/ha/N; they compare native
phone timing, sample counts and finite, unclipped output. Independent diffusion
renders are not evidence of bit-exact acoustic isolation or perceptual success.
The user's listening approval of the previous distortion fix does not establish
listening approval of this timing revision. Public releases are not changed.
