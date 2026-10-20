# Short connected vowel trial — 2026-10-20

Status: audition candidate only. Not installed and not published.

The user hears an extra breath/catch in `ひ→あ` within `あのひあのとき`.
The actual /i/ is 14 acoustic frames; the existing long-vowel handoff
requires at least 20. Its tail also loses mel amplitude before /a/ rises.
This is a diagnostic observation, not proof of the entire perceptual cause.

`short_vowel_link.py` bridges directly adjacent a/e/i/o/u phones when the
previous phone has 6–19 frames and the next at least 6 frames. It replaces
only boundary−3 through boundary+1, interpolating linear mel amplitudes
between boundary−3 and boundary+2 with a smoothstep weight. This is five
frames (58 ms at 512/44100). It preserves the explicit consonant, control
plane, pitch, score, existing volume balancing and long-vowel processing.
An explicit rest/consonant between vowels bypasses this layer. It cannot
identify whether any particular breath component is intentional.

Build against the installed level-handoff acoustic model (SHA256
`3f9c77fe72582286ab2eeadc903b504ca08c52a0a50017185beb5073603153c5`):

```
python tools/portable/short_vowel_link.py SOURCE.onnx NEW_OUTPUT_DIRECTORY
```

Local evidence: `DiffSinger/artifacts/tokine-vowel-link-fine-20261020`.
Build3 is the working candidate; build/build2 failed export and are unused.
1024 token-pair cases plus two dynamic zero-duration/multiple-boundary cases
passed ONNX/Torch agreement, finite values and exact outside-window bypass.
All 627 previous initializers are byte-preserved.

The actual cached OpenUtau mel was used for a controlled comparison. Only
frames 83–87 in the original/donor branches change; control is identical.
The same seeded, installed vocoder produces 165888 samples in each version,
peak .40465/.40407, finite and unclipped. No added output normalization.
The comparison contains full phrase context, before, 0.5 s silence, after.
Numerical differences do not establish perceptual improvement; user audition
is pending. This trial does not establish all short-vowel pairs are natural.

Native OpenUtau candidate rendering with DIFFS JA also completed with no
errors and exactly matching phone metadata and sample counts. The original
project and both installed banks remain unchanged.
