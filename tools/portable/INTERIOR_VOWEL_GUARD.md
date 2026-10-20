# Exclusive short-vowel sustain and stable interior guard

The short-vowel resynthesis guard previously required the entire sustained
vowel to stay within approximately half a semitone. A short onset or release
transition therefore excluded an otherwise steady sustained vowel, leaving
the older fixed-spectrum repair active. The reported D4 /a/ in バックアップ
has 87 frames and a 197.30-cent whole-phone span, but a constant central pitch.
Its old short-vowel eligibility was zero while the sustain mix reached one.
This was one bypass case, not a general explanation of the reported distortion.
The user also reports the same mechanical distortion on steady-pitch sustains.

A separate common mixing defect retained fixed-spectrum sustain underneath
short-vowel resynthesis. At weight w, the final mixture still contained
w*(1-w) of that spectrum. The revised graph zeros the old wet contribution
only where short-vowel resynthesis is eligible, before the existing connection
processing. Thus the later restore blends the unheld baseline with the short
neural prediction once. Both original/donor branches use this change. All other
phones retain their existing path. This does not prove every remaining vocal
artifact comes from this mixture.

The revised guard measures pitch stability in the central 60% of the phone,
excluding at least six frames from each end. The half-semitone threshold,
40-frame minimum, whole-phone voiced floor above 60 Hz and existing sustain
mixture remain. Ongoing vibrato and glides that move through the interior
still bypass. The original F0 is resampled for the compressed neural pass;
the external synthesis F0, note lengths, interpolation endpoints and phone
boundaries are unchanged. This extends the previously accepted short-vowel
method, without a new waveform filter, gain boost or added vibrato.

Use the current bank with short-vowel resynthesis already installed:

```sh
python interior_vowel_guard.py PRIOR_ACOUSTIC OUTPUT_DIRECTORY
python check_interior_vowel_guard.py PRIOR_ACOUSTIC OUTPUT_DIRECTORY PRIVATE_FIXTURE_NPZ
```

The diagnostic fixture contains native renderer input and the old guard
output. It is private and is not distributed in this repository.
288 conditions cover steady pitch, attack/release transitions, continuous
glides, vibrato, unvoiced pitch, phone types and lengths. The prior 96 map /
restore checks also pass, including exact zero-weight bypass and boundary
preservation. Existing learned initializer tensors remain byte-identical.
Four identical-input routing cases verify that eligible fixed-spectrum wet
contributions are zero and ineligible contributions bypass bit-exactly.
These checks establish the routing and guard behavior, not perceptual distortion removal
or identical perceived timbre. Full-context before/after listening is needed.

Local verification on 2026-10-20 used the installed OpenUtau AU renderer:
36 steady-pitch sustained-note cases (た/で/ま/の/す/き, F3/D4/F5,
0.75/2 seconds) rendered before and after. Phone JSON and sample counts
matched for every pair; all outputs were finite and below full scale
(after maximum peak 0.362392). The reported バックアップ phrase also retained
its phone JSON and 184832 samples, with after peak 0.282698. These are
functional checks, not listening approval. Independent diffusion draws
prevent interpreting all waveform differences as the patch effect.

Continuous interior glides and wide vibrato still follow their prior path.
Nasal N is not part of this vowel replacement. Their naturalness and all
other possible distortion sources remain unverified. Do not describe this
as universal distortion removal.
