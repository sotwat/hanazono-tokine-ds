# Short phone connections

The existing class-aware acoustic joins only covered preceding sounds of
at least 40 frames. In the current Charles score, F5 `te → sa` has a nine-frame
`e`, so it received no join. This change extends the same rule to sustained
phone classes of at least four frames (about 46 ms), including short vowels.
It does not move phone boundaries or change F0, the duration models or the
paired WORLD vocoder.

The left window is limited to one quarter of the preceding phone, rounded
down, and at most one frame (two for fricatives/affricates). The right window
is at most half of the following consonant minus its final frame. Thus short
consonant interiors remain available; SP, zero/one-frame consonants and
preceding sounds shorter than four frames bypass this rule. There is no join
across a rest. Existing joins after 40-frame sounds are bit-identical.

```sh
python tools/short_phone_connections.py CURRENT_ACOUSTIC CANDIDATE_ACOUSTIC
python tools/check_short_phone_connections.py CURRENT_ACOUSTIC CANDIDATE_ACOUSTIC checks.json
```

Checks: 68,040 fixed-spectrum cases across both acoustic branches, all 13
sustained phone classes, nonsustained controls, 27 next-phone types, 12 prior
lengths and seven consonant lengths. 15,912 cases changed; all outputs were
finite and differences remained inside the allowed boundary windows.

Using one cached OpenUtau acoustic draw for the reported phrase, the `e → s`
adjacent log-mel distance changed from 5.102 to 4.416 in the original branch
and 7.151 to 6.002 in the donor branch. The waveform is 77,312 samples at
44.1 kHz, finite and unclipped. WORLD noise draws differ between waveform
renders, so waveform differences do not isolate only the join. Numeric
continuity does not establish that the user's perceived noise is gone;
listening confirmation is still required.

The installed bank was subsequently rendered in OpenUtau AU. Both mel
branches matched the three interpolated `e → s` frames to within 2e-6.
The native WAV has the same 77,312 samples and is finite and unclipped.
The user's saved score remained byte-identical to the captured current score.
