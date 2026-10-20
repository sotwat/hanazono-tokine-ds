# ts timing

The local v1.3.0-rc.1 duration revision limits ts anticipation to seven
acoustic frames (81.27 ms at 44.1 kHz / hop 512). It preserves shorter
predictions. The prior first-group bound allowed sixteen frames, about
186 ms, which was excessive in the reported ANIMA phrase.

The initial group uses absolute frame predictions. Later groups are
normalized to the note interval, so the cap is converted into the group's
prediction scale. Removed ts time is returned proportionally to the vowels
in that group. Group sums and vowel anchors remain unchanged; initial SP
is unchanged. Other consonants retain their original predictions. Manual
phoneme overrides in the editor still take precedence.

The acoustic ts articulation pass remains installed. This revision changes
timing, not its learned stop/fricative spectra or the vocoder.

Reproduce with the DiffSinger Python environment:

```sh
python tools/ts_timing.py BANK/dsdur/tokine_song_variance_v1.dur.onnx candidate.onnx
python tools/check_ts_timing.py BANK candidate.onnx checks.json
```

Checks cover 1,296 real-model conditions: ts plus three consonant controls,
six vowel IDs, nine group lengths, three pitches, and two alignment contexts.
They establish finite output, unchanged non-ts predictions, the normalized
cap and preserved group sums. Native OpenUtau AU rendering of the reported
phrase changes ts from 185.88 to 80.96 ms; all following phone positions
and lengths are exactly equal. The waveform is finite and does not clip.
The UI's regenerated phone strip also shows the shortened ts. Listening
acceptance of this timing change remains pending; no public release was
updated.
