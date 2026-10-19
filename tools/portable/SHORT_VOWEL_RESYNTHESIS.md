# Short-vowel resynthesis

This replaces fixed-spectrum sustain with a second neural prediction at a shorter vowel duration. On 2026-10-20 the user preferred the full-context trial and explicitly requested installation in the production voicebank. The exact candidate was installed locally; the public Release has not been updated. This listening result does not establish quality across every song or pitch. Do not interpret non-clipping, timing checks or a noise metric as proof of perceptual quality.

```sh
python tools/portable/short_vowel_resynthesis.py acoustic.onnx candidate
python tools/portable/check_short_vowel_resynthesis.py candidate
```

The experiment uses the current phoneme IDs for a/e/i/o/u. Only vowels of at least 40 frames with voiced F0 above 60 Hz and a pitch span below approximately half a semitone are shortened to 24 frames for the second prediction. The pitch sequence is mapped to that shorter interval. The result is interpolated back to the original phone boundaries and blended using the existing sustain replacement mask. The original prediction and processing remain the baseline elsewhere. No lyric, song, note index or specific pitch is hardcoded.

The score, external F0, duration model, trained weights, vocoder and waveform processing are unchanged. The second prediction shares the learned initializers and increases inference work. Noneligible phones and zero sustain weights bypass the added mixture exactly. Compressed prediction can still alter vowel colour or texture: the same speaker and unchanged weights do not establish identical perceived timbre.

Checks cover 96 length/phone/pitch conditions, zero-length phones, interpolation endpoints and exact bypass. Full-context rendering through the installed OpenUtau renderer retains the target phrase's phone JSON and 130560 samples at 44.1 kHz. Private test scores, weights, fixtures and audio remain outside this repository.

The previous cycle-averaging experiment was rejected because it changed the voice at the sustained vowel. This approach adds neither cycle averaging nor vibrato, and does not amplify the decayed output. Retain full-context listening comparisons when changing this method.
