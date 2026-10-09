# Steady sustain (local v1.2.2-rc.1)

The user reported a vibrato-like sound on a sustained syllable with vibrato
length set to zero. The inspected score has a flat pitch segment and no pitch
curves. WORLD analysis of the rendered vowel found nearly constant F0; changing
spectra and an amplitude dip/recovery were investigated as possible causes.

`stabilize_sustain.py` replaces the periodic scan through the selected donor
window with a fixed mean of five log-mel samples across that window. Donor
selection, phrase ending, adjacent-vowel morphing, and the F0 input are retained.
This rule applies to eligible sustained phonemes, not a particular lyric.

Constant spectra alone left the dip/recovery largely unchanged. The retained
version also advances the sustain onset for 40–59-frame phones from frame20 to
12 (fade8 unchanged), and for 60–171-frame phones from frame32/fade12 to
frame16/fade10. Phones shorter than40 frames and stop consonants remain excluded.
For longer phones, onset/release timing remains unchanged.

```sh
python tools/stabilize_sustain.py approved-v1.2.1.onnx steady.onnx
python tools/check_steady_sustain.py approved-v1.2.1.onnx steady.onnx checks.json
```

14 branch/case checks passed: the plateau is constant, output is finite, and
frames outside the eligible sustain window are exactly preserved with identical
raw input. OpenUtau AU rendered the same excerpt with both banks. Over the
1.30–1.75 s vowel interval, 25 ms RMS windows at 10 ms spacing had a coefficient
of variation of 0.2866 before and 0.1188 after. Mean RMS was 0.0851/0.0895;
median measured F0 was 349.20/349.17 Hz. Both outputs were130560 samples at44100 Hz;
new peak0.33948, finite and unclipped. These native renders include independent
diffusion draws; the measurements do not establish a definitive perceptual cause.

The local candidate is installed for listening. Public v1.2.1 is unchanged.
User approval of the revised sound is pending.

Acoustic SHA256:
`1ab0c4175823d88c876366cad7e17343694f7c7206bb60172ae6012036cb3135`
