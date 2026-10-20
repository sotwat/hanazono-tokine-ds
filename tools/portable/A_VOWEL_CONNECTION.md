# Long vowel → /a/ connection trial (rejected)

2026-10-20 listening result: the user heard no improvement, calling this the same as the earlier attempt. Do not adopt this trial or use the numerical mel differences as evidence of perceptual improvement.

The user rejected +4 dB onset gain as too strong and suspects a connection fault. This experiment changes the spectral connection, with no new waveform gain, EQ, saturation or sustain resynthesis changes.

## Observed mechanism

Production acoustic SHA256: `807a109b5588497dae6b7644b041b009655fd73a2938ce761066f731496171cf`.

The existing general-vowel layer replaces the incoming onset with one spectral frame selected from 25–65% of that vowel. It holds this fixed spectrum, then fades back to the base spectrum over up to 24 frames. In the native「き→あ→か」fixture, the /a/ has 52 frames. The first 12 frames of the connection output are exactly the same spectrum; the fade then returns into a weak segment. Log summed mel amplitude in one diagnostic draw falls from 1.99 to -0.36, then rises to 2.34 later in /a/. The final short-resynthesis blend reduces but does not remove that contour. These are diagnostic observations, not proof of every audible cause.

## Trial

For /a/ after e/i/o/u, with a previous vowel of at least 40 frames and current /a/ of at least 6 frames, replace only the old connection window with the moving base spectrum. Advance its transition by at most two frames (23.22 ms) at the phone boundary, smoothly returning to the original time axis within six frames. The replacement window follows the old join extent: five frames before the boundary, through less than min(half the vowel, 24 frames) after it.

This removes the fixed-spectrum hold and return for these connections. Repeated /a/ and consonant/rest onsets retain their existing path. Subsequent consonant joins and the short-neural sustain blend remain in place. The score, phone times, F0 and learned tensors are unchanged. Natural weakness within the base vowel may remain; this is not a volume-normalization solution.

```sh
python tools/portable/a_vowel_connection.py ACOUSTIC.onnx OUTPUT_DIRECTORY
python tools/portable/check_a_vowel_connection.py ACOUSTIC.onnx OUTPUT_DIRECTORY
```

## Verification and limits

- 6,144 previous/current token and duration cases: exact layer bypass outside its target window, finite output, exact boundary sample from two frames ahead, and unchanged learned initializer bytes.
- Identical-seed acoustic comparison on actual native OpenUtau input: only mel frames 124–148 change; control channels are bit-identical. All other mel frames, including the sustained middle/end and other phones, are identical. Vocoder noise seeds are also shared for the comparison.
- Native OpenUtau candidate render checks phone boundaries, sample count and errors separately. No score editing or installation is involved.
- Artifact: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-vowel-join-20261020`. Comparison is baseline → 0.5 seconds silence → trial, PCM24/44.1 kHz mono, no additional normalization.

This trial was rejected, and was never installed or published. The installed/local production banks remain unchanged. The remaining long-note distortion is a separate unresolved issue.
