# /a/ onset gain trial — listening approval pending

2026-10-20: isolate the user's weak /a/ attack complaint from the unresolved long-note distortion. The rejected full-interior neural resynthesis remains disabled. This is an uninstalled trial; the production acoustic hash remains `807a109b5588497dae6b7644b041b009655fd73a2938ce761066f731496171cf`.

```sh
python tools/portable/a_vowel_attack.py ACOUSTIC.onnx VOCODER.onnx OUTPUT_DIRECTORY
python tools/portable/check_a_vowel_attack.py OUTPUT_DIRECTORY
```

The acoustic addition only marks /a/ immediately following a/e/i/o/u in unused control channel 7. Existing mel computation and trained initializers are unchanged. The vocoder addition applies a smooth scalar gain after existing processing, with no normalization of the other notes. Phones must last at least 120 ms. Gain rises over 5 ms, holds to 25% of the phone (at most 140 ms), then returns to unity by 45% (at most 240 ms). The limit is +4 dB, reduced when the phrase lacks headroom below 0.97 sample peak. Silence, consonant-to-/a/, and the sustained middle/end are outside this trial. Pitch, duration and spectral synthesis are unchanged.

## Verification

Artifact: `/Users/aaa/Documents/DiffSinger/artifacts/tokine-a-attack-20261020`.

- 1,024 previous/current token combinations validate the channel-7 mask.
- 72 waveform conditions validate exact bypass, phase/duration preservation, gain bounds, short-note bypass and headroom handling.
- Fixed-wave comparisons use the *same* baseline native OpenUtau output for both halves, avoiding diffusion draw differences.「き→あ→か」and「あの日あのとき」comparisons change 10,583 and 5,529 samples respectively. Maximum gain is 4 dB; all outside-window samples are bit-identical before 24-bit export. The first half is baseline, followed by 0.5 seconds of silence, then the trial. No extra normalization.
- Both candidate ONNX files retain all existing initializer bytes. A separate unregistered candidate bank allows native integration testing. The installed bank and running OpenUtau remain unchanged.

This creates a more prominent onset; it does not establish perceptual success, fix the full crescendo contour, or solve residual sustained-vowel distortion. The user's listening judgement is still required before adopting the trial.
