# Sustained vowel to fricative connections

`smooth_fricative_connections.py source.onnx output.onnx` extends the steady
sustain model. It requires the existing `su_hold_offsets` and vowel connection
nodes. NumPy and ONNX are required; validation also uses ONNX Runtime.

Both original and donor branches interpolate log-mel spectra with smoothstep
between eight frames before and two frames after an eligible fricative onset.
The nine interior frames change. With the 512/44100 hop, the endpoints are about
93 ms before and 23 ms after onset.

Eligibility is a/e/i/o/u lasting at least 40 frames, followed by f/h/s/sh/z
lasting at least four frames. Token IDs are specific to this bank. This is a
shared acoustic processing rule, with no song, lyric, or score-position lookup.
Phone timing, F0, learned weights and vocoder are unchanged.

Run `check_fricative_connections.py before.onnx after.onnx results.json` to
verify all 25 vowel/fricative pairs and six exclusion cases in both branches
(62 cases). Checks cover the interpolation formula, finite values and exact
preservation outside eligible windows.

## Local verification, 2026-10-10

OpenUtau AU rendered the same phrase with steady sustain and this modification.
The final rendered mel matched the interpolation formula exactly. Maximum
adjacent mel distances in the transition fell from 10.191/9.467 to 4.621/4.723
in the original/donor branches. Both waveforms had 130560 samples at 44100 Hz,
finite values and no clipping.

For the vowel-to-f decay at 1.70–1.82 s, the largest RMS fall was
11.106 -> 5.661 dB per 10 ms (20 ms RMS windows). The broader 1.70–1.86 s
absolute-change statistic, including the following rebound, did not improve:
12.266 -> 13.957 dB. These renders contain independent diffusion draws; the
measurements do not establish perceptual quality for every connection.

An initial amplitude-domain interpolation was rejected because the abrupt
decay remained. The adopted local candidate uses log-mel interpolation and is
named v1.2.2-rc.2 子音接続. Listening approval is pending. The public v1.2.1
release is unchanged. Recordings, scores and comparison audio remain local.
